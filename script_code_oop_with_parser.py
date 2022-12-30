import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from functools import reduce
from pyod.models.abod import ABOD
from sklearn.preprocessing import MinMaxScaler
import argparse
#clone YOLOv5
#!git clone https://github.com/ultralytics/yolov5
#%cd yolov5
#%pip install -qr requirements.txt
#import torch
#from IPython.display import Image, clear_output  # to display images


# Defining a function to return all the arguments we added to the parser defined below and then return them
def arguments():
    # ArgumentParser will hold the necesssary information to show in the command line
    parser = argparse.ArgumentParser(description='Water Control - Chan Diagnostic Plots Recognizer')
    parser.add_argument('--action_required', type = str, help = 'Type of Command to be executed',
                        choices = ['infer_chan_plots', 'plot_chan', 'get_clean_dataset','arrange_dataset',
                                   'pre_process_dataset', 'read_dataset'], default = 'plot_chan')
    parser.add_argument('--file_directory', type = str, help = 'The location of the excel sheet on your local machine', required = True)
    parser.add_argument('--outlier_features', type = str or list, help = 'Features used to remove outliers accordingly', default = ['qo_bpd', 'qw_bpd'])
    parser.add_argument('--main_feature', type = str, help = 'Feature used for visualization against production time', default = 'qo_bpd')
    parser.add_argument('--min_prod_months', type = int, help = 'Minimum number of producing months below which a well performance would be ignored', default = 12)
    parser.add_argument('--plot_outliers_option', type = bool, help = 'To visualize the well after removing outliers or not', default = False)
    # All the arguments can be later called by using arguments.
    args, _ = parser.parse_known_args()
    return (args)


def create_missing_columns(df):
    # In case the online days in month are not given, we assume they produce everyday
    if 'production_online_days' not in df.columns:
        df.loc[:, 'production_online_days'] = df.loc[:, 'report_date'].dt.days_in_month
    # Daily Rates Calculations assuming monthly productions are the only givens
    if 'qo_bpd' not in df.columns:
        df.loc[:, 'qo_bpd']    = df.loc[:, 'monthly_produced_oil'] / df.loc[:, 'production_online_days']
    if 'qw_bpd' not in df.columns:
        df.loc[:, 'qw_bpd']  = df.loc[:, 'monthly_produced_water'] / df.loc[:, 'production_online_days']
    if 'qg_mscfd' not in df.columns:
        df.loc[:, 'qg_mscfd']    = df.loc[:, 'monthly_produced_gas'] / df.loc[:, 'production_online_days']
    if 'ql_bpd' not in df.columns:
        df.loc[:, 'ql_bpd']  = df.loc[:, 'qo_bpd'] + df.loc[:, 'qw_bpd']
    # Other Caclucations
    if 'wor' not in df.columns:
        df.loc[:, 'wor'] = df.loc[:, 'qw_bpd'] / df.loc[:, 'qo_bpd']
    if 'wc_fraction' not in df.columns:
        df.loc[:, 'wc_fraction']= df.loc[:, 'qw_bpd'] / df.loc[:, 'ql_bpd']
    if 'gor' not in df.columns:
        df.loc[:, 'gor']= df.loc[:, 'qg_mscfd'] * 1000 / df.loc[:, 'qo_bpd']

    return (df)

def clean_data(df):
    # to avoid division by zero
    cols = ['qo_bpd', 'qw_bpd', 'qg_mscfd']
    df[cols] = df[cols].astype(np.float64)
    # Dropping duplicated rows
    df.drop_duplicates(inplace = True)

    # Converting dates from strings into date time series
    df['report_date'] = pd.to_datetime(df['report_date'])
    # To avoid any date duplicated by mistake as entered by two different days on the same month
    df = df[df['report_date'].dt.day == 1]

    # Adding a feature for commingled reservoirs
    if 'reservoir_name' in df.columns:
        df['well_reservoir_name'] = df['well_name'] + '_' + df['reservoir_name']
    else:
        df['well_reservoir_name'] = df['well_name']

    # Dropping_duplicates for wells that have duplicated production date for the same reservoir
    df.drop_duplicates(subset = ['report_date', 'well_reservoir_name'], inplace = True)

    # Dropping columns that are no more in use
    try:
        df.drop(['monthly_produced_oil', 'monthly_produced_water', 'monthly_produced_gas'],
                axis = 1, inplace = True)
    except KeyError:
        pass
    try:
        df.drop(['well_name'], axis = 1, inplace = True)
    except KeyError:
        pass
    try:
        df.drop(['reservoir_name'], axis = 1, inplace = True)
    except KeyError:
        pass

    df_preprocessed = df.copy()

    return (df_preprocessed)

def processing_well_data(df_pre_processed, wellname):
    '''
        INPUTS: wellname: well to extract data of
        OUTPUT: dataframe for the well sorted and chan plot parameters calcualted
                                                                                                                    '''
    df = df_pre_processed.copy()
    # Sorting data by reporting dates
    well = df[df['well_reservoir_name'] == wellname].sort_values(by = 'report_date').reset_index(drop = True)
    # Total days of production
    well.loc[:, 'actual_days_of_production'] = (well.loc[:, 'report_date'] - well.loc[0,'report_date']).apply(lambda x: x.days)
    # Normalized days of production assuming all months of 30 days to match all wells on that column
    well['producing_days'] = ((well.loc[:, 'actual_days_of_production'].diff() // 28) * 30).cumsum()
    # Calculating derivative of WOR
    well["wor'"] = well.loc[:, 'wor'].diff() /  well.loc[:, 'actual_days_of_production'].diff()

    # Setting the first day of production as day number one
    well.loc[0, 'producing_days'] = 1
    # Setting name of online producction days
    well.rename(columns= {'production_online_days': 'online_days'}, inplace = True)

    # Cumulative calculations
    well['cumulative_oil_production'] = (well['qo_bpd']*well['online_days']).cumsum()

    # Setting days as index to match on
    well.set_index('producing_days', inplace = True)
    # Dropping unnecessary columns of deltas calculated
    well.drop(['actual_days_of_production'], axis = 1, inplace = True)

    # Setting top level column multiindex of wells names
    well.columns = pd.MultiIndex.from_product([[wellname], well.columns])

    well_processed = well.copy()
    return(well_processed)

def outliers_algorithm(df_wrangled):
    '''
        INPUT:
            df_wrangled: Scaled wrangled dataframe that only includes features to be used for outliers removal
        OUTPUT:
            preds: predictions either 0s or 1s
                                                                                                    '''
    df = df_wrangled.copy()
    ### Outliers Removal Algorithms
    model = ABOD(n_neighbors = 6, contamination = 0.15)

    # fitting models
    model_fit = model.fit(df)
    scores = model_fit.decision_scores_
    preds = model_fit.predict(df)
    return(preds)

def plotting_parameters(df_wrangled, original_days, preds):
    '''
        INPUT:
            df_scaled: original dataset scaled
            original_days: days not scaled
            preds: predictions either outliers or inliers
        OUTPUT:
            unique_clean_days: days that are classified as clean
            unique_outlier_days: days that are classified as outliers
                                                                                                                '''

    # Making a copy of the dataframe to avoid overriding new columns
    df_new = df_wrangled.copy()
    # df_new is the same as df_plot_full_cols but NaNs are filled with zeros in case features other than Qo to be used
    # in the model # Adding original days to avoid it being considered into the algorithm
    df_new['original_days']= original_days

    # clean indices, clean dataframe scaled, and clean days
    clean_ind = np.where(pd.DataFrame(preds) == 0)[0].tolist()
    df_plot_clean_unique = df_new[df_new.index.isin(clean_ind)]
    unique_clean_days = df_plot_clean_unique['original_days']

    # outlier indices, outlier dataframe scaled, and outlier days
    outlier_ind = np.where(pd.DataFrame(preds) == 1)[0].tolist()
    df_plot_outliers_unique = df_new[df_new.index.isin(outlier_ind)]
    unique_outlier_days = df_plot_outliers_unique['original_days']

    return(unique_clean_days, unique_outlier_days)

def plot_inliers_vs_outliers(df, well_name, main_feature, df_plot_full_clean, df_plot_full_outlier):
    '''
        INPUT:
            df: original dataframe
            well_name: the proposed working well
            df_plot_full_cols: the full dataframe of the proposed well
            unique_clean_days: clean indices
            unique_outlier_days: outlier indices
        OUTPUT:
            df_plot_full_clean: clean dataset of the well
            Three plots with full data, inlier data, outlier data
                                                                                                                    '''

    # Plotting axes
    x_col_plot = 'producing_days'
    y_col_plot = main_feature
    # Full table for original scatter plotting that includes the oiriginal data imported from excel sheet
    df_full = df.copy()
    df_full = df_full[well_name].reset_index()
    # Plot section
    fig = plt.figure(figsize = (25,5))
    plt.suptitle(well_name)
    ax = fig.add_subplot(1,3,1)
    plt.scatter(x = df_full[x_col_plot], y= df_full[y_col_plot])
    plt.title('Original Plot Imported')
    plt.xlabel(x_col_plot)
    plt.ylabel(y_col_plot)

    ax1 = fig.add_subplot(1,3,2, sharex = ax, sharey = ax)
    plt.scatter(x = df_plot_full_clean[x_col_plot], y= df_plot_full_clean[y_col_plot], color = 'g', label = 'model_inliers')
    plt.scatter(x = df_plot_full_outlier[x_col_plot], y= df_plot_full_outlier[y_col_plot], color = 'red', label = 'model_outliers')
    plt.legend()
    plt.title('Original Plot Classified')
    plt.xlabel(x_col_plot)
    plt.ylabel(y_col_plot)

    ax2 = fig.add_subplot(1,3,3, sharex = ax, sharey = ax)
    plt.scatter(x = df_plot_full_clean[x_col_plot], y= df_plot_full_clean[y_col_plot], color = 'g', label = 'inliers')
    plt.title('Clean Plot Output')
    plt.xlabel(x_col_plot)
    plt.ylabel(y_col_plot)

class Chan_plots():
    '''
        INPUTS:
            * file_directory: path of excel file that contains data
            * outlier_features: features used to remove outliers accordingle, default = 'qo_bpd', type{str, list}
            * main_feature: feature used for visualization against production time, default = 'qo_bpd', type{str}
            * min_prod_months: minimum number of producing months below which a well performance would be ignored, default = 12, type{int}
            * plot_outliers_option: Boolean to visualize the well after removing outliers or not, default = False

            - The input dataset should have all of the following columns:
              [well_name - report_date - qo_bpd/monthly_produced_oil - qw_bpd/monthly_produced_water]
            - Presence of [qg_mscfd/monthly_produced_gas - wor - wc_fraction - gor - reservoir_name - days] is optional
            - Dataset Definitions:
              1. well_name: name of the well
              2. report_date: reporting month of production
              3. qo_bpd / monthly_produced_oil: average daily oil production in bpd / monthly production in bbls
              4. qw_bpd / monthly_produced_water: average daily water production in bpd / monthly production in bbls
              5. qg_mscfd / monthly_produced_gas:average daily gas production in mscfd / monthly production in mscf
              6. wor / wc_fraction: water oil ratio / water cut
              7. gor: gas oil ratio scf/stb
              8. reservoir_name: name of reservoir
              9. days: online days on production of the month

        OUTPUTS:
            * clean dataset and optional plotting of wells before and after removing outliers
                                                                                                                     '''
    #def __init__(self, file_directory, outlier_features = 'qo_bpd', main_feature = 'qo_bpd',
                # min_prod_months = 12, plot_outliers_option = False):

    def __init__(self, file_directory = arguments().file_directory, outlier_features = arguments().outlier_features, main_feature = arguments().main_feature,
                 min_prod_months = arguments().min_prod_months, plot_outliers_option = arguments().plot_outliers_option):
        self.__file_directory = file_directory
        self.__outlier_features = outlier_features
        self.__main_feature = main_feature
        self.__min_prod_months = min_prod_months
        self.__plot_outliers_option = plot_outliers_option

    def read_data(self):
        # read dataframe from file directory
        df = pd.read_excel(self.__file_directory)
        return(df)

    def pre_process(self):
        '''
            This function is to extract a clean processed Dataset that is ready to be further used
                                                                                                                        '''
        # reading dataframe
        df = self.read_data().copy()
        # auto complete columns that the user did not include
        df = create_missing_columns(df)
        # clean the dataset and make it ready to use
        df = clean_data(df)
        return (df)

    def rearrange_dataset(self):
        '''
            This function is to extract full dataset re-arranged indexed with days of production
                                                                                                                        '''
        df = self.pre_process().copy()
        # unique well names considering commingled wells
        wellames = df['well_reservoir_name'].unique()
        # Setting empty list to append dataframes into
        wells = []
        for wellname in wellames:
            # to show progress
            print(str(wellname) + ' ' + 'is now being processed')
            # Pre-processing using a previous function
            well_data = processing_well_data(df, wellname)
            # Appening the output table
            wells.append(well_data)
        # Concating all datasets on same vertical column as all wells have the same days intervals and empty rows are NaNs
        df = pd.concat(wells, axis = 1)
        return(df)

    def extract_clean_data(self):
        '''
            This function returns clean dataset where outliers are removed
                                                                                                                    '''
        df = self.rearrange_dataset()
        clean_wells = []
        for i, well_name in enumerate(df.columns.get_level_values(0).unique().tolist()):
            # New dataframe for each well & replace inf
            df_well = df[well_name].replace([np.inf, -np.inf], np.nan)
            # keeping points with available oil production only
            df_well = df_well[df_well[self.__main_feature].notna()].reset_index()
            print('Processing well #',i, ' which is: ', well_name)
            if ((df_well['wor'].count() > self.__min_prod_months) and (df_well["wor'"].count() > self.__min_prod_months)):
                # Original production days
                original_days = df_well['producing_days']
                # the full dataset to be used in plotting that includes features not used in modelling
                df_plot_full_cols = df_well.copy()

                # Creating features to be used in outliers removal
                if type(self.__outlier_features) != list:
                    outlier_features_list = ['producing_days']
                    outlier_features_list.append(self.__outlier_features)
                elif type(self.__outlier_features) == list:
                    outlier_features_list = self.__outlier_features
                    outlier_features_list.append('producing_days')
                # Final dataset to be used in outliers removal
                df_well = df_well[outlier_features_list]

                # Scaling dataset with MinMax technique
                df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_well), columns = df_well.columns)
                # Final output
                df_wrangled = df_scaled.copy()

                ################################## CALLING OTHER FUNCTIONS #########################################
                # Calculating predictions
                preds = outliers_algorithm(df_wrangled)
                #Calculating inliers and outliers
                unique_clean_days, unique_outlier_days = plotting_parameters(df_wrangled, original_days, preds)
                # clean and outlier dataframes according to algorithm
                df_plot_full_clean = df_plot_full_cols[df_plot_full_cols['producing_days'].isin(unique_clean_days)]
                df_plot_full_outlier = df_plot_full_cols[df_plot_full_cols['producing_days'].isin(unique_outlier_days)]
                # Plotting outliers vs. inliers
                if self.__plot_outliers_option != False:
                    plot_inliers_vs_outliers(df, well_name, self.__main_feature, df_plot_full_clean, df_plot_full_outlier)
                ################################## CALLING OTHER FUNCTIONS #########################################

                df_plot_full_clean.set_index('producing_days', inplace = True)
                df_plot_full_clean.columns = pd.MultiIndex.from_product([[well_name], df_plot_full_clean.columns])
                clean_wells.append(df_plot_full_clean)
            else:
                continue

        clean_wells_df = reduce(lambda df_left,df_right: pd.merge(df_left, df_right,
                                                  left_index=True, right_index=True,
                                                  how='outer'), clean_wells)
        return(clean_wells_df)


    def plot(self):
        '''
            This function is to plot water-control diagnostic Chan plots and save them in same directory with well name
                                                                                                                            '''
        df = self.extract_clean_data()
        for i,well_name in enumerate(df.columns.get_level_values(0).unique().tolist()):
            df_well = df[well_name]
            # to hide plots and only save them
            plt.ioff()
            fig = plt.figure(figsize = (8,8))
            # Chan plot for the selected well
            plt.scatter(x = df_well.index.tolist(), y= df_well['wor'], label = 'wor', c = 'blue', alpha = 0.3);
            plt.scatter(x = df_well.index.tolist(), y= df_well["wor'"], label = "wor'", marker = 'h', c = 'red', alpha = 0.3);
            plt.xlabel('days')
            plt.ylabel("WOR - WOR'")
            # Transforming into log-log scale
            plt.xscale('log'), plt.yscale('log')
            # Setting limits and legend
            plt.xlim([1e0, 2e4])
            plt.xticks([1e0, 1e1, 1e2, 1e3, 1e4], ['1E0', '1E1', '1E2', '1E3', '1E4'])
            plt.ylim([1e-6, 1e4])
            plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], ['1E-6', '1E-5', '1E-4', '1E-3', '1E-2', '1E-1', '1E0', '1E1', '1E2', '1E3', '1E4'])
            #plt.axis('off')
            plt.grid()
            plt.title(well_name + ' ' + 'Chan Plot')
            plt.legend()
            plt.savefig('{}.png'.format(well_name));
            print(str(well_name) + ' ' + 'is diagnosed with Chan plots')

    def infer(self, plot_path, trained_model_path, min_confidence = 0.50):
        '''
            This function reflects the interpretation of water-control diagnostic plots
            INPUTS:
                - image_path: path to image
                - trained_model_path: path to the trained model .pt
                - min_confidence: the minimum confidence score to show show interpretation results
                                                                                                            '''
        #!python detect.py --trained_model_path --save-txt  --save_conf  --img 256 --conf min_confidence --source plot_path
        print('Please uncomment line#392 first then delete this line')

if __name__ == '__main__':
    # If conditions for the user-input data
    if arguments().action_required == 'infer_chan_plots':
        trained_model_path = input('Please Enter the Path to the Model: ')
        plot_path = input('Please Enter the Path to the Plot(s): ')
        min_confidence = float(input('Please Enter the Minimum Required Confidence Score: '))
        Chan_plots().infer(plot_path, trained_model_path, min_confidence)
    elif arguments().action_required == 'plot_chan':
        Chan_plots().plot()
    elif arguments().action_required == 'get_clean_dataset':
        Chan_plots().extract_clean_data()
    elif arguments().action_required == 'arrange_dataset':
        Chan_plots().rearrange_dataset()
    elif arguments().action_required == 'pre_process_dataset':
        Chan_plots().pre_process()
    elif arguments().action_required == 'read_dataset':
        Chan_plots().read_data()
