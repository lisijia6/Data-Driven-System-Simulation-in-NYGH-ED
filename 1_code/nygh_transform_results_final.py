import numpy as np
import os
import pandas as pd


def convert_row_str_to_nparray(df, column):
    """
    Converts a list of strings to an numpy array of floats

    @ params:
        df (pd.DataFrame): the DataFrame to process
        column (str): the column in which to process in the DataFrame

    @ return:
        column_values_return (numpy array): numpy array of floats
    """

    n = len(df)
    column_values_list = []

    for r in range(n):
        temp_list = [float(i) for i in df[column][r][1:-1].split(', ')]
        column_values_list.append(temp_list)

    column_values_return = np.array(column_values_list).transpose()
    return column_values_return


def read_results_single_sys_state(filename, sheetname, z_val, nruns):
    """
    Converts a list of strings to an numpy array of floats

    @ params:
        filename (str): the filename to read data from
        sheetname (str): the sheet in the file to read data from
        z_val (float): the z-value to calculate the standard error
        nruns (int): specifies the number of replications in simulation

    @ return:
        mean (list): list of expected mean values for 3 patient types
        stderr (list): list of standard errors for the expected mean values for 3 patient types
        median (list): list of expected median values for 3 patient types
        percentile90 (list): list of expected 90th percentile values for 3 patient types
        cutdown (list): list of percentages to cut down the consult patients' LOS

    """

    df = pd.read_excel(filename, sheet_name=sheetname, engine='openpyxl', index_col=0)

    # Sort the DataFrame by cut down %, from smallest to largest
    df.sort_values(by=['Cut Down by (%)'], inplace=True)
    df = df.reset_index(drop=True)

    mean = convert_row_str_to_nparray(df, 'Mean [T123A, T123NA, T45]')
    stderr = None
    if sheetname != 'Naive':
        stdev = convert_row_str_to_nparray(df, 'Stdev [T123A, T123NA, T45]')
        stderr = stdev * z_val / np.sqrt(nruns).round(2)
    median = convert_row_str_to_nparray(df, 'Median [T123A, T123NA, T45]')
    percentile90 = convert_row_str_to_nparray(df, '90th Percentile [T123A, T123NA, T45]')
    cutdown = df['Cut Down by (%)'].values

    return mean, stderr, median, percentile90, cutdown


def save_to_df_by_patient_type(data, column_names, index_names, df_names):
    """
    Saves data to dataframes by patient type

    @ params:
        data (np.DataFrame): the DataFrame containing the data to be split by patient type
        column_names (list): list of names (str) for the columns of the DataFrame to be saved
        index_names (list): list of names (str) for the rows of the DataFrame to be saved
        df_names (list): list of names (str) for the DataFrame to be saved

    @ return:
        df_list_by_patient_type (list): list of DataFrames saved by patient type

    """

    # Initialize empty dataframes
    df_T123A = pd.DataFrame(columns=column_names)
    df_T123NA = pd.DataFrame(columns=column_names)
    df_T45 = pd.DataFrame(columns=column_names)

    for i, table in enumerate(data):
        df_T123A.loc[i] = table[0]
        df_T123NA.loc[i] = table[1]
        df_T45.loc[i] = table[2]

    # Set row names and column names for the dataframes
    df_T123A['Index'] = index_names
    df_T123A.set_index('Index', drop=True, inplace=True)
    df_T123NA['Index'] = index_names
    df_T123NA.set_index('Index', drop=True, inplace=True)
    df_T45['Index'] = index_names
    df_T45.set_index('Index', drop=True, inplace=True)
    df_T123A.name, df_T123NA.name, df_T45.name = df_names[0], df_names[1], df_names[2]

    # Store dataframes in a list
    df_list_by_patient_type = [df_T123A, df_T123NA, df_T45]

    return df_list_by_patient_type



def main_transform_simulation_results(filename, nruns, sys_state_list):
    """
    This function is the main simulation results DataFrame transformation function that reads in the results data,
    parse the results, and save the results data into readable DataFrames. The transformed data will be saved in
    Excel format.

    @ params:
        filename (str): specifies the name of the file to read from
        nruns (int): specifies the number of replications of the simulation that was performed
        sys_state_list (list): specifies the list of system states

    @ return:
        transformed_results_filename (str): the name of the file that the transformed results data is saved in

    """

    print('Transforming naive + simulation results...')

    labels_dict = {0: 'Simulated: General NIS', 1: 'Simulated: NIS by Patient Type', 2: 'Simulated: NIS by Zone',
                   3: 'Simulated: NIS by Patient Type x Zone', 12: 'Simulated: NIS by Patient Type + by Zone'}
    z_val = 1.96

    # For each system state
    idx_names_w_naive, idx_names_wo_naive = ['Naive'], []

    for i in range(len(sys_state_list)):
        idx_names_w_naive.append(labels_dict.get(sys_state_list[i]))
        idx_names_wo_naive.append(labels_dict.get(sys_state_list[i]))

    means_list, stderrs_list, medians_list, percentile90_list = [], [], [], []
    sheetname = 'Naive'
    mean, _, median, percentile90, cutdown = read_results_single_sys_state(filename, sheetname, z_val, nruns)
    means_list.append(mean)
    medians_list.append(median)
    percentile90_list.append(percentile90)

    for s in range(len(sys_state_list)):
        sheetname = 'System_State_{}'.format(sys_state_list[s])
        mean, stderr, median, percentile90, _ = read_results_single_sys_state(filename, sheetname, z_val, nruns)
        means_list.append(mean)
        stderrs_list.append(stderr)
        medians_list.append(median)
        percentile90_list.append(percentile90)

    df_names_suffix = ['mean', 'stderr', 'median', '90P']
    column_names = ['Cut Down Consult LOS by {}%'.format(c) for c in cutdown]
    dfs_by_patient_type_list = []
    all_lists = [means_list, stderrs_list, medians_list, percentile90_list]
    index_names_list = [idx_names_w_naive, idx_names_wo_naive, idx_names_w_naive, idx_names_w_naive]

    for i, suf in enumerate(df_names_suffix):
        df_names = ['df_T123A_{}'.format(suf), 'df_T123NA_{}'.format(suf), 'df_T45_{}'.format(suf)]
        dfs_by_patient_type_list.append(
            save_to_df_by_patient_type(all_lists[i], column_names, index_names_list[i], df_names))

    dfs_mean_stderr_by_patient_type = []
    for i in range(len(dfs_by_patient_type_list[0])):
        dfs_mean_stderr_by_patient_type.append(dfs_by_patient_type_list[0][i])  # expected mean from simulation
        dfs_mean_stderr_by_patient_type.append(dfs_by_patient_type_list[1][i])  # standard error on expected mean

    df_list = [dfs_mean_stderr_by_patient_type, dfs_by_patient_type_list[2], dfs_by_patient_type_list[3]]

    transformed_results_filename = "00_mean_median_90P_results.xlsx"
    writer = pd.ExcelWriter(os.path.join(os.getcwd(), transformed_results_filename), engine='xlsxwriter')
    workbook = writer.book

    worksheet_names = ['mean_with_stderr', 'median', '90th_percentile']

    for i, ws_name in enumerate(worksheet_names):
        print(ws_name)
        worksheet = workbook.add_worksheet(ws_name)
        writer.sheets[ws_name] = worksheet
        count = 0
        for df in df_list[i]:
            worksheet.write_string(count, 0, df.name)
            df.to_excel(writer, sheet_name=ws_name, startrow=count + 1, startcol=0)
            count += 8
    writer.save()

    return transformed_results_filename

