from nygh_pre_process_final import *
from nygh_simulation_final import *
from nygh_histogram_qqplots_kstest_final import *
from nygh_transform_results_final import *


##################################################################################################################
## Author: Sijia (Nancy) Li
## Project Name: Data-driven system simulation in healthcare applications
## Supervised By: Prof. Arik Senderovich, Prof. Dmitry Krass, Prof. Opher Baron
## Email: sijianancy.li@mail.utoronto.ca
##################################################################################################################


def main():
    """
    This function is the main function to CALL to run the pipeline:
        0. Asks users for some inputs
        1. Reads in data
        2. Splits data into training and testing sets
        3. Builds LOS models, simulates the system, computes performance measures, and saves results
        4. Transforms simulation results into Python readable format
        5. Constructs relative frequency histograms and Q-Q plots, and performs Kolmogorov–Smirnov tests
    """

    # Initialize
    step1_inputs, step2_inputs, step3_inputs, step4_inputs, step5_inputs = None, None, None, None, None

    # Asks users for steps to perform
    preprocessing_flag = bool(int(input("Would you like to pre-process your data? 1 for yes, 0 for no --> ")))
    los_modeling_flag = bool(
        int(input("Would you like to build LOS models (and simulate the system)? 1 for yes, 0 for no --> ")))
    plotting_flag = bool(
        int(input("Would you like to create rel_freq histograms and qq-plots? 1 for yes, 0 for no --> ")))

    # If user would like to pre-process the data
    if preprocessing_flag:
        step1_inputs = step1_preprocess_data_inputs() # Ask user for inputs for step 1
        # If user would like to build LOS models and perform simulation , ask user for for inputs for steps 2, 3, and 4.
        # Currently, no input is needed to perform step 4 (this step is performed automatically)
        if los_modeling_flag:
            step2_inputs = step2_df_train_test_split_inputs()
            step3_inputs = step3_E2E_simulation_inputs()
            # step4_inputs = step4_transform_results_data_clustered_bar_inputs()

    # If user doesn't want to pre-process the data (e.g., cleaned data already readily available)
    else:
        # If user wants to use cleaned data to build LOS models and simulate
        if los_modeling_flag:
            # Ask user for the file in which the cleaned data is saved in
            done1 = False
            while not done1:
                step1_inputs = input("\nEnter the filename of your CLEANED data in EXCEL format, for example: cleaned_data_all_final.xlsx --> ")
                print("====================================================================================================")
                print("Filename for cleaned data is: {}".format(step1_inputs))
                print("====================================================================================================")
                done1 = bool(int(input(
                    "Is the above parameter correct? 1 for yes, 0 for no (If no, will reset and re-enter the information) --> ")))
            # Ask user for inputs for steps 2, 3, and 4.
            # Currently, no input is needed to perform step 4 (this step is performed automatically)
            step2_inputs = step2_df_train_test_split_inputs()
            step3_inputs = step3_E2E_simulation_inputs()
            # step4_inputs = step4_transform_results_data_clustered_bar_inputs()

    print('\n')
    '''
    If user wants to construct relative frequency histograms and Q-Q plots, as well as perform K-S tests
    Note: this can be run independently if simulation was previously performed and results were saved with appropriate 
    file names. OR it can be run along with simulation (as the above code will automatically produce results files that
    are compatible with the plotting function below.
    '''
    if plotting_flag:
        step5_inputs = step5_relative_freq_histograms_and_qqplots_inputs()

    # After obtaining the user inputs for all steps, execute all the steps
    execute_all_steps(step1_inputs, step2_inputs, step3_inputs, step5_inputs, preprocessing_flag, los_modeling_flag, plotting_flag)

    return 'Done!'


def execute_steps_234(df, step2_inputs, step3_inputs):
    """
    This function takes in data, user inputs for steps 2 to 4 of the pipeline, and executes steps 2 to 4:
        Step 2. Splits data into training and testing sets
        Step 3. Builds LOS models, simulates the system, computes performance measures, and saves results
        Step 4. Transforms simulation results into a format directly readable as DataFrames
    """

    # STEP 2: Split training and testing DataFrames
    print("\n\nEXECUTING STEP 2...")
    start_year_month, end_year_month, train_start, train_end, test_start, test_end = step2_inputs
    df = select_df_year_month(df, start_year_month, end_year_month)
    df = add_num_wks_since_start_feature(df)

    df_train = select_df_year_month(df, train_start, train_end)
    df_train.drop(columns=['arrival_year'], inplace=True)

    df_test = select_df_year_month(df, test_start, test_end)
    df_test.drop(columns=['arrival_year'], inplace=True)

    # STEP 3: End-to-end simulation
    print("\n\nEXECUTING STEP 3...")
    sys_state_list, categorical_cols, conts_cols, interventions, log_LOS_flag, simulate_flag, nRuns, performance_measures_flag = step3_inputs
    df_naive_results, df_results_list, results_filename = main_model_simulation_performance(df, df_train, df_test, sys_state_list,
                                                        categorical_cols, conts_cols, interventions, log_LOS_flag,
                                                        simulate_flag, nRuns, performance_measures_flag)

    print(list(df_naive_results.columns))
    print(np.array(df_naive_results.head()))

    for df in df_results_list:
        print(list(df.columns))
        print(np.array(df.head()))

    # STEP 4: Transform mean, median, 90th percentile results from simulation
    print("\n\nEXECUTING STEP 4...")
    # clustered_bar_flag = step4_inputs
    transformed_results_filename = main_transform_simulation_results(filename=results_filename, nruns=nRuns, sys_state_list=sys_state_list)


def execute_all_steps(step1_inputs, step2_inputs, step3_inputs, step5_inputs, preprocessing_flag, los_modeling_flag, plotting_flag):
    """
    This function takes in all user inputs and executes steps 1 to 5 of the pipeline:
        Step 1. Reads in data
        Step 2. Splits data into training and testing sets
        Step 3. Builds LOS models, simulates the system, computes performance measures, and saves results
        Step 4. Transforms simulation results into a format that is directly readable as DataFrames
        Step 5. Constructs relative frequency histograms and Q-Q plots, and performs Kolmogorov–Smirnov tests
    """

    if preprocessing_flag:
        # STEP 1: Read in raw data + Pre-process data
        print("\n\nEXECUTING STEP 1...")
        filename, columns, years, cleaned_data_filename, write_data = step1_inputs
        df = main_preprocess_data(filename, columns, years, cleaned_data_filename, write_data)
        if los_modeling_flag:
            # STEPS 2, 3, and 4
            execute_steps_234(df, step2_inputs, step3_inputs)
    else:
        if los_modeling_flag:
            # STEP 1: Read in cleaned / pre - processed data
            print("\n\nEXECUTING STEP 1...")
            print("Reading in data...")
            df = pd.read_excel(os.path.join(os.getcwd(), step1_inputs), engine='openpyxl', header=0)
            columns = df.columns
            check_cols_list = ['Age Category', 'Initial Zone', 'arrival_hour', 'arrival_day_of_week', 'arrival_week_number', 'arrival_month']
            for col in check_cols_list:
                if col in columns: df[col] = df[col].astype("category")
                else: continue

            # STEPS 2, 3, and 4
            execute_steps_234(df, step2_inputs, step3_inputs)

    # STEP 5: Create relative frequency histograms and qq-plots, and perform KS tests
    if plotting_flag:
        print("\n\nEXECUTING STEP 5...")
        plotting_sys_state_list = step5_inputs
        main_rel_freq_histograms_qqplots_kstest(plotting_sys_state_list)

    return 'Done execute all steps'


def user_input_columns():
    """
    This function asks users to enter information regarding the column names to filter for in a DataFrame
    """

    done, columns = False, []
    while not done:
        nCols = input("How many column names you'd like to add? 1 for one column, 2 for more than one columns --> ")
        if int(nCols) == 1:
            col = input("Enter column name --> ")
            columns.append(col)
        elif int(nCols) == 2:
            cols = input("Enter column names separated by comma, for example: Age (Registration), Gender Code --> ")
            cols = cols.replace(', ', ',')
            cols = cols.split(',')
            columns.extend(cols)
        else:
            print("Error")
            continue

        valid_answer = False
        print("Current columns selected: ", columns)
        answer = input("Do you need to add more columns? Enter 0 for no, 1 for yes, 2 for reset --> ")
        while not valid_answer:
            if int(answer) == 0:
                done = True
                valid_answer = True
            elif int(answer) == 1:
                valid_answer = True
            elif int(answer) == 2:
                columns = []
                print("Current columns selected: ", columns)
                valid_answer = True
            else:
                answer = input(
                    "Try Again...\nDo you need to add more columns? Enter 0 for no, 1 for yes, 2 for reset --> ")
    return columns


def step1_preprocess_data_inputs():
    """
    STEP 1: PRE-PROCESSING DATA
    This function asks for some user inputs to read in the data, clean the data, save the cleaned data in an Excel file,
    and return the cleaned data in a DataFrame

    @ return:
        step1_inputs (list): filename of raw data, columns to pre-process, years of data to pre-process, whether or not
            to save cleaned data in a file, and the filename to save the cleaned data (if applicable)
    """
    print("\nSTEP 1: Pre-processing Inputs...")
    filename, columns, years, cleaned_data_filename, write_data = None, None, None, 'N/Ap', False
    done = False

    while not done:
        # step 1.1: ask user to give the name of the data file
        filename = input("Enter the filename of your data in CSV format, for example: NYGH_1_8_v1.csv -->  ")

        # step 1.2: ask user to provide the names of the columns that would like to be included in the cleaned data
        print('Provide column names from raw dataset...')
        print('For example: Age (Registration), Gender Code, Arrival Mode, Ambulance Arrival DateTime, Triage DateTime, Triage Code, Left ED DateTime, Initial Zone, Consult Service Description (1st), Diagnosis Code Description, CACS Cell Description, Discharge Disposition Description')
        columns = user_input_columns()
        '''
        Age (Registration), Gender Code, Arrival Mode, Ambulance Arrival DateTime, Triage DateTime, Triage Code, Left ED DateTime, Initial Zone, Consult Service Description (1st), Diagnosis Code Description, CACS Cell Description, Discharge Disposition Description
        '''

        # step 1.3: ask user for the years in which the user like to analyze for the "holidays" feature
        years = input("\nEnter years to clean/pre-process (separated by comma), for example: 2016, 2017, 2018 --> ")
        years = years.replace(', ', ',')
        years = [int(i) for i in years.split(',')]
        print("Current years selected: ", years)


        # step 1.4: ask user if want to write data to EXCEL file
        write_data = bool(int(input("\nDo you want to save data into an EXCEL file? 1 for yes, 0 for no --> ")))

        if write_data:
            # step 1.5: ask user for the filename in which the user would like to save the cleaned data to
            cleaned_data_filename = input(
                "Enter the filename you'd like to save the cleaned data to, for example: cleaned_data_all_final.xlsx --> ")

        print("====================================================================================================")
        print("Filename of raw data is: ", filename)
        print("Columns selected from raw data: ", columns)
        print("Years to analyze: ", years)
        print("Write data is: {}, and the filename to save the cleaned / preprocessed data is: {}".format(write_data, cleaned_data_filename))
        print("====================================================================================================")
        done = bool(int(input(
            "Are the above parameters correct? 1 for yes, 0 for no (If no, will reset and re-enter the information) --> ")))

    step1_inputs = [filename, columns, years, cleaned_data_filename, write_data]

    return step1_inputs


def step2_df_train_test_split_inputs():
    """
    STEP 2: TRAIN-TEST SPLIT
    This function asks for user inputs on the start year and month as well as end year and month for:
        1. All data (training + testing)
        2. Training data
        3. Testing data

    @ return:
        step2_inputs (list): start year and month (all data), end year and month (all data),
            start year and month (training), end year and month (training),
            start year and month (testing), end year and month (testing)
    """

    print("\nSTEP 2: Train and Test DataFrame Split Inputs...")
    start_year_month, end_year_month, train_start, train_end, test_start, test_end = None, None, None, None, None, None
    done = False

    while not done:
        start_year_month = input("Enter the START (year, month) of training AND testing data, separated by comma, e.g., for May 2018, enter: 2018, 5 --> ")
        end_year_month = input("Enter the END (year, month) of training AND testing data, separated by comma, e.g., for August 2018, enter: 2018, 8 --> ")
        # start_year_month, end_year_month = (2018, 5), (2018, 8)

        train_start = input("Enter the START (year, month) of TRAINING data, separated by comma, e.g., for May 2018, enter: 2018, 5 --> ")
        train_end = input("Enter the END (year, month) of TRAINING data, separated by comma, e.g., for July 2018, enter: 2018, 7 --> ")
        # train_start, train_end = (2018, 5), (2018, 7)

        test_start = input("Enter the START (year, month) of TESTING data, separated by comma, e.g., for August 2018, enter: 2018, 8 --> ")
        test_end = input("Enter the END (year, month) of TESTING data, separated by comma, e.g., for August 2018, enter: 2018, 8 --> ")
        # test_start, test_end = (2018, 8), (2018, 8)

        print("====================================================================================================")
        print("Range of all data (training and testing) -- start: {}; end: {}".format(start_year_month, end_year_month))
        print("Range of training data -- start: {}; end: {}".format(train_start, train_end))
        print("Range of testing data -- start: {}; end: {}".format(test_start, test_end))
        print("====================================================================================================")
        done = bool(int(input(
            "Are the above parameters correct? 1 for yes, 0 for no (If no, will reset and re-enter the information) --> ")))

    step2_inputs_temp = [start_year_month, end_year_month, train_start, train_end, test_start, test_end]

    step2_inputs = []
    for s in step2_inputs_temp:
        s = s.replace(', ', ',')
        s = [int(i) for i in s.split(',')]
        s = s[0], s[1]
        step2_inputs.append(s)

    return step2_inputs


def step3_E2E_simulation_inputs():
    """
    STEP 3: LOS Model Building, Simulation, Performance Measures Computation
    This function asks for some user inputs to execute end-to-end simulation

    @ return:
        step3_inputs (list): list of system states interested in analyzing, names of categorical columns in the data,
            names of continuous columns in the data, list of interventions level, whether to use log of LOS in the
            models, whether to simulate the system, number of replications for simulation, whether to compute the
            performance measures during simulation
    """

    print("\n\nSTEP 3: End-to-end Simulation Inputs...")
    sys_state_list, categorical_cols, conts_cols, interventions = [], [], [], []
    log_LOS_flag, simulate_flag, nRuns, performance_measures_flag = True, True, 30, True
    done = False
    while not done:
        print('System state 0: overall NIS, 1: NIS by Patient Type, 2: NIS by Zone, 3: NIS by Patient Type x Zone')
        sys_state_list = input("Enter system state(s) you'd like to build LOS models and separate them by comma, for example: 0, 1 --> ")
        sys_state_list = sys_state_list.replace(', ', ',')
        sys_state_list = [int(i) for i in sys_state_list.split(',')]

        print('Provide CATEGORICAL column names from cleaned data (x features used to train LOS model)...')
        print('For example: Age Category, Gender Code, Triage Category, Ambulance, Consult, Initial Zone, arrival_hour, arrival_day_of_week, arrival_week_number, arrival_month, holiday_CAN_ON')
        categorical_cols = user_input_columns()
        print('\nProvide CONTINUOUS column names (except for system state features) from cleaned data (x features and y feature to train the LOS model)...')
        print('For example: arrival_num_week_since_start, sojourn_time(minutes)')
        conts_cols = user_input_columns()

        '''
        categorical_cols = Age Category, Gender Code, Triage Category, Ambulance, Consult, Initial Zone, arrival_hour, arrival_day_of_week, arrival_week_number, arrival_month, holiday_CAN_ON
        conts_cols = arrival_num_week_since_start, sojourn_time(minutes)
        '''

        print('\nInterventions (% LOS cutdown for consult patients)...')
        print('Entering 1.0 means cut down by (1-1.0)*100% = 0% --> no intervention is applied')
        print('Entering 0.6 means cut down by (1-0.6)*100% = 40% --> intervention is applied, LOS of consult patient will be reduced by 40%')
        interventions = input("Enter interventions separated by comma, for example: 0.5, 1.0 --> ")
        interventions = interventions.replace(', ', ',')
        interventions = [round(float(i),1) for i in interventions.split(',')]

        log_LOS = input("\nDo you want to take the log of LOS? 1 for yes, 0 for no --> ")
        log_LOS_flag = bool(int(log_LOS))

        simulate = input("Do you want to simulate the system? 1 for yes, 0 for no --> ")
        simulate_flag = bool(int(simulate))

        if simulate_flag:
            nRuns = input("How many replications / runs would you like to simulate? For example, 30 --> ")
            nRuns = int(nRuns)

            performance_measures = input("Do you want to compute the simulation performance measures? 1 for yes, 0 for no --> ")
            performance_measures_flag = bool(int(performance_measures))
        else:
            nRuns = 0
            performance_measures_flag = False

        print("====================================================================================================")
        print("System states selected: ", sys_state_list)
        print("Categorical variables columns: ", categorical_cols)
        print("Continuous variables columns: ", conts_cols)
        print("Inverventions selected: ", interventions)
        print("log(LOS) flag is: {}, simulate flag is: {}, performance measures flag is: {}, number of simulation runs will be performed is: {}".format(log_LOS_flag, simulate_flag, performance_measures_flag, nRuns))
        print("====================================================================================================")
        done = bool(int(input("Are the above parameters correct? 1 for yes, 0 for no (If no, will reset and re-enter the information) --> ")))

    step3_inputs = [sys_state_list, categorical_cols, conts_cols, interventions, log_LOS_flag, simulate_flag, nRuns, performance_measures_flag]

    return step3_inputs


# def step4_transform_results_data_clustered_bar_inputs():
#     """
#     ========================================PLACEHOLDER FUNCTION========================================
#     STEP 4: Transform results from simulation into a format that is directly readable as DataFrames
#     This step currently doesn't require user input
#
#     @ return:
#         step4_inputs (list):
#     """
#
#     clustered_bar_flag = None
#     done = False
#     while not done:
#         clustered_bar_flag = bool(int(input("Would you like to create clustered bar plots for the results? 1 for yes, 0 for no --> ")))
#         print("====================================================================================================")
#         print("Clustered bar plots flag is: {}".format(clustered_bar_flag))
#         print("====================================================================================================")
#         done = bool(int(input(
#             "Is the above parameter correct? 1 for yes, 0 for no (If no, will reset and re-enter the information) --> ")))
#
#     step4_inputs = clustered_bar_flag
#     return step4_inputs


def step5_relative_freq_histograms_and_qqplots_inputs():
    """
    STEP 5: Create relative frequency histograms and qq-plots, and perform KS tests
    This function asks for a user input to construct histograms and Q-Q plots, as well as perform KS tests

    @ return:
        step5_inputs (list): only one user input from this step -- the list of system states to construct plots and
            perform KS test
    """

    step5_inputs = None
    done = False
    while not done:
        step5_inputs = input(
            "Enter system state(s) you'd like to create rel_freq histograms and qq-plots for, separated them by comma, for example: 0, 1 --> ")
        step5_inputs = step5_inputs.replace(', ', ',')
        step5_inputs = [int(i) for i in step5_inputs.split(',')]
        print("====================================================================================================")
        print("System states to create plots for are: {}".format(step5_inputs))
        print("====================================================================================================")
        done = bool(int(input(
            "Is the above parameter correct? 1 for yes, 0 for no (If no, will reset and re-enter the information) --> ")))

    return step5_inputs


if __name__ == "__main__":
    main()

