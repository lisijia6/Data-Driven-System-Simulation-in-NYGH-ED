import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
import heapq as hq
import time
import itertools


def select_df_year_month(df, start_year_month, end_year_month):
    """
    Selects the relevant data based on given start year and month, as well as end year and month

    @ params:
        df (pd.DataFrame): the DataFrame to select from
        start_year_month (tuple): the (start year, start month) of the data to be selected
        end_year_month (tuple): the (end year, end month) of the data to be selected

    @ return:
        df_return (pd.DataFrame): the subset of the original DataFrame based on start year and month,
                                    as well as end year and month
    """

    df_return = df[0:0]

    # If training and testing data are from the same year
    if start_year_month[0] == end_year_month[0]:
        print('Data from the same year...')
        df1 = df[df['arrival_year'] == start_year_month[0]]
        start_month, end_month = start_year_month[1], end_year_month[1]
        month_list = list(np.arange(start_month, end_month + 1))
        for m in month_list:
            df_temp = df1[df1['arrival_month'] == m]
            df_return = pd.concat([df_return, df_temp])

    # If training and testing data are from 2 different years
    elif (end_year_month[0] - start_year_month[0]) == 1:
        print('Data from 2 different years...')
        start_year, end_year = start_year_month[0], end_year_month[0]
        start_month, end_month = start_year_month[1], end_year_month[1]

        y1_month_list = list(np.arange(start_month, 13))
        y2_month_list = list(np.arange(1, end_month + 1))

        # First year of data
        df1 = df[df['arrival_year'] == start_year]
        for m1 in y1_month_list:
            df_temp = df1[df1['arrival_month'] == m1]
            df_return = pd.concat([df_return, df_temp])
        # Second year of data
        df2 = df[df['arrival_year'] == end_year]
        for m2 in y2_month_list:
            df_temp = df2[df2['arrival_month'] == m2]
            df_return = pd.concat([df_return, df_temp])

    # If training and testing data are from more than 2 different years
    else:
        print('Data from more than 2 different years...')
        start_year, end_year = start_year_month[0], end_year_month[0]
        start_month, end_month = start_year_month[1], end_year_month[1]
        full_year_list = list(np.arange(start_year - 1, end_year))

        y1_month_list = list(np.arange(start_month, 13))
        yn_month_list = list(np.arange(1, end_month + 1))

        # First year of data
        df1 = df[df['arrival_year'] == start_year]
        for m1 in y1_month_list:
            df_temp = df1[df1['arrival_month'] == m1]
            df_return = pd.concat([df_return, df_temp])
        # Full years of data
        for y in full_year_list:
            df_temp = df[df['arrival_year'] == y]
            df_return = pd.concat([df_return, df_temp])
        # Last year of data
        dfn = df[df['arrival_year'] == end_year]
        for mn in yn_month_list:
            df_temp = dfn[dfn['arrival_month'] == mn]
            df_return = pd.concat([df_return, df_temp])

    df_return = df_return.reset_index(drop=True)
    return df_return


def add_num_wks_since_start_feature(df):
    """
    Adds 'arrival_num_week_since_start' as an additional feature to the DataFrame
    """
    start_date = df.iloc[0]['patient_arrival_times']
    df['arrival_num_week_since_start'] = df.apply(
        lambda row: compute_week_since_start(row['patient_arrival_times'], start_date), axis=1)
    df['arrival_num_week_since_start'] = df["arrival_num_week_since_start"].astype("int64")
    return df


def compute_week_since_start(arrival_time, start_date):
    """
    Calculates the number of weeks since the start of the training data
    """
    delta = arrival_time - start_date
    delta_days = delta.days - start_date.weekday()
    week_since_beginning_of_training = delta_days // 7 + 1
    if start_date.weekday() != 0:
        week_since_beginning_of_training += 1
    return week_since_beginning_of_training


def get_nis_features(df, system_state):
    """
    Retrieves unique patient types, unique initial zones, and unique patient type x initial zone based on system state,
    and also retrieves the NIS column names corresponding to the system state in the DataFrame
    """

    unique_patient_types = df['Triage Category'].unique()
    unique_zones = df['Initial Zone'].unique()
    unique_patient_type_zones = list(itertools.product(unique_patient_types, unique_zones))

    if system_state == 0:
        return 'N/Ap', ['NIS Upon Arrival']
    elif system_state == 1:
        nis_pt_features = ['NIS Upon Arrival Patient_{}'.format(pt) for pt in unique_patient_types]
        return unique_patient_types, nis_pt_features
    elif system_state == 2:
        nis_zone_features = ['NIS Upon Arrival Zone_{}'.format(zone) for zone in unique_zones]
        return unique_zones, nis_zone_features
    elif system_state == 3:
        nis_pt_zone_features = ['NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1]) for pt_zone in
                                unique_patient_type_zones]
        return unique_patient_type_zones, nis_pt_zone_features


def get_x_and_y_train_test_split_helper(df_train, df_test, x_train_cols, x_test_cols, y_cols):
    """
    Splits training and testing data into x_train and y_train, x_test and y_test
    """
    x_train, y_train = df_train[x_train_cols], df_train[y_cols]
    x_test, y_test = df_test[x_test_cols], df_test[y_cols]
    y_train_reshaped = np.array(y_train).reshape(len(y_train), )
    y_test_reshaped = np.array(y_test).reshape(len(y_test), )

    print('x_train.shape, x_test.shape: ', x_train.shape, x_test.shape)
    print('y_train.shape, y_test.shape: ', y_train.shape, y_test.shape)
    return x_train, x_test, y_train_reshaped, y_test_reshaped


def get_x_and_y_train_test_split_los_model(df, df_train, df_test, categorical_cols, conts_cols, triage_category, system_state=0, log_LOS=True):
    """
    Converts categorical variables into dummy variables, obtains x and y features for los model, and splits data
    into x_train, y_train, x_test, and y_test

    @ params:
        df (pd.DataFrame): the overall DataFrame that contains both training AND testing data
        df_train (pd.DataFrame): contains TRAINING data
        df_test (pd.DataFrame): contains TESTING data
        categorical_cols (list): list of columns names (str) for CATEGORICAL variables
        conts_cols (list): list of columns names (str) for CONTINUOUS variables, except for NIS features
        triage_category (str): specifies the type of patient (T123 Admitted, T123 Not Admitted, or T45)
        system_state (int, default=0): specifies the system state (used for NIS features selection)
        log_LOS (bool, default=True): specifies whether model's outcome variable is log(LOS --True or LOS -- False

    @ return:
        los_x_train (pd.DataFrame): features data (training)
        los_x_test (pd.DataFrame): features data (testing)
        los_y_train (np array): outcomes data (training)
        los_y_test (np array): outcomes data (testing)
        los_x_train_features (list): list of column names(s) for model training features
        x_test_zone_dropped_dummy (str): the initial zone that was dropped when converting initial zone into dummy variable
    """

    print('Getting x and y features with dummy variables for LOS model for {}...'.format(triage_category))
    _, nis_columns = get_nis_features(df, system_state)  # select NIS features based on system state
    columns = categorical_cols + conts_cols + nis_columns
    df_los_train, df_los_test = df_train[columns], df_test[columns]

    # Select only the rows for patients in the specific triage category
    df_los_train = df_los_train[df_los_train['Triage Category'] == triage_category]
    df_los_train.drop(columns=['Triage Category'], inplace=True)
    df_los_test = df_los_test[df_los_test['Triage Category'] == triage_category]
    df_los_test.drop(columns=['Triage Category'], inplace=True)
    categorical_cols.remove('Triage Category')

    # Take the log of LOS
    if log_LOS:
        df_los_train['log_LOS(minutes)'] = df_los_train.apply(lambda row: np.log(row['sojourn_time(minutes)']), axis=1)
        df_los_test['log_LOS(minutes)'] = df_los_test.apply(lambda row: np.log(row['sojourn_time(minutes)']), axis=1)
        los_y_feature = ['log_LOS(minutes)']
    else:
        los_y_feature = ['sojourn_time(minutes)']
    conts_cols.remove('sojourn_time(minutes)')

    # Before converting categorical variables into dummy variables, get all the unique Initial Zone categories
    x_test_all_zones = df_los_test['Initial Zone'].unique()

    # Obtain dummy variables for categorical variables
    df_los_train = pd.get_dummies(data=df_los_train, drop_first=True)
    df_los_test = pd.get_dummies(data=df_los_test, drop_first=True)

    # Obtain the list of column dummy names for features in the model (except for Initial Zone)
    los_x_features = conts_cols
    categorical_cols.remove('Initial Zone')
    for feature in categorical_cols:
        los_x_features += [i for i in df_los_train.columns if i.startswith(feature + '_')]

    los_x_features += nis_columns  # add NIS features

    # Initial Zone features -- note: # of dummy variables created will correspond to the # of categories in the Initial Zone column
    x_train_zone_cols = [i for i in df_los_train.columns if i.startswith('Initial Zone_')]
    los_x_train_features = los_x_features + x_train_zone_cols
    x_test_zone_cols = [i for i in df_los_test.columns if i.startswith('Initial Zone_')]
    los_x_test_features = los_x_features + x_test_zone_cols

    # Find the category that was dropped when converting categorical variables into dummy variables --> needed for later
    x_test_dummy_zones = [i.split('_')[1] for i in x_test_zone_cols]
    x_test_zone_dropped_dummy = ''
    for z in x_test_all_zones:
        if z not in x_test_dummy_zones:
            x_test_zone_dropped_dummy = z

    print('Getting x_train, y_train, x_test, y_test split for {}...'.format(triage_category))
    # Get x_train, y_train, x_test, and y_test
    los_x_train, los_x_test, los_y_train, los_y_test = get_x_and_y_train_test_split_helper(df_los_train, df_los_test,
                                                                                           los_x_train_features,
                                                                                           los_x_test_features,
                                                                                           los_y_feature)

    print('Done: Getting x and y features with dummy variables for LOS model for {}...'.format(triage_category))
    return los_x_train, los_x_test, los_y_train, los_y_test, los_x_train_features, x_test_zone_dropped_dummy


def build_los_model(df, df_train, df_test, categorical_cols, conts_cols, triage_category, system_state=0, log_LOS=True, feature_importance_flag=False):
    """
    Constructs LOS model given the patient type (aka triage_category)

    @ params:
        df (pd.DataFrame): the overall DataFrame that contains both training AND testing data
        df_train (pd.DataFrame): contains TRAINING data
        df_test (pd.DataFrame): contains TESTING data
        categorical_cols (list): list of columns names (str) for CATEGORICAL variables
        conts_cols (list): list of columns names (str) for CONTINUOUS variables, except for NIS features
        triage_category (str): specifies the type of patient (T123 Admitted, T123 Not Admitted, or T45)
        system_state (int, default=0): specifies the system state (used for NIS features selection)
        log_LOS (bool, default=True): specifies whether model's outcome variable is log(LOS --True or LOS -- False
        feature_importance_flag (bool, default=False): specifies whether or not to compute and print out
                                                        feature importance information about the model

    @ return:
        los_x_train (pd.DataFrame): features data (training)
        los_x_test (pd.DataFrame): features data (testing)
        los_y_train (np array): outcomes data (training)
        los_y_test (np array): outcomes data (testing)
        model_los_RF (RandomForestRegressor): the trained LOS model
        x_test_zone_dropped_dummy (str): the initial zone that was dropped when converting initial zone into dummy variable
    """

    los_x_train, los_x_test, los_y_train, los_y_test, los_x_train_features, x_test_zone_dropped_dummy = get_x_and_y_train_test_split_los_model(
        df, df_train, df_test, categorical_cols, conts_cols, triage_category, system_state, log_LOS)

    print('Building LOS model for {}...'.format(triage_category))
    model_los_RF = RandomForestRegressor(min_samples_leaf=30, max_depth=None, n_estimators=100, random_state=0)
    model_los_RF.fit(los_x_train, los_y_train)

    if feature_importance_flag:
        # Get numerical feature importances
        importances = list(model_los_RF.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(los_x_train_features, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        print(triage_category)
        [print('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances]

    print('Done: Building LOS model for {}...'.format(triage_category))
    return los_x_train, los_x_test, los_y_train, los_y_test, model_los_RF, x_test_zone_dropped_dummy


def get_los_errors(model, x_train, y_train):
    """
    Retrieves the list of training prediction errors given a model

    @ params:
        model (RandomForestRegressor): the LOS model
        x_train (pd.DataFrame): features data (training)
        y_train (np array): outcomes data (training)

    @ return:
        errors (list): list of training prediction errors given a model
    """
    actual = pd.Series(data=y_train, index=x_train.index)
    y_preds = model.predict(np.array(x_train))
    predicted = pd.Series(data=y_preds, index=x_train.index)

    errors = []
    for i in range(len(actual)):
        err = actual.iloc[i] - predicted.iloc[i]
        errors.append(err)
    return errors


def sample_from_RF_regressor(model, sample, error_list, log_LOS_flag):
    """
    Samples an LOS given the LOS model and noise (list of training prediction errors)
    """

    los_RF_pred = model.predict(np.array(sample).reshape(1, -1))
    error = np.random.choice(error_list, 1)
    if log_LOS_flag:
        return np.exp(los_RF_pred + error)
    else:
        return los_RF_pred + error


def simulate_single_scenario(system_state, cutdown, nRuns, initial_event_calendar, triage_categories, los_models, los_x_tests, los_train_errs, x_test_zone_dropped_dummy, unique_sys_states_categories, log_LOS_flag):
    """
    Performs simulation on one set of parameters

    @ params:
        system_state (int): specifies the system state (used for NIS features selection)
        cutdown (float): specifies the percentage in which to cut down consult patients' LOS (value between 0.0 to 1.0)
        nRuns (int): specifies the number of replications for the simulation
        initial_event_calendar (heap): a heap consisting of (ts_, event_, id_) -- timestamp of event, event (a = arrival,
            or d = departure), and id of the patient. This is used to keep track the arrivals and departures of
            patients in the system.
        triage_categories (list): a list of patient types corresponding to each patient in the system
        los_models (list): a list of LOS models (3 models, one for each patient type)
        los_x_tests (list): a list of pd.DataFrame containing features data (testing); (3 dataframes, one for each patient type)
        los_train_errs (list): a list of training prediction errors (3 lists, one for each patient type)
        x_test_zone_dropped_dummy (str): the initial zone that was dropped when converting initial zone into dummy variable
        unique_sys_states_categories (list): a list containing 3 lists - 1) unique patient types, 2) unique initial zones,
            3) unique patient type x initial zone
        log_LOS_flag (bool): specifies whether model's outcome variable is log(LOS --True or LOS -- False

    @ return:
        los_tr_nruns (list): contains lists of LOS trackers for n runs / simulation replications, each patient type
            has a separate LOS tracker
        nPatients (list): contains patient counts for each patient type
        nConsultPatients (list): contains consult patient counts for each patient type
        percentConsult (list): contains percentages of consult patients for each patient type

    """

    np.random.seed(3)  # set random seed

    # ----------------------------Start of: ungrouping from arguments to the function----------------------------
    model_los_RF_T123A, model_los_RF_T123NA, model_los_RF_T45 = los_models  # LOS models by patient type

    # x_test by patient type
    los_x_test_T123A, los_x_test_T123NA, los_x_test_T45 = los_x_tests
    los_x_test_T123A, los_x_test_T123NA, los_x_test_T45 = los_x_test_T123A.reset_index(), los_x_test_T123NA.reset_index(), los_x_test_T45.reset_index()
    los_x_test_T123A.drop(columns=['index'], inplace=True)
    los_x_test_T123NA.drop(columns=['index'], inplace=True)
    los_x_test_T45.drop(columns=['index'], inplace=True)

    los_errors_T123A, los_errors_T123NA, los_errors_T45 = los_train_errs  # LOS training errors by patient type
    unique_patient_types, unique_zones, unique_patient_type_zones = unique_sys_states_categories  # Unique categories in all training and testing data
    # ----------------------------End of: ungrouping from arguments to the function----------------------------


    # Compute the total number of patients for each patient type
    n_T123A, n_T123NA, n_T45 = len(los_x_test_T123A), len(los_x_test_T123NA), len(los_x_test_T45)

    # Initializations
    los_tr_nruns = []  # LOS trackers list for n runs (aka simulation replications)
    n_consult_T123A, n_consult_T123NA, n_consult_T45 = 0, 0, 0  # Counters for total number of consult patients for each patient type
    zones = [i for i in los_x_tests[0].columns if i.startswith('Initial Zone_')]  # List of zones in testing data

    # Simulation for n runs starts...
    for r in range(nRuns):
        print('\nRunning simulation run #: ' + str(r + 1))

        # ---------------------------------------Start of: Initialization---------------------------------------
        event_calendar = initial_event_calendar.copy()

        # LOS tracker for Run #r, in the format of: [[triage123A], [triage123NA], [triage45]]
        los_tr = [[] for _ in range(3)]

        # Initalize current NIS tracker for Run #r:
        if system_state == 0:  # System State 0 (General NIS)
            curr_nis = 0
        else:
            curr_nis = ()
            if system_state == 1:  # System State 1: NIS by Patient Type
                for i in range(len(unique_patient_types)):
                    curr_nis += (0,)
            elif system_state == 2:  # System State 2: NIS by Zone
                for i in range(len(unique_zones)):
                    curr_nis += (0,)
            elif system_state == 3:  # System State 3: NIS by Patient Type x Zone
                for i in range(len(unique_patient_type_zones)):
                    curr_nis += (0,)
        # ---------------------------------------End of: Initialization---------------------------------------

        """
        Keep track of arrival_index (overall patient # in the entire system) and the corresponding
        index of the individual patient type dataframe --> needed for departure event NIS update
        """
        T123A_dict, T123NA_dict, T45_dict = {}, {}, {}  # arrival_index : df_patient_type_index
        T123A_idx, T123NA_idx, T45_idx = 0, 0, 0

        # Simulation Starts
        while len(list(event_calendar)) > 0:
            ts_, event_, id_ = hq.heappop(event_calendar)  # Take an event from the event_calendar

            # Arrival Event
            if event_ == 'a':
                # Prints to console for every 1000 patients
                if id_ % 1000 == 0:
                    print('arrival: ', id_)

                df_los_test, los_model, los_errors = None, None, None
                idx_, n_consult, los_tr_idx = None, None, None
                # Parameters retrieval for each patient type
                if triage_categories[id_] == 'T123 Admitted':
                    df_los_test, los_model, los_errors = los_x_test_T123A.copy(), model_los_RF_T123A, los_errors_T123A
                    idx_, n_consult, los_tr_idx = T123A_idx, n_consult_T123A, 0
                    T123A_dict[id_] = T123A_idx
                elif triage_categories[id_] == 'T123 Not Admitted':
                    df_los_test, los_model, los_errors = los_x_test_T123NA.copy(), model_los_RF_T123NA, los_errors_T123NA
                    idx_, n_consult, los_tr_idx = T123NA_idx, n_consult_T123NA, 1
                    T123NA_dict[id_] = T123NA_idx
                elif triage_categories[id_] == 'T45':
                    df_los_test, los_model, los_errors = los_x_test_T45.copy(), model_los_RF_T45, los_errors_T45
                    idx_, n_consult, los_tr_idx = T45_idx, n_consult_T45, 2
                    T45_dict[id_] = T45_idx

                # -------------------------------Start of: update NIS counts based on the system state-------------------------------
                if system_state == 0:  # System State 0: General NIS
                    curr_nis += 1
                    df_los_test.at[idx_, 'NIS Upon Arrival'] = curr_nis

                elif system_state == 1:  # System State 1: NIS by Patient Type
                    if triage_categories[id_] == 'T123 Admitted':
                        curr_nis = (curr_nis[0] + 1, curr_nis[1], curr_nis[2])
                    elif triage_categories[id_] == 'T123 Not Admitted':
                        curr_nis = (curr_nis[0], curr_nis[1] + 1, curr_nis[2])
                    elif triage_categories[id_] == 'T45':
                        curr_nis = (curr_nis[0], curr_nis[1], curr_nis[2] + 1)
                    df_los_test.at[idx_, 'NIS Upon Arrival Patient_T123 Admitted'] = curr_nis[0]
                    df_los_test.at[idx_, 'NIS Upon Arrival Patient_T123 Not Admitted'] = curr_nis[1]
                    df_los_test.at[idx_, 'NIS Upon Arrival Patient_T45'] = curr_nis[2]

                elif system_state == 2:  # System State 2: NIS by Zone
                    row_zones = df_los_test.iloc[idx_][zones]
                    zone_of_sample = ''
                    for c in zones:
                        if row_zones[c] == 1:
                            zone_of_sample = c.split('_')[1]
                    if zone_of_sample == '':
                        zone_of_sample = x_test_zone_dropped_dummy

                    temp_nis_by_zone = ()
                    for i, zone in enumerate(unique_zones):
                        if zone_of_sample == zone:
                            temp_nis_by_zone = temp_nis_by_zone + (curr_nis[i] + 1,)
                            df_los_test.at[idx_, 'NIS Upon Arrival Zone_{}'.format(zone)] = curr_nis[i] + 1
                        else:
                            temp_nis_by_zone = temp_nis_by_zone + (curr_nis[i],)
                            df_los_test.at[idx_, 'NIS Upon Arrival Zone_{}'.format(zone)] = curr_nis[i]
                    curr_nis = temp_nis_by_zone

                elif system_state == 3:  # System State 3: NIS by Patient Type x Zone
                    row_patients_zones = df_los_test.iloc[idx_][zones]
                    zone_of_sample = ''
                    for c in zones:
                        if row_patients_zones[c] == 1:
                            zone_of_sample = c.split('_')
                    if zone_of_sample == '':
                        zone_of_sample = x_test_zone_dropped_dummy

                    temp_nis_by_pt_zone = ()
                    for j, pt_zone in enumerate(unique_patient_type_zones):
                        if (triage_categories[id_], zone_of_sample) == pt_zone:
                            temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis[j] + 1,)
                            df_los_test.at[idx_, 'NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1])] = curr_nis[j] + 1
                        else:
                            temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis[j],)
                            df_los_test.at[idx_, 'NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1])] = curr_nis[j]
                    curr_nis = temp_nis_by_pt_zone
                # -------------------------------End of: update NIS counts based on the system state-------------------------------

                # Sample LOS for the arriving patient, apply intervention to if it is a consult patient
                sample = df_los_test.iloc[idx_]
                idx_ += 1

                los = sample_from_RF_regressor(los_model, sample, los_errors, log_LOS_flag)  # LOS sampling

                if sample['Consult_Yes'] == 1:  # Intervention for consult patients
                    los = los * cutdown
                    if r == 0:
                        n_consult += 1

                los_tr[los_tr_idx].append(round(los[0], 2))  # Update LOS tracker

                d = ts_ + timedelta(minutes=int(los[0]))
                hq.heappush(event_calendar, (d, 'd', id_))  # Add a departure event to the event_calendar

                # Update other counts
                if triage_categories[id_] == 'T123 Admitted':
                    T123A_idx, n_consult_T123A = idx_, n_consult
                elif triage_categories[id_] == 'T123 Not Admitted':
                    T123NA_idx, n_consult_T123NA = idx_, n_consult
                elif triage_categories[id_] == 'T45':
                    T45_idx, n_consult_T45 = idx_, n_consult


            # Departure Event
            else:
                idx_, df_los_test = None, None

                if triage_categories[id_] == 'T123 Admitted':
                    idx_, df_los_test = T123A_dict.get(id_), los_x_test_T123A.copy()
                elif triage_categories[id_] == 'T123 Not Admitted':
                    idx_, df_los_test = T123NA_dict.get(id_), los_x_test_T123NA.copy()
                elif triage_categories[id_] == 'T45':
                    idx_, df_los_test = T45_dict.get(id_), los_x_test_T45.copy()

                # -------------------------------Start of: update NIS counts based on the system state-------------------------------
                if system_state == 0:  # System State 0: General NIS
                    curr_nis -= 1  # update current number of customers in the system

                elif system_state == 1:  # System State 1: NIS by Patient Type
                    if triage_categories[id_] == 'T123 Admitted':
                        curr_nis = (curr_nis[0] - 1, curr_nis[1], curr_nis[2])
                    elif triage_categories[id_] == 'T123 Not Admitted':
                        curr_nis = (curr_nis[0], curr_nis[1] - 1, curr_nis[2])
                    elif triage_categories[id_] == 'T45':
                        curr_nis = (curr_nis[0], curr_nis[1], curr_nis[2] - 1)

                elif system_state == 2:  # System State 2: NIS by Zone
                    row_zones = df_los_test.iloc[idx_][zones]
                    zone_of_sample = ''
                    for c in zones:
                        if row_zones[c] == 1:
                            zone_of_sample = c.split('_')
                    if zone_of_sample == '':
                        zone_of_sample = x_test_zone_dropped_dummy

                    temp_nis_by_zone = ()
                    for i, zone in enumerate(unique_zones):
                        if zone_of_sample == zone:
                            temp_nis_by_zone = temp_nis_by_zone + (curr_nis[i] - 1,)
                        else:
                            temp_nis_by_zone = temp_nis_by_zone + (curr_nis[i],)
                    curr_nis = temp_nis_by_zone

                elif system_state == 3:  # System State 3: NIS by Patient Type x Zone
                    row_zones = df_los_test.iloc[idx_][zones]
                    zone_of_sample = ''
                    for c in zones:
                        if row_zones[c] == 1:
                            zone_of_sample = c.split('_')
                    if zone_of_sample == '':
                        zone_of_sample = x_test_zone_dropped_dummy

                    temp_nis_by_pt_zone = ()
                    for j, pt_zone in enumerate(unique_patient_type_zones):
                        if (triage_categories[id_], zone_of_sample) == pt_zone:
                            temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis[j] - 1,)
                        else:
                            temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis[j],)
                    curr_nis = temp_nis_by_pt_zone
                # -------------------------------End of: update NIS counts based on the system state-------------------------------

        los_tr_nruns.append(los_tr)
        # print("curr_nis: ", curr_nis)
        # print("nConsultPatients ", [n_consult_T123A, n_consult_T123NA, n_consult_T45])

    nPatients = [n_T123A, n_T123NA, n_T45]
    nConsultPatients = [n_consult_T123A, n_consult_T123NA, n_consult_T45]
    percentConsult = [round((n_consult_T123A / n_T123A) * 100, 1), round((n_consult_T123NA / n_T123NA) * 100, 1),
                      round((n_consult_T45 / n_T45) * 100, 1)]

    return los_tr_nruns, nPatients, nConsultPatients, percentConsult


def main_simulate(sys_state, interventions, nRuns, initial_event_calendar, triage_categories, los_x_tests, los_train_errs, los_models, x_test_zone_dropped_dummy, unique_sys_states_categories, performance_measures_flag=True, log_LOS_flag=True):
    """
    Performs simulation for a selection / multiple simulation scenarios

   @ params:
        system_state (int): specifies the system state (used for NIS features selection)
        interventions (list): a list of percentages to cut down consult patients' LOS (value between 0.0 to 1.0)
        nRuns (int): specifies the number of replications for the simulations
        initial_event_calendar (heap): a heap consisting of (ts_, event_, id_) -- timestamp of event, event (a = arrival,
            or d = departure), and id of the patient. This is used to keep track the arrivals and departures of
            patients in the system.
        triage_categories (list): a list of patient types corresponding to each patient in the system
        los_x_tests (list): a list of pd.DataFrame containing features data (testing); (3 dataframes, one for each patient type)
        los_train_errs (list): a list of training prediction errors (3 lists, one for each patient type)
        los_models (list): a list of LOS models (3 models, one for each patient type)
        x_test_zone_dropped_dummy (str): the initial zone that was dropped when converting initial zone into dummy variable
        unique_sys_states_categories (list): a list containing 3 lists - 1) unique patient types, 2) unique initial zones,
            3) unique patient type x initial zone
        performance_measures_flag (bool, default=True): specifies whether to compute the performance measures
            for the simulation (mean, median, standard deviation, 90th percentile, 95% confidence interval, RMSE)
        log_LOS_flag (bool, default=True): specifies whether model's outcome variable is log(LOS --True or LOS -- False

    @ return:
        df_results (pd.DataFrame): contains the simulation results with performance measures (if computed)

    """

    df_results = pd.DataFrame(
        columns=['System State', 'nPatients [T123A, T123NA, T45]', 'nConsultPatients [T123A, T123NA, T45]',
                 'percentConsult [T123A, T123NA, T45]',
                 'nRuns', 'Cut Down by (%)', 'RMSE [T123A, T123NA, T45]', 'Mean [T123A, T123NA, T45]',
                 'Median [T123A, T123NA, T45]',
                 'Stdev [T123A, T123NA, T45]', 'CI on Mean [T123A, T123NA, T45]',
                 '90th Percentile [T123A, T123NA, T45]', 'Time to Run (seconds)'])

    for i, cutdown in enumerate(interventions):
        print("Consult Patients shorten by {} Percent, System State = {}, Total Runs = {}...".format((1 - cutdown) * 100, sys_state, nRuns))
        start1 = time.time()

        # Run simulation for n runs (aka simulation replications) for a single scenario. Arrivals from testing data is used
        los_tr_nruns, nPatients, nConsultPatients, percentConsult = simulate_single_scenario(sys_state, cutdown, nRuns,
                                                                                             initial_event_calendar,
                                                                                             triage_categories,
                                                                                             los_models, los_x_tests,
                                                                                             los_train_errs,
                                                                                             x_test_zone_dropped_dummy,
                                                                                             unique_sys_states_categories,
                                                                                             log_LOS_flag)

        # Store simulated LOS data
        if cutdown == 1.0:  # No intervention
            print('Calculating Simulated LOS...')
            los_tr_nruns = np.array(los_tr_nruns)
            T123A_simulated_los, T123NA_simulated_los, T45_simulated_los = los_tr_nruns[:, 0], los_tr_nruns[:,
                                                                                               1], los_tr_nruns[:, 2]
            T123A_temp_list, T123NA_temp_list, T45_temp_list = [], [], []

            for r in range(nRuns):
                T123A_temp_list.append(T123A_simulated_los[r])
                T123NA_temp_list.append(T123NA_simulated_los[r])
                T45_temp_list.append(T45_simulated_los[r])

            T123A_simulated_mean_los = pd.DataFrame(np.array(T123A_temp_list).transpose())
            T123NA_simulated_mean_los = pd.DataFrame(np.array(T123NA_temp_list).transpose())
            T45_simulated_mean_los = pd.DataFrame(np.array(T45_temp_list).transpose())

            with pd.ExcelWriter('LOS_Data_Simulated_System_State={}.xlsx'.format(sys_state)) as writer:
                T123A_simulated_mean_los.to_excel(writer, sheet_name='T123A')
                T123NA_simulated_mean_los.to_excel(writer, sheet_name='T123NA')
                T45_simulated_mean_los.to_excel(writer, sheet_name='T45')

        # Compute performance measures if flag is True
        mean, median, stdev, P90, CI_mean, rmse = '', '', '', '', '', ''
        if performance_measures_flag:
            z_value = 1.96
            mean, median, stdev, P90, CI_mean, rmse = total_performance_measures(nRuns, los_tr_nruns, z_value)

        end1 = time.time()
        time_to_run1 = round((end1 - start1), 3)
        print("Total time to run simulation for this cutdown percentage ({} runs): {} seconds".format(nRuns,
                                                                                                      time_to_run1))

        # Save the simulation results for the single scenario
        df_results.loc[i] = sys_state, nPatients, nConsultPatients, percentConsult, nRuns, (1 - cutdown) * 100, rmse, mean, median, stdev, CI_mean, P90, time_to_run1

    return df_results


def total_performance_measures(nRuns, los_tr_nruns, z_val):
    """
    Computes the expected mean, expected median, expected standard deviation, expected 90th percentile,
    95% CI for expected mean, RMSE across all the simulations runs. The performance measures are measured separately
    for all patient types.

   @ params:
        nRuns (int): specifies the number of replications for the simulation
        los_tr_nruns (list): contains lists of LOS trackers for n runs / simulation replications, each patient type
            has a separate LOS tracker
        z_val (float): equals 1.96 for computing 95% CI

    @ return:
        mean (list): a list of expected mean values by patient types
        median (list): a list of expected median values by patient types
        stdev (list): a list of expected standard deviation values by patient types
        P90 (list): a list of expected 90th percentile values by patient types
        CI (list): a list of tuples (lower 95% CI for expected mean, upper 95% CI for expected mean) by patient types
        rmse (list): a list of RMSE values by patient types

    """

    data_mean_list = compute_per_run_performance(los_tr_nruns, "mean")
    data_median_list = compute_per_run_performance(los_tr_nruns, "median")
    data_90percentile_list = compute_per_run_performance(los_tr_nruns, "P90")

    mean_list, median_list = np.mean(data_mean_list, axis=0), np.mean(data_median_list, axis=0)
    mean = [round(x, 2) for x in mean_list]
    median = [round(x, 2) for x in median_list]
    ssq = [0, 0, 0]
    for l in data_mean_list:
        ssq[0] += (l[0] - mean[0]) ** 2
        ssq[1] += (l[1] - mean[1]) ** 2
        ssq[2] += (l[2] - mean[2]) ** 2
    variance = [ssq[0] / (nRuns - 1), ssq[1] / (nRuns - 1), ssq[2] / (nRuns - 1)]
    stdev_list = np.sqrt(variance)
    stdev = [round(x, 2) for x in stdev_list]
    P90 = [round(x, 2) for x in np.mean(data_90percentile_list, axis=0)]

    CI = []
    for i in range(len(mean)):
        lower = round(mean[i] - z_val * stdev[i] / np.sqrt(nRuns), 2)
        upper = round(mean[i] + z_val * stdev[i] / np.sqrt(nRuns), 2)
        CI.append((lower, upper))

    data_rmse_list = compute_per_run_performance(los_tr_nruns, "RMSE", data_mean_list)
    rmse = [round(x, 2) for x in np.mean(data_rmse_list, axis=0)]

    return mean, median, stdev, P90, CI, rmse


def compute_per_run_performance(los_tr_nruns, performance, data_means_list=None):
    """
    Computes the mean, median, 90th percentile, or RMSE for each simulation run

    @ params:
        los_tr_nruns (list): contains lists of LOS trackers for n runs / simulation replications, each patient type
            has a separate LOS tracker
        performance (str): specifies the performance measure to compute - "mean", "median", "P90", or "RMSE"
        data_means_list (list, default=None): list of mean values for n simulation runs, needed to compute RMSE

    @ return:
        return_list (list): a list of performance measures for each of the n simulation runs (e.g., if performed
            30 simulation runs and want to compute the median, then the return list will contain 30 median values,
            one for each simulation run)

    """

    n_list = [len(x) for x in los_tr_nruns[0]]
    return_list = []
    for r, l in enumerate(los_tr_nruns):
        temp_list = []

        for idx in range(len(n_list)):

            if performance == "mean":
                temp_list.append(np.mean(l[idx]))
            elif performance == "median":
                temp_list.append(np.median(l[idx]))
            elif performance == "P90":
                temp_list.append(np.percentile(l[idx], q=90))  # manual: int(np.ceil(len(c)*0.9))-1
            elif performance == "RMSE":
                mean = data_means_list[r][idx]
                sum_sq_deviation = 0
                for item in l[idx]:
                    sum_sq_deviation += np.square(item - mean)
                temp_list.append(np.sqrt(sum_sq_deviation / len(l[idx])))

        return_list.append(temp_list)
    return return_list


def naive_calculations_single_cutdown(df_list, cutdown):
    """
    Calculates the naive (baseline) of cutting down consult patients' LOS for a single scenario

    @ params:
        df_list (list): contains 3 np.DataFrame (one for each patient type) with testing features data and outcome
            variable data (patient sojourn time)
        cutdown (float): specifies the percentage in which to cut down consult patients' LOS (value between 0.0 to 1.0)

    @ return:
        mean (list): a list of expected mean values by patient types
        median (list): a list of expected median values by patient types
        stdev (list): a list of expected standard deviation values by patient types
        P90 (list): a list of expected 90th percentile values by patient types
        CI (list): a list of tuples (lower 95% CI for expected mean, upper 95% CI for expected mean) by patient types
        rmse (list): a list of RMSE values by patient types

    """

    los_list = []

    for df_t in df_list:
        los = []
        for j in range(len(df_t)):
            if df_t['Consult'][j] == "Yes":
                los.append(df_t['sojourn_time(minutes)'][j] * cutdown)
            else:
                los.append(df_t['sojourn_time(minutes)'][j])
        los_list.append(los)

    mean, median, stdev, P90 = [], [], [], []
    for l in range(len(los_list)):
        mean.append(round(np.mean(los_list[l]), 2))
        median.append(round(np.median(los_list[l]), 2))
        stdev.append(round(np.std(los_list[l]), 2))
        P90.append(round(np.percentile(los_list[l], q=90), 2))

    return mean, median, stdev, P90


def main_naive_calculations(df_test, interventions):
    """
    Calculates the naive (baseline) of cutting down consult patients' LOS for a single scenario

    @ params:
        df_test (pd.DataFrame): contains testing features data and outcome variable data (patient sojourn time) for
            all patient types
        interventions (list): a list of percentages to cut down consult patients' LOS (value between 0.0 to 1.0)

    @ return:
        df_naive_results (pd.DataFrame): contains the results with performance measures for naive baseline (cutting
            down consult patients' LOS directly from the test data)

    """

    print('\nNaive interventions on actual patient LOS data...')
    df_t123A = df_test[df_test['Triage Category'] == 'T123 Admitted'].reset_index()
    df_t123NA = df_test[df_test['Triage Category'] == 'T123 Not Admitted'].reset_index()
    df_t45 = df_test[df_test['Triage Category'] == 'T45'].reset_index()

    df_list = [df_t123A, df_t123NA, df_t45]
    nPatients, nConsultPatients, percentConsult = [], [], []

    for i, df in enumerate(df_list):
        nPatients.append(len(df))
        nConsultPatients.append(len(df[df['Consult'] == "Yes"]))
        percentConsult.append(round(nConsultPatients[i] / nPatients[i] * 100, 1))

    df_naive_results = pd.DataFrame(
        columns=['nPatients [T123A, T123NA, T45]', 'nConsultPatients [T123A, T123NA, T45]',
                 'percentConsult [T123A, T123NA, T45]',
                 'Cut Down by (%)', 'Mean [T123A, T123NA, T45]', 'Median [T123A, T123NA, T45]',
                 'Stdev [T123A, T123NA, T45]', '90th Percentile [T123A, T123NA, T45]'])

    for i, cutdown in enumerate(interventions):
        print("Consult Patients shorten by {} Percent".format((1 - cutdown) * 100))
        mean, median, stdev, P90 = naive_calculations_single_cutdown(df_list, cutdown)
        df_naive_results.loc[i] = nPatients, nConsultPatients, percentConsult, (
                1 - cutdown) * 100, mean, median, stdev, P90

    return df_naive_results


def main_model_simulation_performance(df, df_train, df_test, sys_state_list, categorical_columns, conts_columns, interventions, log_LOS_flag=True, simulate_flag=True, nRuns=30, performance_measures_flag=True):
    """
    This function is the main LOS model building, simulation, and performance measures calculation function that
    takes in the cleaned data, a list of column features and a list of system states, a list of interventions,
    and the number of simulation replications. The results will be saved in Excel format.

    @ params:
        df (pd.DataFrame):
        df_train (pd.DataFrame):
        df_test (pd.DataFrame):
        sys_state_list (list): specifies the list of system states (used for NIS features selection)
        categorical_columns (list): list of columns names (str) for CATEGORICAL variables
        conts_columns (list): list of columns names (str) for CONTINUOUS variables, except for NIS features
        interventions (list): a list of percentages to cut down consult patients' LOS (value between 0.0 to 1.0)
        log_LOS_flag (bool, default=True): specifies whether model's outcome variable is log(LOS --True or LOS -- False
        simulate_flage (bool, default=True): specifies whether to perform simulation
        nRuns (int, default=30): specifies the number of replications for the simulations
        performance_measures_flag (bool, default=True): specifies whether to compute the performance measures
            for the simulation (mean, median, standard deviation, 90th percentile, 95% confidence interval, RMSE)


        initial_event_calendar (heap): a heap consisting of (ts_, event_, id_) -- timestamp of event, event (a = arrival,
            or d = departure), and id of the patient. This is used to keep track the arrivals and departures of
            patients in the system.
        triage_categories (list): a list of patient types corresponding to each patient in the system
        los_x_tests (list): a list of pd.DataFrame containing features data (testing); (3 dataframes, one for each patient type)
        los_train_errs (list): a list of training prediction errors (3 lists, one for each patient type)
        los_models (list): a list of LOS models (3 models, one for each patient type)
        x_test_zone_dropped_dummy (str): the initial zone that was dropped when converting initial zone into dummy variable
        unique_sys_states_categories (list): a list containing 3 lists - 1) unique patient types, 2) unique initial zones,
            3) unique patient type x initial zone

    @ return:
        df_naive_results (pd.DataFrame): contains the results from naive (baseline) for all interventation levels
        df_results_list (list): list of pd.DataFrames that each contains the simulation results
            for all intervention levels, and for all models
        results_filename (str): the filename in which the results are saved in

    """
    df_results_list = []

    for s, sys_state in enumerate(sys_state_list):
        # Build separate LOS models for each patient type
        categorical_columns_copy, conts_columns_copy = categorical_columns.copy(), conts_columns.copy()
        los_x_train_T123A, los_x_test_T123A, los_y_train_T123A, los_y_test_T123A, model_los_RF_T123A, x_test_zone_dummy = build_los_model(
            df=df, df_train=df_train, df_test=df_test, categorical_cols=categorical_columns_copy, conts_cols=conts_columns_copy,
            triage_category='T123 Admitted', system_state=sys_state, log_LOS=log_LOS_flag, feature_importance_flag=False)

        categorical_columns_copy, conts_columns_copy = categorical_columns.copy(), conts_columns.copy()
        los_x_train_T123NA, los_x_test_T123NA, los_y_train_T123NA, los_y_test_T123NA, model_los_RF_T123NA, x_test_zone_dummy = build_los_model(
            df=df, df_train=df_train, df_test=df_test, categorical_cols=categorical_columns_copy, conts_cols=conts_columns_copy,
            triage_category='T123 Not Admitted', system_state=sys_state, log_LOS=log_LOS_flag, feature_importance_flag=False)

        categorical_columns_copy, conts_columns_copy = categorical_columns.copy(), conts_columns.copy()
        # df_copy, df_train_copy, df_test_copy, categorical_columns_copy = df.copy(), df_train.copy(), df_test.copy(), categorical_columns
        los_x_train_T45, los_x_test_T45, los_y_train_T45, los_y_test_T45, model_los_RF_T45, x_test_zone_dummy = build_los_model(
            df=df, df_train=df_train, df_test=df_test, categorical_cols=categorical_columns_copy, conts_cols=conts_columns_copy,
            triage_category='T45', system_state=sys_state, log_LOS=log_LOS_flag, feature_importance_flag=False)

        # The actual LOS distributions for the 3 patient types only need to be stored once
        if s == 0:
            if log_LOS_flag:
                T123A_actual_los, T123NA_actual_los, T45_actual_los = np.exp(los_y_test_T123A), np.exp(los_y_test_T123NA), np.exp(los_y_test_T45)
            else:
                T123A_actual_los, T123NA_actual_los, T45_actual_los = los_y_test_T123A, los_y_test_T123NA, los_y_test_T45
            np.savetxt("LOS_Dist_Actual_Patient_Type={}.csv".format("T123A"), T123A_actual_los, delimiter=",")
            np.savetxt("LOS_Dist_Actual_Patient_Type={}.csv".format("T123NA"), T123NA_actual_los, delimiter=",")
            np.savetxt("LOS_Dist_Actual_Patient_Type={}.csv".format("T45"), T45_actual_los, delimiter=",")

        # Compute the training prediction errors for the LOS models. These are used to sample noise.
        los_errors_T123A = get_los_errors(model_los_RF_T123A, los_x_train_T123A, los_y_train_T123A)
        los_errors_T123NA = get_los_errors(model_los_RF_T123NA, los_x_train_T123NA, los_y_train_T123NA)
        los_errors_T45 = get_los_errors(model_los_RF_T45, los_x_train_T45, los_y_train_T45)

        # If we'd like to simulate the system, the following code will execute
        if simulate_flag:
            print('\nSimulation start...')

            # Store training errors, test features data, LOS models for all patient types in lists
            los_train_errors = [los_errors_T123A, los_errors_T123NA, los_errors_T45]
            los_x_tests = [los_x_test_T123A.reset_index(drop=True), los_x_test_T123NA.reset_index(drop=True),
                           los_x_test_T45.reset_index(drop=True)]
            los_models = [model_los_RF_T123A, model_los_RF_T123NA, model_los_RF_T45]

            # Store arrivals and patient types in testing data in lists (all patients regardless of patient type)
            all_arrival_times = df_test['patient_arrival_times'].tolist()
            all_patient_types = df_test['Triage Category'].tolist()

            # Retrieves unique patient types, unique initial zones, and unique patient type x initial zone for
            # system states 1, 2, and 3 and stores in lists
            unique_patient_types, unique_zones, unique_patient_type_zones = get_nis_features(df=df, system_state=1)[0], \
                                                                            get_nis_features(df=df, system_state=2)[0], \
                                                                            get_nis_features(df=df, system_state=3)[0]
            unique_sys_states_categories = [unique_patient_types, unique_zones, unique_patient_type_zones]

            # Initialize the event calendar based on patient arrivals
            initial_event_calendar = [(a, 'a', i) for i, a in enumerate(all_arrival_times)]
            hq.heapify(initial_event_calendar)

            # Initialize the DataFrame to store simulation results
            df_results = main_simulate(sys_state=sys_state, interventions=interventions, nRuns=nRuns,
                                       initial_event_calendar=initial_event_calendar,
                                       triage_categories=all_patient_types, los_x_tests=los_x_tests,
                                       los_train_errs=los_train_errors, los_models=los_models,
                                       x_test_zone_dropped_dummy=x_test_zone_dummy,
                                       unique_sys_states_categories=unique_sys_states_categories,
                                       performance_measures_flag=performance_measures_flag)
            df_results_list.append(df_results)

    # Computes naive (baseline) results
    df_naive_results = main_naive_calculations(df_test, interventions)

    # Saving results from all system states for all intervention levels (naive baseline and simulation)
    print('Saving results...')
    results_filename = '00_Naive+Simulation_Results.xlsx'
    with pd.ExcelWriter(results_filename) as writer:
        df_naive_results.to_excel(writer, sheet_name='Naive')
        for i, df in enumerate(df_results_list):
            df.to_excel(writer, sheet_name='System_State_{}'.format(sys_state_list[i]))

    return df_naive_results, df_results_list, results_filename
