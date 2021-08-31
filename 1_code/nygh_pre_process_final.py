import pandas as pd
import os
from datetime import datetime
import heapq as hq
import itertools
import holidays


def categorize_patients(df, categorical_columns):
    """
    Categorizes patients into 3 categories (T123 Admitted, T123 Not Admitted, and T45),
    and adds this information in a new column in the DataFrame

    @ params:
        df (pd.DataFrame): the DataFrame to add new column with patient categories
        categorical_columns (list): list of categorical variables' column names to keep track of

    @ return:
        df_return (pd.DataFrame): the DataFrame after new column is added
        categorical_columns (list): list of categorical variables' column names to keep track of with new columns added
    """

    columns = df.columns
    # Add a new column called 'Admission' based on the 'Discharge Disposition Description' column:
    # "Admit" if discharge disposition starts with Admit, otherwise "Not admitted"
    if 'Discharge Disposition Description' in columns:
        temp_list = []
        for _, val in df['Discharge Disposition Description'].items():
            if val.split(" ", 1)[0] == "Admit":
                val = "Admitted"
            else:
                val = "Not Admitted"
            temp_list.append(val)
        df['Admission'] = temp_list
        categorical_columns.append('Admission')
        df.drop(columns=['Discharge Disposition Description'], inplace=True)
        columns = df.columns

    # Categorize patients based on triage code and admission
    if 'Triage Code' in columns and 'Admission' in columns:
        # Triage score (get rid of 9.0; categorize)
        df = df[df['Triage Code'] != 9.0]
        # T123: 'Resuscitation, Emergent & Urgent', T45: 'Less Urgent & Non-Urgent'
        df.loc[(df['Admission'] == 'Admitted') & ((df['Triage Code'] == 1) | (df['Triage Code'] == 2) | (
                df['Triage Code'] == 3)), 'Triage Category'] = 'T123 Admitted'
        df.loc[(df['Admission'] == 'Not Admitted') & ((df['Triage Code'] == 1) | (df['Triage Code'] == 2) | (
                df['Triage Code'] == 3)), 'Triage Category'] = 'T123 Not Admitted'
        df.loc[(df['Triage Code'] == 4) | (df['Triage Code'] == 5), 'Triage Category'] = 'T45'
        df.drop(columns=['Triage Code'], inplace=True)
        categorical_columns.append('Triage Category')

    return df, categorical_columns


def check_float_and_convert_to_datetime(input_val):
    """
    Converts a given string to DateTime format
    """
    if type(input_val) != float:
        return datetime.strptime(input_val, "%Y-%m-%d %H:%M:%S")


def max_arrival_time(ambulance, triage):
    """
    Adds a new column "patient_arrival_times" by taking MAX(ambulance arrival datetime, triage datetime)
    """
    if type(ambulance) == type(pd.NaT):
        return triage
    else:
        return max(ambulance, triage)


def add_arrival_columns(df, yrs, categorical_columns):
    """
    Adds columns with information related to patient arrival time
    """
    print('Arrival: hour, day_of_week, week_number, month, year, holidays ...')
    df['arrival_hour'] = df.apply(lambda row: row['patient_arrival_times'].hour, axis=1)
    df['arrival_day_of_week'] = df.apply(lambda row: row['patient_arrival_times'].weekday(), axis=1)
    df['arrival_week_number'] = df.apply(lambda row: row['patient_arrival_times'].isocalendar()[1], axis=1)
    df['arrival_month'] = df.apply(lambda row: row['patient_arrival_times'].month, axis=1)
    df['arrival_year'] = df.apply(lambda row: row['patient_arrival_times'].year, axis=1)
    # for date, name in sorted(holidays.CA(years=[2016,2017,2018], prov='ON').items()):
    #     print(date, name)
    holidays_can_on = list(holidays.CA(years=yrs, prov='ON').keys())
    df['holiday_CAN_ON'] = df.apply(lambda row: check_holidays(row['patient_arrival_times'], holidays_can_on), axis=1)

    categorical_columns.extend(
        ['arrival_hour', 'arrival_day_of_week', 'arrival_week_number', 'arrival_month', 'arrival_year',
         'holiday_CAN_ON'])
    df = df.reset_index()
    df.drop(columns=['index'], inplace=True)
    return df, categorical_columns


def check_holidays(date, holidays_can_on):
    """
    Checks whether the given date is in the list of holidays provided and returns either 0 or 1
    """
    if date in holidays_can_on:
        return 1
    else:
        return 0


def compute_nis_features(df):
    """
    Computes NIS features for all system states with the following high-level procedure:
        - Add arrival & departure events to event calendar, organize events based on timestamp.
        - Keep a counter called curr_nis.
            - If event is arrival, curr_nis+1 and the arriving patient's NIS upon arrival is set to this curr_nis+1.
            - If event is departure, curr_nis-1.

    @ params:
        df (pd.DataFrame): the DataFrame to add new columns with NIS information for all system states

    @ return:
        df (pd.DataFrame): the DataFrame with NIS features added
    """

    print("Computing NIS...")

    unique_zones = df['Initial Zone'].unique()
    unique_patient_types = df['Triage Category'].unique()
    unique_patient_type_zones = list(itertools.product(unique_patient_types, unique_zones))

    # Initialize system state 0: general NIS
    df['NIS Upon Arrival'] = 0

    # Initialize system state 1: NIS by patient type
    pt_features, nis_by_patient_type = [], ()
    for _, pt in enumerate(unique_patient_types):
        df['NIS Upon Arrival Patient_{}'.format(pt)] = 0
        pt_features.append('NIS Upon Arrival Patient_{}'.format(pt))
        nis_by_patient_type += (0,)

    # Initialize system state 2: NIS by zone
    zone_features, nis_by_zone = [], ()
    for _, zone in enumerate(unique_zones):
        df['NIS Upon Arrival Zone_{}'.format(zone)] = 0
        zone_features.append('NIS Upon Arrival Zone_{}'.format(zone))
        nis_by_zone += (0,)

    # Initialize system state 3: NIS by patient type x zone
    pt_zone_features, nis_by_pt_zone = [], ()
    for _, pt_zone in enumerate(unique_patient_type_zones):
        df['NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1])] = 0
        pt_zone_features.append('NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1]))
        nis_by_pt_zone += (0,)

    arrival_times = df['patient_arrival_times'].tolist()
    departure_times = df['Left ED DateTime'].tolist()

    arrival_times = [(arrival_time, patient_number, 'a') for patient_number, arrival_time in
                     enumerate(arrival_times)]
    departure_times = [(departure_time, patient_number, 'd') for patient_number, departure_time in
                       enumerate(departure_times)]

    event_calendar = arrival_times + departure_times
    hq.heapify(event_calendar)

    curr_nis = 0
    curr_nis_by_pt, curr_nis_by_zone, curr_nis_by_pt_zone = nis_by_patient_type, nis_by_zone, nis_by_pt_zone

    while len(event_calendar) != 0:
        timestamp, id_, event_type = hq.heappop(event_calendar)
        if event_type == 'a':
            # System State = 0
            curr_nis += 1
            df.at[id_, 'NIS Upon Arrival'] = curr_nis

            # System State = 1
            if df['Triage Category'][id_] == 'T123 Admitted':
                curr_nis_by_pt = (curr_nis_by_pt[0] + 1, curr_nis_by_pt[1], curr_nis_by_pt[2])
            elif df['Triage Category'][id_] == 'T123 Not Admitted':
                curr_nis_by_pt = (curr_nis_by_pt[0], curr_nis_by_pt[1] + 1, curr_nis_by_pt[2])
            elif df['Triage Category'][id_] == 'T45':
                curr_nis_by_pt = (curr_nis_by_pt[0], curr_nis_by_pt[1], curr_nis_by_pt[2] + 1)
            df.at[id_, 'NIS Upon Arrival Patient_T123 Admitted'] = curr_nis_by_pt[0]
            df.at[id_, 'NIS Upon Arrival Patient_T123 Not Admitted'] = curr_nis_by_pt[1]
            df.at[id_, 'NIS Upon Arrival Patient_T45'] = curr_nis_by_pt[2]

            # System State = 2
            temp_nis_by_zone = ()
            for i, zone in enumerate(unique_zones):
                if df['Initial Zone'][id_] == zone:
                    temp_nis_by_zone = temp_nis_by_zone + (curr_nis_by_zone[i] + 1,)
                    df.at[id_, 'NIS Upon Arrival Zone_{}'.format(zone)] = curr_nis_by_zone[i] + 1
                else:
                    temp_nis_by_zone = temp_nis_by_zone + (curr_nis_by_zone[i],)
                    df.at[id_, 'NIS Upon Arrival Zone_{}'.format(zone)] = curr_nis_by_zone[i]
            curr_nis_by_zone = temp_nis_by_zone

            # System State = 3
            temp_nis_by_pt_zone = ()
            for j, pt_zone in enumerate(unique_patient_type_zones):
                if (df['Triage Category'][id_], df['Initial Zone'][id_]) == pt_zone:
                    temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis_by_pt_zone[j] + 1,)
                    df.at[id_, 'NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1])] = \
                    curr_nis_by_pt_zone[j] + 1
                else:
                    temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis_by_pt_zone[j],)
                    df.at[id_, 'NIS Upon Arrival Patient_{} Zone_{}'.format(pt_zone[0], pt_zone[1])] = \
                    curr_nis_by_pt_zone[j]
            curr_nis_by_pt_zone = temp_nis_by_pt_zone

        elif event_type == 'd':
            # System State = 0
            curr_nis -= 1

            # System State = 1
            if df['Triage Category'][id_] == 'T123 Admitted':
                curr_nis_by_pt = (curr_nis_by_pt[0] - 1, curr_nis_by_pt[1], curr_nis_by_pt[2])
            elif df['Triage Category'][id_] == 'T123 Not Admitted':
                curr_nis_by_pt = (curr_nis_by_pt[0], curr_nis_by_pt[1] - 1, curr_nis_by_pt[2])
            elif df['Triage Category'][id_] == 'T45':
                curr_nis_by_pt = (curr_nis_by_pt[0], curr_nis_by_pt[1], curr_nis_by_pt[2] - 1)

            # System State = 2
            temp_nis_by_zone = ()
            for i, zone in enumerate(unique_zones):
                if df['Initial Zone'][id_] == zone:
                    temp_nis_by_zone = temp_nis_by_zone + (curr_nis_by_zone[i] - 1,)
                else:
                    temp_nis_by_zone = temp_nis_by_zone + (curr_nis_by_zone[i],)
            curr_nis_by_zone = temp_nis_by_zone

            # System State = 3
            temp_nis_by_pt_zone = ()
            for j, pt_zone in enumerate(unique_patient_type_zones):
                if (df['Triage Category'][id_], df['Initial Zone'][id_]) == pt_zone:
                    temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis_by_pt_zone[j] - 1,)
                else:
                    temp_nis_by_pt_zone = temp_nis_by_pt_zone + (curr_nis_by_pt_zone[j],)
            curr_nis_by_pt_zone = temp_nis_by_pt_zone
    return df


def convert_columns_to_category_type(df, categorical_columns):
    """
    Converts columns of categorical features to 'category' type in the DataFrame
    """
    for col in categorical_columns:
        df[col] = df[col].astype("category")
    return df


def main_preprocess_data(filename, columns, years, cleaned_data_filename, write_data):
    """
    This function is the main data pre-processing function that reads in the raw data,
    cleans the raw data columns (sometimes call other functions in this script),
    and if write_data is True, the cleaned DataFrame will be saved in Excel format.

    @ params:
        filename (str): name of the file in which the raw data is saved at
        columns (list): list of column names (str) in the raw data to be pre-processed
        years (list): list of years (int) to be pre-proceesed
        cleaned_data_filename (str): the filename that will be used to save cleaned data if write_data is True
        write_data (bool): True means will save cleaned data in Excel file, False means don't save

    @ return:
        df: the DataFrame after data pre-processing
    """

    print('Pre-Processing...')
    filepath = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(filepath)
    print("DataFrame columns shape and names (before pre-processing): ", df.shape, df.columns)
    df = df[columns]
    categorical_columns = []

    # Categorize age into 4 age groups
    if 'Age (Registration)' in columns:
        df = df[df['Age (Registration)'].notna()]  # drop columns with null values
        # Categorize 'Age'
        df.loc[(df['Age (Registration)'] >= 0) & (df['Age (Registration)'] <= 14), 'Age Category'] = 'Children'
        df.loc[(df['Age (Registration)'] >= 15) & (df['Age (Registration)'] <= 24), 'Age Category'] = 'Youth'
        df.loc[(df['Age (Registration)'] >= 25) & (df['Age (Registration)'] <= 64), 'Age Category'] = 'Adult'
        df.loc[df['Age (Registration)'] >= 65, 'Age Category'] = 'Seniors'
        df.drop(columns=['Age (Registration)'], inplace=True)
        categorical_columns.append('Age Category')

    # Drop rows with null values and unknown gender
    if 'Gender Code' in columns:
        df = df[df['Gender Code'].notna()]
        df = df[df['Gender Code'] != 'U']
        categorical_columns.append('Gender Code')

    # Ambulance: "Yes" if "Ambulance Arrival DateTime" column is not null, "No" otherwise
    if 'Ambulance Arrival DateTime' in columns:
        df.loc[df['Ambulance Arrival DateTime'].isnull() == True, 'Ambulance'] = 'No'
        df.loc[df['Ambulance Arrival DateTime'].isnull() == False, 'Ambulance'] = 'Yes'
        categorical_columns.append('Ambulance')

    # Consult: "Yes" if "Consult Service Description (1st)" column is not null, "No" otherwise¶
    if 'Consult Service Description (1st)' in columns:
        df.loc[df['Consult Service Description (1st)'].isnull() == True, 'Consult'] = 'No'
        df.loc[df['Consult Service Description (1st)'].isnull() == False, 'Consult'] = 'Yes'
        categorical_columns.append('Consult')

    # Fill in missing values with "U" as unknown for initial zone
    if 'Initial Zone' in columns:
        df['Initial Zone'].fillna(value='U', inplace=True)
        categorical_columns.append('Initial Zone')

    # Categorize patients by triage code and by admission (T123 admitted, T123 not admitted, T45)
    df, categorical_columns = categorize_patients(df, categorical_columns)

    # Convert column values to DateTime format
    df['Ambulance Arrival DateTime'] = df.apply(lambda row: check_float_and_convert_to_datetime(row['Ambulance Arrival DateTime']), axis=1)
    df['Triage DateTime'] = df.apply(lambda row: check_float_and_convert_to_datetime(row['Triage DateTime']), axis=1)
    df['Left ED DateTime'] = df.apply(lambda row: check_float_and_convert_to_datetime(row['Left ED DateTime']), axis=1)
    # Define patient arrival times:
    df['patient_arrival_times'] = df.apply(
        lambda row: max_arrival_time(row['Ambulance Arrival DateTime'], row['Triage DateTime']), axis=1)
    # Add a new column "sojourn_times(minutes)" by calculating (left ED datetime - patient arrival times)
    df['sojourn_time(minutes)'] = df.apply(
        lambda row: (row['Left ED DateTime'] - row['patient_arrival_times']).seconds // 60, axis=1)
    # Select "sojourn_times(minutes)" that are larger than 0
    df = df[df['sojourn_time(minutes)'] > 0].reset_index()

    df = df.reset_index()
    df.drop(columns=['index'], inplace=True)

    df, categorical_columns = add_arrival_columns(df, years, categorical_columns)
    df = compute_nis_features(df)
    df = convert_columns_to_category_type(df, categorical_columns)

    print("Categorical columns: ", categorical_columns)
    print("DataFrame columns shape and names (after pre-processing): ", df.shape, df.columns)
    print("First 5 columns: ", df.head())

    if write_data:
        print('Saving Results...')
        save_path = os.path.join(os.getcwd(), cleaned_data_filename)
        df.to_excel(save_path, engine='xlsxwriter')

    return df