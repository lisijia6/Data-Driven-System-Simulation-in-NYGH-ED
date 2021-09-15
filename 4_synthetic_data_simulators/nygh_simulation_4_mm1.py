import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import heapq as hq
import time
import os


def build_models(df):
    print("\nBuilding LOS Model...")
    df['log_LOS'] = df.apply(lambda row: np.log(row['LOS']), axis=1)
    df_los_train, df_los_test = df[1000:3000], df[3000:4000]

    # los_x_train, los_x_test = df_los_train[['Type_No_Consult', 'NIS']], df_los_test[['Type_No_Consult', 'NIS']]
    los_x_train, los_x_test = df_los_train[['Type_No_Consult', 'NIS_consult', 'NIS_no_consult']], df_los_test[['Type_No_Consult', 'NIS_consult', 'NIS_no_consult']]
    los_y_train, los_y_test = df_los_train[['log_LOS']], df_los_test[['log_LOS']]
    los_y_train = np.array(los_y_train).reshape(len(los_y_train), )
    los_y_test = np.array(los_y_test).reshape(len(los_y_test), )
    print('x_train.shape, x_test.shape: ', los_x_train.shape, los_x_test.shape)
    print('y_train.shape, y_test.shape: ', los_y_train.shape, los_y_test.shape)

    model_los_RF = RandomForestRegressor(min_samples_leaf=30, max_depth=None, n_estimators=100, random_state=0)
    model_los_RF.fit(los_x_train, los_y_train)

    # Get numerical feature importances
    importances = list(model_los_RF.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(['Type_No_Consult', 'NIS_consult', 'NIS_no_consult'], importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances]

    return los_x_train, los_x_test, los_y_train, los_y_test, model_los_RF


def sample_from_RF_regressor(model, sample, error_list):
    los_RF_pred = model.predict(np.array(sample).reshape(1, -1))
    error = np.random.choice(error_list, 1)
    return np.exp(los_RF_pred+error)


def get_los_errors(model, x_train, y_train):
    y_preds = model.predict(np.array(x_train))
    predicted = pd.Series(data=y_preds, index=x_train.index)
    actual = pd.Series(data=y_train, index=x_train.index)
    errors = []
    for i in range(len(actual)):
        err = actual.iloc[i] - predicted.iloc[i]
        errors.append(err)
    return errors


def simulate_nygh_n_servers(nruns, cutdown, initial_event_calendar, los_test, los_model, los_errors, nservers):
    # Currently, the nservers feature is not used
    np.random.seed(3)  # set random seed
    los_test = los_test.reset_index()
    los_test.drop(columns=['index'], inplace=True)
    los_tr_nruns, nCustomers, n_consult = [], len(los_test), 0

    for r in range(nruns):
        print('running simulation run #: ' + str(r + 1))
        event_calendar = initial_event_calendar.copy()

        los_tr = []
        curr_nis, idx_ = (0, 0), 0
        while len(list(event_calendar)) > 0:
            ts_, event_, id_ = hq.heappop(event_calendar)  # take an event from the event_calendar
            # arrival event happens, need to check if servers are available
            if event_ == 'a':
                # curr_nis += 1  # update current number of customers in the system
                if id_ % 1000 == 0:
                    print('arrival: ', id_)

                sample = los_test.iloc[idx_]
                if sample['Type_No_Consult'] == 0:  # consult patient
                    curr_nis = (curr_nis[0] + 1, curr_nis[1])
                else:
                    curr_nis = (curr_nis[0], curr_nis[1] + 1)

                los_test.at[idx_, 'NIS_consult'] = curr_nis[0]
                los_test.at[idx_, 'NIS_no_consult'] = curr_nis[1]
                # curr_nis += 1
                # los_test.at[idx_, 'NIS'] = curr_nis
                sample = los_test.iloc[idx_]
                idx_ += 1

                los = sample_from_RF_regressor(los_model, sample, los_errors)
                if sample['Type_No_Consult'] == 0:  # consult patient
                    los = los * cutdown
                    if r == 0:
                        n_consult += 1
                los_tr.append(los[0])
                d = ts_ + los[0]
                hq.heappush(event_calendar, (d, 'd', id_))

            # departure event happens
            else:
                if los_test.at[id_, 'Type_No_Consult'] == 0:  # consult patient
                    curr_nis = (curr_nis[0] - 1, curr_nis[1])
                else:
                    curr_nis = (curr_nis[0], curr_nis[1] - 1)
                # curr_nis -= 1  # update current number of customers in the system

        print("curr_nis: ", curr_nis)
        los_tr_nruns.append(los_tr)
        print("Total time for this run: {} seconds".format(time_to_run3))
        print("nConsultPatients: ", n_consult)

    percentConsult = round((n_consult / nCustomers) * 100, 1)
    return los_tr_nruns, nCustomers, n_consult, percentConsult


def simulate_nygh_infinite_servers(nruns, cutdown, initial_event_calendar, los_test, los_model, los_errors, nservers):
    # Currently, the nservers feature is not used
    np.random.seed(3)  # set random seed
    los_test = los_test.reset_index()
    los_test.drop(columns=['index'], inplace=True)
    los_tr_nruns, nCustomers, n_consult = [], len(los_test), 0

    for r in range(nruns):
        print('running simulation run #: ' + str(r + 1))
        start3 = time.time()
        event_calendar = initial_event_calendar.copy()

        los_tr = []
        curr_nis, idx_ = (0, 0), 0
        while len(list(event_calendar)) > 0:
            ts_, event_, id_ = hq.heappop(event_calendar)  # take an event from the event_calendar
            # arrival event happens, need to check if servers are available
            if event_ == 'a':
                # curr_nis += 1  # update current number of customers in the system
                if id_ % 1000 == 0:
                    print('arrival: ', id_)

                sample = los_test.iloc[idx_]
                if sample['Type_No_Consult'] == 0:  # consult patient
                    curr_nis = (curr_nis[0] + 1, curr_nis[1])
                else:
                    curr_nis = (curr_nis[0], curr_nis[1] + 1)

                los_test.at[idx_, 'NIS_consult'] = curr_nis[0]
                los_test.at[idx_, 'NIS_no_consult'] = curr_nis[1]
                # curr_nis += 1
                # los_test.at[idx_, 'NIS'] = curr_nis
                sample = los_test.iloc[idx_]
                idx_ += 1

                los = sample_from_RF_regressor(los_model, sample, los_errors)
                if sample['Type_No_Consult'] == 0:  # consult patient
                    los = los * cutdown
                    if r == 0:
                        n_consult += 1
                los_tr.append(los[0])
                d = ts_ + los[0]
                hq.heappush(event_calendar, (d, 'd', id_))

            # departure event happens
            else:
                if los_test.at[id_, 'Type_No_Consult'] == 0:  # consult patient
                    curr_nis = (curr_nis[0] - 1, curr_nis[1])
                else:
                    curr_nis = (curr_nis[0], curr_nis[1] - 1)
                # curr_nis -= 1  # update current number of customers in the system

        print("curr_nis: ", curr_nis)
        los_tr_nruns.append(los_tr)
        end3 = time.time()
        time_to_run3 = round((end3 - start3), 3)
        print("Total time for this run: {} seconds".format(time_to_run3))
        print("nConsultPatients: ", n_consult)

    percentConsult = round((n_consult/nCustomers)*100, 1)
    return los_tr_nruns, nCustomers, n_consult, percentConsult


def total_performance_measures(nRuns, los_tr_nruns, z_val):
    data_mean_list = compute_per_run_performance(los_tr_nruns, "mean")
    data_median_list = compute_per_run_performance(los_tr_nruns, "median")
    data_90percentile_list = compute_per_run_performance(los_tr_nruns, "P90")

    mean = round(np.mean(data_mean_list), 2)
    median = round(np.mean(data_median_list), 2)
    ssq = 0
    for l in data_mean_list:
        ssq += (l - mean) ** 2
    variance = ssq / (nRuns - 1)
    stdev = round(np.sqrt(variance), 2)
    P90 = round(np.mean(data_90percentile_list), 2)

    lower = round(mean - z_val * stdev / np.sqrt(nRuns), 2)
    upper = round(mean + z_val*stdev/np.sqrt(nRuns), 2)
    CI = (lower, upper)

    data_rmse_list = compute_per_run_performance(los_tr_nruns, "RMSE", data_mean_list)
    rmse = round(np.mean(data_rmse_list), 2)

    return mean, median, stdev, P90, CI, rmse


def compute_per_run_performance(los_tr_nruns, performance, data_means_list=None):
    return_list = []
    for r, l in enumerate(los_tr_nruns):
        if performance == "mean":
            return_list.append(np.mean(l))
        elif performance == "median":
            return_list.append(np.median(l))
        elif performance == "P90":
            return_list.append(np.percentile(l, q=90))  # manual: int(np.ceil(len(c)*0.9))-1
        elif performance == "RMSE":
            mean = data_means_list[r]
            sum_sq_deviation = 0
            for item in l:
                sum_sq_deviation += np.square(item - mean)
            return_list.append(np.sqrt(sum_sq_deviation/len(l)))
    # print('return_list', return_list)
    return return_list


if __name__ == "__main__":
    # ------------------------------------------PRE-PROCESSING DATA------------------------------------------
    start1 = time.time()
    print('Reading data...')
    df = pd.read_csv('MM1_baseline_separate_nis.csv')
    df = df[df['LOS']>0].reset_index()
    print('Splitting df_train and df_test sets...')
    df_train, df_test = df[1000:3000], df[3000:4000]  # modify index to specify training and testing data
    print("df_test length", len(df_test))

    # ------------------------------------------BUILDING SAMPLING MODELS------------------------------------------
    print('Building LOS models...')
    arrivals = df_test['arrival_time'].tolist()
    los_x_train, los_x_test, los_y_train, los_y_test, model_los_RF = build_models(df)
    actual_los = np.exp(los_y_test)
    np.savetxt("LOS_Dist_Actual_MM1_2.csv", actual_los, delimiter=",")

    print("Final dataframes...")

    los_errors = get_los_errors(model_los_RF, los_x_train, los_y_train)

    end1 = time.time()
    time_to_run1 = round((end1 - start1))
    print("Pre-processing time: {} seconds".format(time_to_run1))

    simulate_Flag= True
    nServers = 1
    if simulate_Flag:
        df_results = pd.DataFrame(
            columns=['nPatients', 'nConsultPatients', 'percentConsult', 'nRuns', 'Cut Down by (%)', 'RMSE',
                     'Mean', 'Median', 'Stdev', 'CI on Mean', '90th Percentile', 'Time to Run (seconds)'])

        # ------------------------------------------STARTING SIMULATION------------------------------------------
        print('Simulation start...')
        # cutdown_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        cutdown_percentage = [0.5, 1.0]

        initial_event_calendar = [(a, 'a', i) for i, a in enumerate(arrivals)]
        hq.heapify(initial_event_calendar)

        # NO TOLERANCE
        nRuns = 3
        los_x_test = los_x_test.reset_index(drop=True)

        for i, cutdown in enumerate(cutdown_percentage):
            print("Consult Patients shorten by {} Percent".format((1 - cutdown) * 100))
            print("{} Runs...".format(nRuns))
            start2 = time.time()
            start3 = time.time()
            los_tr_nruns, nPatients, nConsultPatients, percentConsult = simulate_nygh_infinite_servers(nRuns, cutdown,
                                                                                 initial_event_calendar, los_x_test,
                                                                                 model_los_RF, los_errors, nServers)

            end3 = time.time()
            time_to_run3 = round((end3 - start3), 3)
            print("Time to run the /simulate/ function: {} seconds".format(time_to_run3))

            z_value = 1.96
            mean, median, stdev, P90, CI_mean, rmse = total_performance_measures(nRuns, los_tr_nruns, z_value)
            # print(mean, median, stdev, P90, CI_mean, rmse)

            if cutdown == 1.0:

                print('Calculating Simulated LOS')
                los_tr_nruns = np.array(los_tr_nruns)
                simulated_los = los_tr_nruns[:, 0]
                temp_list = []
                for r in range(nRuns):
                    temp_list.append(simulated_los[r])
                print(temp_list)
                simulated_los = pd.DataFrame(np.array(temp_list).transpose())
                with pd.ExcelWriter('LOS_Data_Simulated_2.xlsx') as writer:
                    simulated_los.to_excel(writer, sheet_name='MM{}_Experiment'.format(nServers))

            end2 = time.time()
            time_to_run2 = round((end2 - start2), 3)
            print("Total time to run simulation for this cutdown percentage ({} runs): {} seconds".format(nRuns, time_to_run2))
            df_results.loc[i] = nPatients, nConsultPatients, percentConsult, nRuns, (1 - cutdown) * 100, rmse, mean, median, stdev, CI_mean, P90, time_to_run2

        # ------------------------------------------SAVING RESULTS------------------------------------------
        print('Saving results...')
        save_path_results = os.path.join(os.getcwd(), "Consult_Reduction_Results_MM1.csv")
        df_results.to_csv (save_path_results, index = False, header=True)
