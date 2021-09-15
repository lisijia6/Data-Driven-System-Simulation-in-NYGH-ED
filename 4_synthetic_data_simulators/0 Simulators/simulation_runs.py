from q_intervention import *
from plotting import *

def sim2run(nCustomer, nReplications):
    """
    Assumptions:
    1) one type of customer
    2) fixed arrival rate: lambda = 1
    3) fixed service rate before intervention: mu = 1.1
    4) 4 types of performance measures: Wait Time in Queue, Time in System, Number in Queue, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2, 2.5, 3])
    p_intervention_list = np.linspace(0, 1, 11)
    count = 0

    # Calculate the value of rho, run the simulation only if rho < 1
    # rho = lam / mu
    # if rho >= 1:
    #     return

    # keep track of statistics, 3 values of mu, 11 values of P(intervention)
    tis_mean, wiq_mean, niq_mean, nis_mean = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_sd, wiq_sd, niq_sd, nis_sd = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_90per, wiq_90per, niq_90per, nis_90per = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))

    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0], probs = [1.0],
                         mus = [mu], prob_speedup=[p_interv], mus_speedup=[mu_prime],
                         servers = 1)

            q_.simulate_q(customers=nCustomer, runs=nReplications)
            q_.generate_data(sla_=0.9, quant_flag=True, write_file=True)

            cwd = os.getcwd()  # get current working directory
            folder = "Lambda{}Mu{}ProbIntervention{}MuSpeedup{}".format(lam, mu, p_interv, mu_prime)
            directory = os.path.join(cwd, folder)

            print('Simulation Run #', count+1)

            average_tis, average_tis_sd, percentiles_all_tis = system_time(directory)
            tis_mean[i-1, j-1] = average_tis
            tis_sd[i-1, j-1] = average_tis_sd
            tis_90per[i-1, j-1] = percentiles_all_niq[8]  # 90th percentile

            average_wiq, average_wiq_sd, percentiles_all_wiq = queue_waiting(directory)
            wiq_mean[i-1, j-1] = average_wiq
            wiq_sd[i-1, j-1] = average_wiq_sd
            wiq_90per[i-1, j-1] = percentiles_all_wiq[8]

            average_niq, average_niq_sd, percentiles_all_niq = queue_number(directory)
            niq_mean[i-1, j-1] = average_niq
            niq_sd[i-1, j-1] = average_niq_sd
            niq_90per[i-1, j-1] = percentiles_all_niq[8]

            average_nis, average_nis_sd, percentiles_all_nis = system_number(directory)
            nis_mean[i-1, j-1] = average_nis
            niq_sd[i-1, j-1] = average_nis_sd
            nis_90per[i-1, j-1] = percentiles_all_nis[8]

    plotting.plot_mean(tis_mean, tis_sd, performance='tis')

def calculate_sd(data_list):
    sd_list = []
    for data in data_list:
        N = len(data)
        var = ((np.std(data))**2)*(N/(N-1))
        sd_list.append(np.sqrt(var))
    return sd_list

# helper function to calculate the average number in queue or system
def calculate_average_number(data, queue=True):
    total_num_list = []
    col = 'Number_in_Queue' if queue else 'Number_in_System'
    t_n = data.at[len(data) - 1, 'timestamp']
    # t_n = data_list[0].at[len(data_list[0]) - 1, 'timestamp']
    # for data in data_list:
    total_num = 0
    for i in range(1, len(data)):
        area = (data.at[i, 'timestamp'] - data.at[i-1, 'timestamp']) * data.at[i, col]
        total_num += area
    total_num_list.append(total_num / t_n)
    return total_num_list

# calculate percentiles
def calculate_percentiles(data):
    percentile_all = [np.percentile(a=data, q=x) for x in range(10, 101, 10)]
    return percentile_all

def queue_waiting(directory):
    _, df_all_wiq = read_in_csv(directory, "data_WIQ_TIS")
    # Mean values and standard deviation calculations, stored as [all, class 1, class 2, ...]
    average_wiq = [np.mean(df_all_wiq['elapsed'])]
    average_wiq_sd = calculate_sd([df_all_wiq['elapsed']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_wiq = calculate_percentiles(df_all_wiq['elapsed'])
    return average_wiq, average_wiq_sd, all_wiq

def system_time(directory):
    df_all_tis, _ = read_in_csv(directory, "data_WIQ_TIS")
    # Mean values and standard deviation calculations, stored as [all, class 1, class 2, ...]
    average_tis = [np.mean(df_all_tis['elapsed'])]
    average_tis_sd = calculate_sd([df_all_tis['elapsed']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_tis = calculate_percentiles(df_all_tis['elapsed'])
    return average_tis, average_tis_sd, all_tis

def queue_number(directory):
    df_all_niq = read_in_csv(directory, "data_NIQ")
    # Mean values and standard deviation calculations, stored ad [all, class 1, class 2, ...]
    # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n
    average_niq = calculate_average_number(df_all_niq, True)
    average_niq_sd = calculate_sd([df_all_niq['Number_in_Queue']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_niq = calculate_percentiles(df_all_niq['Number_in_Queue'])
    return average_niq, average_niq_sd, all_niq

def system_number(directory):
    df_all_nis = read_in_csv(directory, "data_NIS")
    # Mean values and standard deviation calculations, stored ad [all, class 1, class 2, ...]
    # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n
    average_nis = calculate_average_number(df_all_nis, False)
    average_nis_sd = calculate_sd([df_all_nis['Number_in_System']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_nis = calculate_percentiles(df_all_nis['Number_in_System'])
    return average_nis, average_nis_sd, all_nis

def read_in_csv(directory, filename):

    if filename == "data_WIQ_TIS":
        # file_1: System/Time (TIS) and Queue/Waiting (WIQ)
        file_path_1 = os.path.join(directory, filename + ".csv")
        df_all = pd.read_csv(file_path_1, header=0)
        # condition statements
        TIS, WIQ = df_all.event_type == 'd', df_all.event_type == 's'
        # create DataFrames for TIS and WIQ
        df_all_tis, df_all_wiq = df_all[TIS], df_all[WIQ]
        return df_all_tis, df_all_wiq

    elif filename == "data_NIQ":
        # file_2: Queue/Number (NIQ)
        file_path_2 = os.path.join(directory, filename + ".csv")
        df_all_niq = pd.read_csv(file_path_2, header=0)
        return df_all_niq

    elif filename == "data_NIS":
        # file_3: System/Number
        file_path_3 = os.path.join(directory, filename + ".csv")
        df_all_nis = pd.read_csv(file_path_3, header=0)
        return df_all_nis
    else:
        return "ERROR: filename not found"

if __name__ == "__main__":
    nCust, nReps = 1000, 10
    sim2run(nCust, nReps)