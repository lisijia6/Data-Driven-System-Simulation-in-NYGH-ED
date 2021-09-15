from q_intervention import *
import matplotlib.pyplot as plt
from q_priority import *

def simulation(nCustomer, nReplications):
    """
    Four types of performance measures: Wait Time in Queue, Time in System, Number in Queue, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2])
    p_intervention_list = np.linspace(0, 1, 11)

    # Keep track of statistics, 3 values of mu, 11 values of P(intervention)
    tis_mean, wiq_mean = np.zeros((1,11)), np.zeros((1,11))
    tis_mean_stdev, wiq_mean_stdev = np.zeros((1,11)), np.zeros((1,11))

    # Exact Analysis
    tis_PK_mean, wiq_PK_mean, niq_PK_mean, nis_PK_mean = np.zeros((1,11)), np.zeros((1,11)), np.zeros((1,11)), np.zeros((1,11))

    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            print("Parameters: mu_prime={}, p_intervention={}".format(mu_prime, p_interv))
            q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0], probs = [1.0],
                         mus = [mu], prob_speedup=[p_interv], mus_speedup=[mu_prime],
                         servers = 1)
            q_.simulate_q(customers=nCustomer, runs=nReplications)
            # q_.generate_data(sla_=0.9, quant_flag=True, write_file=True)

            # ------------------------------------------- SIMULATION -------------------------------------------
            # Time in System
            tis_mean[i, j], tis_mean_stdev[i, j], _, _ = los_wiq(q_, nReplications, los=True)
            # Wait Time in Queue
            wiq_mean[i, j], wiq_mean_stdev[i, j], _, _ = los_wiq(q_, nReplications, los=False)

            # ------------------------------------------- EXACT ANALYSIS -------------------------------------------
            wiq_PK_mean[i, j], tis_PK_mean[i, j], niq_PK_mean[i, j], nis_PK_mean[i, j] = compute_PK_means(lam, mu,
                                                                                                          mu_prime,
                                                                                                          p_interv)

    parametric_tis_mean = np.array([10.14, 8.66, 5, 4.02, 2.91, 2.51, 2.11, 1.7, 1.39, 1.21, 1.01])
    nonparametric_tis_mean = np.array([12.31, np.nan, 5, np.nan, 3.12, np.nan, np.nan, np.nan, 1.76, 1.48, 1.28])

    parametric_wiq_mean = np.array([9.23, 7.79, 4.18, 3.24, 2.17, 1.82, 1.45, 1.09, 0.82, 0.67, 0.5])
    nonparametric_wiq_mean = np.array([11.38, np.nan, 4.16, np.nan, 2.36, np.nan, np.nan, np.nan, 1.16, 0.92, 0.76])

    print("Simulation: tis_mean", tis_mean)
    print("Exact Analysis: tis_PK_mean", tis_PK_mean)
    print("Simulation: wiq_mean", wiq_mean)
    print("Exact Analysis: wiq_PK_mean", wiq_PK_mean)
    # Plotting Graphs (All Classes)
    plot_mean_4_methods(sim=tis_mean, exact=tis_PK_mean, parametric=parametric_tis_mean, nonparametric=nonparametric_tis_mean, plot_name="Length of Stay (mu_prime = 2)")
    plot_mean_4_methods(sim=wiq_mean, exact=wiq_PK_mean, parametric=parametric_wiq_mean, nonparametric=nonparametric_wiq_mean, plot_name="Waiting Time (mu_prime = 2)")

# ------------------------------------------- SIMULATION -------------------------------------------
def los_wiq(queueing_system, nReplications, los=False):

    if los:
        tracker = queueing_system.get_los_tracker()
    else:
        tracker = queueing_system.get_wait_time_tracker()

    avg_time_by_class, avg_time_all_classes = [], []

    for rep in tracker:
        # Expected Values
        avg_time_classes, time_percentile90_classes = [], []
        time_list_all_classes, time_percentile90_list_all_classes = [], []
        for time_list in rep:
            avg_time_classes.append(np.mean(time_list))
            time_list_all_classes = np.concatenate([time_list_all_classes, time_list])

        avg_time_by_class.append(avg_time_classes)  # Individual Classes
        avg_time_all_classes.append(np.mean(time_list_all_classes))  # All Classes


    # Individual Classes - 1) expected value with CI, 2) 90th percentile with CI
    time_mean_by_class, time_mean_by_class_stdev = compute_mean_and_stdev(avg_time_by_class, nReplications, all_classes=False)

    # All Classes - 1) expected value with CI, 2) 90th percentile with CI
    time_mean_all, time_mean_all_stdev = compute_mean_and_stdev(avg_time_all_classes, nReplications, all_classes=True)

    return time_mean_all, time_mean_all_stdev, time_mean_by_class, time_mean_by_class_stdev


def compute_mean_and_stdev(data, n, all_classes=False):
    if all_classes:
        data_mean = np.mean(data)
        data_variance = ((np.std(data))**2)*(n/(n - 1))
        data_stdev = np.sqrt(data_variance)
    else:
        data_mean = np.mean(data, axis=0)
        data_variance = ((np.std(data, axis=0))**2)*(n/(n - 1))
        data_stdev = np.sqrt(data_variance)
    return data_mean, data_stdev

# ------------------------------------------- EXACT ANALYSIS -------------------------------------------
def compute_PK_means(lam, mu, mu_prime, p_intervention):
    expected_S = (1 - p_intervention)/mu + p_intervention/mu_prime  # first moment
    expected_S_sq = (2/(mu*mu)) * (1 - p_intervention) + (2/(mu_prime*mu_prime)) * p_intervention  # second moment
    rho = lam * expected_S
    expected_W = (lam * expected_S_sq) / (2 * (1 - rho))
    expected_T = expected_S + expected_W
    expected_Nq = lam * expected_W
    expected_N = lam * expected_T
    return expected_W, expected_T, expected_Nq, expected_N


# ------------------------------------------- GENERATING PLOT -------------------------------------------
def plot_mean_4_methods(sim, exact, parametric, nonparametric, plot_name):
    x = np.linspace(0, 1, 11)
    plt.figure()

    # Plotting Expected Values
    plt.plot(x, sim[0], color='b', label='FCFS Simulation \n(10,000 Customers\n 1 Customer Type \n 30 Replications)', linestyle='-', marker='o')
    plt.plot(x, exact[0], color='g', label='Exact Analysis', linestyle='-', marker='o')
    plt.plot(x, parametric, color='r', label='Parametric', linestyle='-', marker='o')
    plt.plot(x, nonparametric, color='c', label='Nonparametric', linestyle='-', marker='o')

    plt.xlabel('P(speedup)')
    plt.ylabel('Expected Value')
    plt.title(plot_name)
    plt.legend()
    save_path = os.path.join(os.getcwd(), plot_name + '.png')
    plt.savefig(save_path, dpi=600)


if __name__ == "__main__":
    simulation(nCustomer=10000, nReplications=30)