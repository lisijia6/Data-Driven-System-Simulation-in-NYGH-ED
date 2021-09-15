from q_intervention import *
import matplotlib.pyplot as plt
from q_priority import *
from mpmath import *


def simulation_FCFS_single_class(nCustomer, nReplications, confidence_level, generate_data=False):
    """
    Four types of performance measures: Wait Time in Queue, Time in System, Number in Queue, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2, 2.5, 3])
    p_intervention_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Calculate the value of rho, run the simulation only if rho < 1
    # rho = lam / mu
    # if rho >= 1:
    #     return

    # Keep track of statistics, 3 values of mu, 11 values of P(intervention)
    # ------------------------------------------- SIMULATION -------------------------------------------
    # All classes
    tis_mean, wiq_mean, niq_mean, nis_mean = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_ci, wiq_ci, niq_ci, nis_ci = np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f')
    tis_90per, wiq_90per, niq_90per, nis_90per = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_90per_ci, wiq_90per_ci, niq_90per_ci, nis_90per_ci = np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f')

    # ------------------------------------------- EXACT ANALYSIS -------------------------------------------
    tis_PK_mean, wiq_PK_mean, niq_PK_mean, nis_PK_mean = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    wiq_PK_90per, tis_PK_90per, niq_PK_90per, nis_PK_90per = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))

    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            print("Parameters: mu_prime={}, p_intervention={}".format(mu_prime, p_interv))
            q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0], probs = [1.0],
                         mus = [mu], prob_speedup=[p_interv], mus_speedup=[mu_prime],
                         servers = 1)
            q_.simulate_q(customers=nCustomer, runs=nReplications)
            if generate_data:
                q_.generate_data()

            # ------------------------------------------- SIMULATION (Single Class FCFS) -------------------------------------------
            # Time in System
            los_mean_all, los_mean_all_ci, los_mean_percentile90_all, los_mean_percentile90_all_ci, los_means, los_means_ci, los_mean_percentile90s, los_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=True)
            tis_mean[i, j], tis_ci[i, j] = los_mean_all, los_mean_all_ci
            tis_90per[i, j], tis_90per_ci[i, j] = los_mean_percentile90_all, los_mean_percentile90_all_ci

            # Wait Time in Queue
            wiq_mean_all, wiq_mean_all_ci, wiq_mean_percentile90_all, wiq_mean_percentile90_all_ci, wiq_means, wiq_means_ci, wiq_mean_percentile90s, wiq_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=False)
            wiq_mean[i, j], wiq_ci[i, j] = wiq_mean_all, wiq_mean_all_ci
            wiq_90per[i, j], wiq_90per_ci[i, j] = wiq_mean_percentile90_all, wiq_mean_percentile90_all_ci

            # Number in Queue
            niq_num_all_classes, niq_num_all_classes_ci, niq_mean_percentile90_all, niq_mean_percentile90_all_ci, niq_nums, niq_nums_ci, niq_mean_percentile90s, niq_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=True)
            niq_mean[i, j], niq_ci[i, j] = niq_num_all_classes, niq_num_all_classes_ci
            niq_90per[i, j], niq_90per_ci[i, j] = niq_mean_percentile90_all, niq_mean_percentile90_all_ci

            # Number in System
            nis_num_all_classes, nis_num_all_classes_ci, nis_mean_percentile90_all, nis_mean_percentile90_all_ci, nis_nums, nis_nums_ci, nis_mean_percentile90s, nis_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=False)
            nis_mean[i, j], nis_ci[i, j] = nis_num_all_classes, nis_num_all_classes_ci
            nis_90per[i, j], nis_90per_ci[i, j] = nis_mean_percentile90_all, nis_mean_percentile90_all_ci

            # ------------------------------------------- EXACT ANALYSIS (Single Class FCFS) -------------------------------------------
            wiq_PK_mean[i, j], tis_PK_mean[i, j], niq_PK_mean[i, j], nis_PK_mean[i, j] = compute_PK_means(lam, mu,
                                                                                                          mu_prime,
                                                                                                          p_interv)
            wiq_PK_90per[i, j], tis_PK_90per[i, j], niq_PK_90per[i, j], nis_PK_90per[i, j] = compute_PK_90percentiles(lam, mu,
                                                                                                          mu_prime,
                                                                                                          p_interv)

    # ------------------------------------------- PRINTING RESULTS -------------------------------------------
    print("Single Class FCFS Numerical Results")
    print("\nSIMULATION:")
    print("tis_mean", tis_mean, "\ntis_90per", tis_90per, "\nwiq_mean", wiq_mean, "\nwiq_90per", wiq_90per)
    print("tis_ci", tis_ci, "\ntis_90per_ci", tis_90per_ci, "\nwiq_ci", wiq_ci, "\nwiq_90per_ci", wiq_90per_ci)
    print("niq_mean", niq_mean, "\nniq_90per", niq_90per, "\nnis_mean", nis_mean, "\nnis_90per", nis_90per)
    print("niq_ci", niq_ci, "\nniq_90per_ci", niq_90per_ci, "\nnis_ci", nis_ci, "\nnis_90per_ci", nis_90per_ci)
    print("\nEXACT ANALYSIS:")
    print("tis_PK_mean", tis_PK_mean, "\ntis_PK_90per", tis_PK_90per, "\nwiq_PK_mean", wiq_PK_mean, "\nwiq_PK_90per", wiq_PK_90per)
    print("niq_PK_mean", niq_PK_mean, "\nniq_PK_90per", niq_PK_90per, "\nnis_PK_mean", nis_PK_mean, "\nnis_PK_90per", nis_PK_90per)

    # ------------------------------------------- PLOTTING (Single Class FCFS) -------------------------------------------
    # Plotting Graphs (All Classes)
    plot_mean_90percentile_with_CI(mu_prime_list, tis_mean, tis_ci, tis_90per, tis_90per_ci,
                                   "Single Class FCFS - Time in System With {}% Confidence Interval, {} Customers, {} Replications".format(
                                       confidence_level*100, nCustomer, nReplications), tis_PK_mean, tis_PK_90per)
    plot_mean_90percentile_with_CI(mu_prime_list, wiq_mean, wiq_ci, wiq_90per, wiq_90per_ci,
                                   "Single Class FCFS - Wait Time in Queue With {}% Confidence Interval, {} Customers, {} Replications".format(
                                       confidence_level*100, nCustomer, nReplications), wiq_PK_mean, wiq_PK_90per)
    plot_mean_90percentile_with_CI(mu_prime_list, niq_mean, niq_ci, niq_90per, niq_90per_ci,
                                   "Single Class FCFS - Number in Queue With {}% Confidence Interval, {} Customers, {} Replications".format(
                                       confidence_level*100, nCustomer, nReplications), niq_PK_mean, niq_PK_90per)
    plot_mean_90percentile_with_CI(mu_prime_list, nis_mean, nis_ci, nis_90per, nis_90per_ci,
                                   "Single Class FCFS - Number in System With {}% Confidence Interval, {} Customers, {} Replications".format(
                                       confidence_level*100, nCustomer, nReplications), nis_PK_mean, nis_PK_90per)


def simulation_priority_two_classes(nCustomer, nReplications, confidence_level, nClasses=2, generate_data=False):
    """
    Four types of performance measures: Wait Time in Queue, Time in System, Number in Queue, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2, 2.5, 3])
    p_intervention_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Keep track of statistics, 3 values of mu, 11 values of P(intervention)
    # ------------------------------------------- SIMULATION -------------------------------------------
    # All classes
    tis_mean, wiq_mean, niq_mean, nis_mean = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_ci, wiq_ci, niq_ci, nis_ci = np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f')
    tis_90per, wiq_90per, niq_90per, nis_90per = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_90per_ci, wiq_90per_ci, niq_90per_ci, nis_90per_ci = np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f')

    # Individual Classes
    nRow = len(mu_prime_list) * nClasses
    tis_mean_by_class, wiq_mean_by_class, niq_mean_by_class, nis_mean_by_class = np.zeros(
        (nRow, 11)), np.zeros((nRow, 11)), np.zeros(
        (nRow, 11)), np.zeros((nRow, 11))
    tis_mean_ci_by_class, wiq_mean_ci_by_class, niq_mean_ci_by_class, nis_mean_ci_by_class = np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f'), np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f')
    tis_90per_by_class, wiq_90per_by_class, niq_90per_by_class, nis_90per_by_class = np.zeros(
        (nRow, 11)), np.zeros((nRow, 11)), np.zeros(
        (nRow, 11)), np.zeros((nRow, 11))
    tis_90per_ci_by_class, wiq_90per_ci_by_class, niq_90per_ci_by_class, nis_90per_ci_by_class = np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f')

    # ------------------------------------------- EXACT ANALYSIS -------------------------------------------
    # All Classes
    wiq_priority_exact_mean, tis_priority_exact_mean, niq_priority_exact_mean, nis_priority_exact_mean = np.zeros(
        (3, 11)), np.zeros((3, 11)), np.zeros((3, 11)), np.zeros((3, 11))
    # Individual Classes
    tis_priority_exact_mean_by_class, wiq_priority_exact_mean_by_class, niq_priority_exact_mean_by_class, nis_priority_exact_mean_by_class = np.zeros(
        (nRow, 11)), np.zeros((nRow, 11)), np.zeros((nRow, 11)), np.zeros((nRow, 11))

    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            print("Parameters: mu_prime={}, p_intervention={}".format(mu_prime, p_interv))
            q_ = multi_class_single_station_priority(lambda_=1, classes=[0, 1], probs=[0.5, 0.5], mus=[mu, mu],
                                                     prob_speedup=[p_interv, p_interv], mus_speedup=[mu_prime, mu_prime],
                                                     servers=1, priority=[0, 1])
            q_.simulate_priority_q(customers=nCustomer, runs=nReplications)
            if generate_data:
                q_.generate_data()

            # ------------------------------------------- SIMULATION (All Classes - Non-preemptive Priority) -------------------------------------------
            # Time in System
            los_mean_all, los_mean_all_ci, los_mean_percentile90_all, los_mean_percentile90_all_ci, los_means, los_means_ci, los_mean_percentile90s, los_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=True)
            tis_mean[i, j], tis_ci[i, j] = los_mean_all, los_mean_all_ci
            tis_90per[i, j], tis_90per_ci[i, j] = los_mean_percentile90_all, los_mean_percentile90_all_ci

            # Wait Time in Queue
            wiq_mean_all, wiq_mean_all_ci, wiq_mean_percentile90_all, wiq_mean_percentile90_all_ci, wiq_means, wiq_means_ci, wiq_mean_percentile90s, wiq_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=False)
            wiq_mean[i, j], wiq_ci[i, j] = wiq_mean_all, wiq_mean_all_ci
            wiq_90per[i, j], wiq_90per_ci[i, j] = wiq_mean_percentile90_all, wiq_mean_percentile90_all_ci

            # Number in Queue
            niq_num_all_classes, niq_num_all_classes_ci, niq_mean_percentile90_all, niq_mean_percentile90_all_ci, niq_nums, niq_nums_ci, niq_mean_percentile90s, niq_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=True)
            niq_mean[i, j], niq_ci[i, j] = niq_num_all_classes, niq_num_all_classes_ci
            niq_90per[i, j], niq_90per_ci[i, j] = niq_mean_percentile90_all, niq_mean_percentile90_all_ci

            # Number in System
            nis_num_all_classes, nis_num_all_classes_ci, nis_mean_percentile90_all, nis_mean_percentile90_all_ci, nis_nums, nis_nums_ci, nis_mean_percentile90s, nis_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=False)
            nis_mean[i, j], nis_ci[i, j] = nis_num_all_classes, nis_num_all_classes_ci
            nis_90per[i, j], nis_90per_ci[i, j] = nis_mean_percentile90_all, nis_mean_percentile90_all_ci

            # ------------------------------------------- SIMULATION (Individual Classes - Non-preemptive Priority) -------------------------------------------
            offset = 0
            for k in range(len(los_means)):
                # Time in System
                tis_mean_by_class[i + offset, j], tis_mean_ci_by_class[i + offset, j]  = los_means[k], (los_means_ci[0][k], los_means_ci[1][k])
                tis_90per_by_class[i + offset, j], tis_90per_ci_by_class[i + offset, j]  = los_mean_percentile90s[k], (los_mean_percentile90s_ci[0][k], los_mean_percentile90s_ci[1][k])
                # Wait Time in Queue
                wiq_mean_by_class[i + offset, j], wiq_mean_ci_by_class[i + offset, j] = wiq_means[k], (wiq_means_ci[0][k], wiq_means_ci[1][k])
                wiq_90per_by_class[i + offset, j], wiq_90per_ci_by_class[i + offset, j]  = wiq_mean_percentile90s[k], (wiq_mean_percentile90s_ci[0][k], wiq_mean_percentile90s_ci[1][k])
                # Number in Queue
                niq_mean_by_class[i + offset, j], niq_mean_ci_by_class[i + offset, j]  = niq_nums[k], (niq_nums_ci[0][k], niq_nums_ci[1][k])
                niq_90per_by_class[i + offset, j], niq_90per_ci_by_class[i + offset, j] = niq_mean_percentile90s[k], (niq_mean_percentile90s_ci[0][k], niq_mean_percentile90s_ci[1][k])
                # Number in System
                nis_mean_by_class[i + offset, j], nis_mean_ci_by_class[i + offset, j] = nis_nums[k], (nis_nums_ci[0][k], nis_nums_ci[1][k])
                nis_90per_by_class[i + offset, j], nis_90per_ci_by_class[i + offset, j] = nis_mean_percentile90s[k], (nis_mean_percentile90s_ci[0][k], nis_mean_percentile90s_ci[1][k])
                offset += len(mu_prime_list)

            # ------------------------------------------- EXACT ANALYSIS (All Classes - Non-preemptive Priority) -------------------------------------------
            # assume arrival rate (lambda), service rate (mu), speedup service rate (mu'), and P(speedup) are the same for the 2 classes
            wiq_exact_mean_all, tis_exact_mean_all, niq_exact_mean_all, nis_exact_mean_all, \
            wiq_exact_mean_by_class, tis_exact_mean_by_class, niq_exact_mean_by_class, nis_exact_mean_by_class = compute_priority_mean(
                0.5*lam, 0.5*lam, mu, mu, mu_prime, mu_prime, p_interv, p_interv, non_preemptive=True)
            wiq_priority_exact_mean[i, j], tis_priority_exact_mean[i, j] = wiq_exact_mean_all, tis_exact_mean_all
            niq_priority_exact_mean[i, j], nis_priority_exact_mean[i, j] = niq_exact_mean_all, nis_exact_mean_all

            # ------------------------------------------- EXACT ANALYSIS (Individual Classes - Non-preemptive Priority) -------------------------------------------
            offset = 0
            for k in range(len(wiq_exact_mean_by_class)):
                wiq_priority_exact_mean_by_class[i + offset, j] = wiq_exact_mean_by_class[k]
                tis_priority_exact_mean_by_class[i + offset, j] = tis_exact_mean_by_class[k]
                niq_priority_exact_mean_by_class[i + offset, j] = niq_exact_mean_by_class[k]
                nis_priority_exact_mean_by_class[i + offset, j] = nis_exact_mean_by_class[k]
                offset += len(mu_prime_list)

    # ------------------------------------------- PRINTING RESULTS -------------------------------------------
    print("Non-preemptive Priority (Two Classes) Numerical Results")
    print("\nSIMULATION (ALL CLASSES):")
    print("tis_mean", tis_mean, "\ntis_90per", tis_90per, "\nwiq_mean", wiq_mean, "\nwiq_90per", wiq_90per)
    print("niq_mean", niq_mean, "\nniq_90per", niq_90per, "\nnis_mean", nis_mean, "\nnis_90per", nis_90per)
    print("\nEXACT ANALYSIS (ALL CLASSES):")
    print("tis_priority_exact_mean", tis_priority_exact_mean, "\nwiq_priority_exact_mean", wiq_priority_exact_mean)
    print("niq_priority_exact_mean", niq_priority_exact_mean, "\nnis_priority_exact_mean", nis_priority_exact_mean)
    print("\nSIMULATION (INDIVIDUAL CLASSES):")
    print("tis_mean_by_class", tis_mean_by_class, "\ntis_90per_by_class", tis_90per_by_class)
    print("wiq_mean_by_class", wiq_mean_by_class, "\nwiq_90per_by_class", wiq_90per_by_class)
    print("niq_mean_by_class", niq_mean_by_class, "\nniq_90per_by_class", niq_90per_by_class)
    print("nis_mean_by_class", nis_mean_by_class, "\nnis_90per_by_class", nis_90per_by_class)
    print("\nEXACT ANALYSIS (INDIVIDUAL CLASSES):")
    print("tis_priority_exact_mean_by_class", tis_priority_exact_mean_by_class, "\nwiq_priority_exact_mean_by_class", wiq_priority_exact_mean_by_class)
    print("niq_priority_exact_mean_by_class", niq_priority_exact_mean_by_class, "\nnis_priority_exact_mean_by_class", nis_priority_exact_mean_by_class)


    # ------------------------------------------- PLOTTING -------------------------------------------
    # Plotting Graphs (All Classes)
    plot_mean_90percentile_with_CI(mu_prime_list, tis_mean, tis_ci, tis_90per, tis_90per_ci,
                                   "Non-preemptive Priority (Two Classes) - Time in System With {}% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       confidence_level*100, nCustomer, nReplications, nClasses), tis_priority_exact_mean)
    plot_mean_90percentile_with_CI(mu_prime_list, wiq_mean, wiq_ci, wiq_90per, wiq_90per_ci,
                                   "Non-preemptive Priority (Two Classes) - Wait Time in Queue With {}% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       confidence_level*100, nCustomer, nReplications, nClasses), wiq_priority_exact_mean)
    plot_mean_90percentile_with_CI(mu_prime_list, niq_mean, niq_ci, niq_90per, niq_90per_ci,
                                   "Non-preemptive Priority (Two Classes) - Number in Queue With {}% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       confidence_level*100, nCustomer, nReplications, nClasses), niq_priority_exact_mean)
    plot_mean_90percentile_with_CI(mu_prime_list, nis_mean, nis_ci, nis_90per, nis_90per_ci,
                                   "Non-preemptive Priority (Two Classes) - Number in System With {}% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       confidence_level*100, nCustomer, nReplications, nClasses), nis_priority_exact_mean)

    # Plotting Graphs (Individual Classes For Multi-Class Systems)
    start, end = 0, len(mu_prime_list)
    for g in range(nClasses):
        plot_mean_90percentile_with_CI(mu_prime_list, tis_mean_by_class[start:end], tis_mean_ci_by_class[start:end],
                                       tis_90per_by_class[start:end], tis_90per_ci_by_class[start:end],
                                       "Non-preemptive Priority (Two Classes) - Time in System With {}% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           confidence_level*100, nCustomer, nReplications, g+1), tis_priority_exact_mean_by_class[start:end])
        plot_mean_90percentile_with_CI(mu_prime_list, wiq_mean_by_class[start:end], wiq_mean_ci_by_class[start:end],
                                       wiq_90per_by_class[start:end], wiq_90per_ci_by_class[start:end],
                                       "Non-preemptive Priority (Two Classes) - Wait Time in Queue With {}% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           confidence_level*100, nCustomer, nReplications, g+1), wiq_priority_exact_mean_by_class[start:end])
        plot_mean_90percentile_with_CI(mu_prime_list, niq_mean_by_class[start:end], niq_mean_ci_by_class[start:end],
                                       niq_90per_by_class[start:end], niq_90per_ci_by_class[start:end],
                                       "Non-preemptive Priority (Two Classes) - Number in Queue With {}% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           confidence_level*100, nCustomer, nReplications, g+1), niq_priority_exact_mean_by_class[start:end])
        plot_mean_90percentile_with_CI(mu_prime_list, nis_mean_by_class[start:end], nis_mean_ci_by_class[start:end],
                                       nis_90per_by_class[start:end], nis_90per_ci_by_class[start:end],
                                       "Non-preemptive Priority (Two Classes) - Number in System With {}% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           confidence_level*100, nCustomer, nReplications, g+1), nis_priority_exact_mean_by_class[start:end])
        start += len(mu_prime_list)
        end += len(mu_prime_list)


def simulation_MGnInfinity_single_class(nCustomer, nReplications, confidence_level, generate_data=False):
    """
    Two types of performance measures: Time in System, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2, 2.5, 3])
    p_intervention_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Keep track of statistics, 3 values of mu, 11 values of P(intervention)
    # ------------------------------------------- SIMULATION -------------------------------------------
    # All classes
    tis_mean, nis_mean = np.zeros((3,11)), np.zeros((3,11))
    tis_ci, nis_ci = np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f')
    tis_90per, nis_90per = np.zeros((3,11)), np.zeros((3,11))
    tis_90per_ci, nis_90per_ci = np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f')


    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            print("Parameters: mu_prime={}, p_intervention={}".format(mu_prime, p_interv))
            q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0], probs = [1.0],
                         mus = [mu], prob_speedup=[p_interv], mus_speedup=[mu_prime],
                         servers = 1)
            q_.simulate_q(customers=nCustomer, runs=nReplications)
            if generate_data:
                q_.generate_data()

            # ------------------------------------------- SIMULATION (Single Class MGnInfinity) -------------------------------------------
            # Time in System
            los_mean_all, los_mean_all_ci, los_mean_percentile90_all, los_mean_percentile90_all_ci, los_means, los_means_ci, los_mean_percentile90s, los_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=True)
            tis_mean[i, j], tis_ci[i, j] = los_mean_all, los_mean_all_ci
            tis_90per[i, j], tis_90per_ci[i, j] = los_mean_percentile90_all, los_mean_percentile90_all_ci

            # Number in System
            nis_num_all_classes, nis_num_all_classes_ci, nis_mean_percentile90_all, nis_mean_percentile90_all_ci, nis_nums, nis_nums_ci, nis_mean_percentile90s, nis_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=False)
            nis_mean[i, j], nis_ci[i, j] = nis_num_all_classes, nis_num_all_classes_ci
            nis_90per[i, j], nis_90per_ci[i, j] = nis_mean_percentile90_all, nis_mean_percentile90_all_ci


    # ------------------------------------------- PRINTING RESULTS -------------------------------------------
    print("Single Class FCFS Numerical Results")
    print("\nSIMULATION:")
    print("tis_mean", tis_mean, "\ntis_90per", tis_90per, "\nnis_mean", nis_mean, "\nnis_90per", nis_90per)
    print("tis_ci", tis_ci, "\ntis_90per_ci", tis_90per_ci, "\nnis_ci", nis_ci, "\nnis_90per_ci", nis_90per_ci)

    # ------------------------------------------- PLOTTING (Single Class MGnInfinity) -------------------------------------------
    # Plotting Graphs (All Classes)
    plot_mean_90percentile_with_CI(mu_prime_list, tis_mean, tis_ci, tis_90per, tis_90per_ci,
                                   "Single Class MG(n)Infinity - Time in System With {}% Confidence Interval, {} Customers, {} Replications".format(
                                       confidence_level*100, nCustomer, nReplications), None, None)
    plot_mean_90percentile_with_CI(mu_prime_list, nis_mean, nis_ci, nis_90per, nis_90per_ci,
                                   "Single Class MG(n)Infinity - Number in System With {}% Confidence Interval, {} Customers, {} Replications".format(
                                       confidence_level*100, nCustomer, nReplications), None, None)


# ------------------------------------------- SIMULATION -------------------------------------------
def los_wiq(queueing_system, confidence_level, nReplications, los=False):

    if los:
        tracker = queueing_system.get_los_tracker()
    else:
        tracker = queueing_system.get_wait_time_tracker()

    avg_time_by_class, avg_time_all_classes = [], []
    time_percentile90_by_class, time_percentile90_all_classes = [], []
    for rep in tracker:
        # Expected Values
        avg_time_classes, time_percentile90_classes = [], []
        time_list_all_classes, time_percentile90_list_all_classes = [], []
        for time_list in rep:
            avg_time_classes.append(np.mean(time_list))
            time_list_all_classes = np.concatenate([time_list_all_classes, time_list])

        avg_time_by_class.append(avg_time_classes)  # Individual Classes
        avg_time_all_classes.append(np.mean(time_list_all_classes))  # All Classes

        # 90th Percentile
        for time_percentile90_list in rep:
            time_percentile90_classes.append(np.percentile(time_percentile90_list, 90))
            time_percentile90_list_all_classes = np.concatenate([time_percentile90_list_all_classes, time_percentile90_list])
        time_percentile90_by_class.append(time_percentile90_classes)  # Individual Classes
        time_percentile90_all_classes.append(np.percentile(time_percentile90_list_all_classes, 90))   # All Classes

    # Individual Classes - 1) expected value with CI, 2) 90th percentile with CI
    time_mean_by_class, time_mean_by_class_ci = compute_mean_and_CI(avg_time_by_class, confidence_level, nReplications,
                                                                  all_classes=False)
    time_mean_percentile90_by_class, time_mean_percentile90_by_class_ci = compute_mean_and_CI(time_percentile90_by_class,
                                                                                            confidence_level, nReplications,
                                                                                            all_classes=False)

    # All Classes - 1) expected value with CI, 2) 90th percentile with CI
    time_mean_all, time_mean_all_ci = compute_mean_and_CI(avg_time_all_classes, confidence_level, nReplications,
                                                                        all_classes=True)
    time_mean_percentile90_all, time_mean_percentile90_all_ci = compute_mean_and_CI(time_percentile90_all_classes,
                                                                                    confidence_level, nReplications,
                                                                                    all_classes=True)

    return time_mean_all, time_mean_all_ci, time_mean_percentile90_all, time_mean_percentile90_all_ci, \
           time_mean_by_class, time_mean_by_class_ci, time_mean_percentile90_by_class, time_mean_percentile90_by_class_ci


def niq_nis(queueing_system, confidence_level, nReplications, queue=False):
    classes = queueing_system.get_classes()
    if queue:
        tracker = queueing_system.get_queue_tracker()
    else:
        tracker = queueing_system.get_nis_tracker()
    avg_num_list_by_class, avg_num_list_all_classes = [], []
    percentile90_list_by_class, percentile90_list_all_classes = [], []
    for rep in tracker:
        t_n = rep[0][-1][0]
        area_all_classes = 0
        num_list_all_classes = []  # [n1, n2, n3, ...] N's are not unique
        num_sets_by_class = []  # [[n1, n2, n3, ...], [n1, n2, n3, ...], ...]
        num_pmf_list_by_class = []  # normalized probabilities [[p1, p2, p3, ...], [p1, p2, p3, ...], ...]

        avg_num_list_classes = []  # [rep1_c1_mean, rep1_c2_mean, ...]
        percentile90_by_class = []  # [rep1_c1_90thperc, rep1_c2_90thperc, ...]

        for c in classes:
            num_list_all_classes += [tup[1] for tup in rep[c]]  # list of N's for customers of all classes
        num_set_all_classes = list(set(num_list_all_classes))  # unique list of N's for all classes
        num_pmf_all_classes = np.zeros(len(num_set_all_classes))

        for c in classes:
            class_x_tracker = rep[c]

            area_by_class = 0

            num_list_class_x = [tup[1] for tup in class_x_tracker]  # list of N's for customers of class x
            num_set_class_x = list(set(num_list_class_x))  # unique list of N's for customers of class x
            num_sets_by_class.append(num_set_class_x)  # append to sets
            num_pmf_by_class = np.zeros(len(num_set_class_x))  # initialize pmf with probabilities of 0

            for i in range(1, len(class_x_tracker)):
                t_i1, t_i0 = class_x_tracker[i][0], class_x_tracker[i-1][0]
                Num_i0 = class_x_tracker[i-1][1]

                # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n; E(NIS) = sum_i=1_to_n_ [(t_i - t_i-1)*NIS_i] / t_n
                area_by_class += (t_i1 - t_i0) * Num_i0

                # For 90th percentile by class
                idx = num_sets_by_class[c].index(Num_i0)  # find the index for N = Num_i1
                num_pmf_by_class[idx] += t_i1 - t_i0  # accumulate time units for N = n1, n2, ... for each class
                # For 90th percentile of all classes
                idx = num_set_all_classes.index(Num_i0)
                num_pmf_all_classes[idx] += t_i1 - t_i0

            # For expected values
            avg_num_list_classes.append(area_by_class / t_n)
            area_all_classes += area_by_class

            # For 90th percentile by class
            num_pmf_class_x = num_pmf_by_class / t_n  # normalize pmf for customers in class x
            num_pmf_list_by_class.append(num_pmf_class_x)

            cum_prob, n, done = 0, 0, False
            while not done:
                cum_prob += num_pmf_class_x[n]
                if cum_prob >= 0.9:
                    percentile90_by_class.append(num_set_class_x[n])
                    done = True
                n += 1

        avg_num_list_by_class.append(avg_num_list_classes)  # [[rep1_c1_mean, rep1_c2_mean, ...], [rep2_c1_mean, rep2_c2_mean, ...]]
        avg_num_list_all_classes.append(area_all_classes / t_n)  # [rep1_all_mean, rep2_all_mean, ...]

        percentile90_list_by_class.append(percentile90_by_class)  # [[rep1_c1_90thperc, rep1_c2_90thperc, ...], [rep2_c1_90thperc, rep2_c2_90thperc, ...], ...]

        num_pmf_all_classes = num_pmf_all_classes / (t_n * len(classes))  # normalize pmf for customers in all classes
        cum_prob_all, n_all, done_all = 0, 0, False
        while not done_all:
            cum_prob_all += num_pmf_all_classes[n_all]
            if cum_prob_all >= 0.9:
                percentile90_list_all_classes.append(num_set_all_classes[n_all])  # [rep1_all_90thperc, rep2_all_90thperc, ...]
                done_all = True
            n_all += 1

    # Individual Classes - 1) expected value with CI, 2) 90th percentile with CI
    num_mean_by_class, mean_num_by_class_ci = compute_mean_and_CI(avg_num_list_by_class, confidence_level, nReplications,
                                                                        all_classes=False)
    num_mean_percentile90_by_class, num_mean_percentile90_by_class_ci = compute_mean_and_CI(percentile90_list_by_class,
                                                                                            confidence_level, nReplications,
                                                                                            all_classes=False)

    # All Classes - 1) expected value with CI, 2) 90th percentile with CI
    num_mean_all_classes, num_mean_all_classes_ci = compute_mean_and_CI(avg_num_list_all_classes, confidence_level, nReplications,
                                                                        all_classes=True)
    num_mean_percentile90_all_classes, num_mean_percentile90_all_classes_ci = compute_mean_and_CI(
        percentile90_list_all_classes, confidence_level, nReplications, all_classes=True)

    return num_mean_all_classes, num_mean_all_classes_ci, num_mean_percentile90_all_classes, \
           num_mean_percentile90_all_classes_ci, num_mean_by_class, mean_num_by_class_ci, \
           num_mean_percentile90_by_class, num_mean_percentile90_by_class_ci


def compute_mean_and_CI(data, confidence_level, n, all_classes=False):
    z_vals_dict = {0.8 : 1.28, 0.9 : 1.645, 0.95 : 1.96, 0.98 : 2.33, 0.99 : 2.58}  # confidence level : z*-value
    z_val = z_vals_dict.get(confidence_level)
    if all_classes:
        data_mean = np.mean(data)
        data_variance = ((np.std(data))**2)*(n/(n - 1))
        data_stdev = np.sqrt(data_variance)
        data_ci = (data_mean - z_val*data_stdev/np.sqrt(n), data_mean + z_val*data_stdev/np.sqrt(n))
        # data_mean_stderr = sem(data)
        # if data_mean_stderr == 0:  # all data points are equal
        #     data_mean_ci = (data_mean, data_mean)
        # else:
        #     data_mean_ci = t.interval(confidence_level, n - 1, data_mean, data_mean_stderr)
    else:
        data_mean = np.mean(data, axis=0)
        data_variance = ((np.std(data, axis=0))**2)*(n/(n - 1))
        data_stdev = np.sqrt(data_variance)
        data_ci = (data_mean - z_val*data_stdev/np.sqrt(n) , data_mean + z_val*data_stdev/np.sqrt(n))
        # data_mean_stderr = sem(data, axis=0)
        # data_mean_ci = t.interval(confidence_level, n - 1, data_mean, data_mean_stderr)
    return data_mean, data_ci

# ------------------------------------------- EXACT ANALYSIS (First Come First Serve) -------------------------------------------
def compute_PK_means(lam, mu, mu_prime, p_intervention):
    expected_S = (1 - p_intervention)/mu + p_intervention/mu_prime  # first moment
    expected_S_sq = (2/(mu*mu)) * (1 - p_intervention) + (2/(mu_prime*mu_prime)) * p_intervention  # second moment
    rho = lam * expected_S
    expected_W = (lam * expected_S_sq) / (2 * (1 - rho))
    expected_T = expected_S + expected_W
    expected_Nq = lam * expected_W
    expected_N = lam * expected_T
    return expected_W, expected_T, expected_Nq, expected_N


def compute_PK_90percentiles(lam, mu1, mu_prime, p_intervention):
    sojournDone, waitingDone = False, False
    time = 1e-08
    percentile90_T, percentile90_W = 0, 0
    time_increment = 0.25
    rho = (mu1*p_intervention + mu_prime*(1-p_intervention)) / (mu1*mu_prime) * lam
    # Laplace transform formula for service time
    S_star = lambda w: (1-p_intervention)*(mu1/(w+mu1)) + p_intervention*mu_prime/(w+mu_prime)
    # PK transform formula for the sojourn time
    T_star = lambda w: (1-rho)*w*S_star(w) / (w-lam+lam*S_star(w))
    # PK transform formula for the waiting time
    W_star = lambda w: (1-rho)*w / (w-lam+lam*S_star(w))
    while not (sojournDone and waitingDone):
        if not sojournDone:
            sojourn = float(invertlaplace(lambda w: T_star(w) / w, time, method='talbot'))
            if sojourn >= 0.9:
                percentile90_T = time
                sojournDone = True
        if not waitingDone:
            waiting = float(invertlaplace(lambda w: W_star(w) / w, time, method='talbot'))
            if waiting >= 0.9:
                percentile90_W = time
                waitingDone = True
        if not (sojourn >= 0.9 and waiting >= 0.9):
            time += time_increment
    percentile90_Nq = lam * percentile90_W
    percentile90_N = lam * percentile90_T
    return percentile90_W, percentile90_T, percentile90_Nq, percentile90_N


# ------------------------------------------- EXACT ANALYSIS (Non-preemptive Priority) -------------------------------------------
def compute_priority_mean(lam1, lam2, mu1, mu2, mu_prime1, mu_prime2, p_intervention1, p_intervention2, non_preemptive=True):
    expected_S_1 = (1 - p_intervention1) / mu1 + p_intervention1 / mu_prime1  # first moment
    expected_S_2 = (1 - p_intervention2) / mu2 + p_intervention2 / mu_prime2  # first moment
    expected_S_sq_1 = (2/(mu1*mu1)) * (1 - p_intervention1) + (2/(mu_prime1*mu_prime1)) * p_intervention1  # second moment
    expected_S_sq_2 = (2/(mu2*mu2)) * (1 - p_intervention2) + (2/(mu_prime2*mu_prime2)) * p_intervention2  # second moment

    r_bar = 0.5 * (lam1 * expected_S_sq_1 + lam2 * expected_S_sq_2)  # mean residual service time
    rho1, rho2 = lam1 * expected_S_1, lam2 * expected_S_2
    print(r_bar, rho1, rho2)

    wiq_1, wiq_2 = r_bar / (1 - rho1), r_bar / ((1 - rho1) * (1 - rho1 - rho2))
    tis_1, tis_2 = wiq_1 + expected_S_1, wiq_2 + expected_S_2
    wiq_exact_mean_by_class, tis_exact_mean_by_class = [wiq_1, wiq_2], [tis_1, tis_2]
    niq_exact_mean_by_class, nis_exact_mean_by_class = [wiq_1*lam1, wiq_2*lam2], [tis_1*lam1, tis_2*lam2]

    wiq_exact_mean_all, tis_exact_mean_all = (lam1*wiq_1 + lam2*wiq_2) / (lam1 + lam2), (lam1*tis_1 + lam2*tis_2) / (lam1 + lam2)
    niq_exact_mean_all, nis_exact_mean_all = lam1*wiq_1 + lam2*wiq_2, lam1*tis_1 + lam2*tis_2

    return wiq_exact_mean_all, tis_exact_mean_all, niq_exact_mean_all, nis_exact_mean_all, \
            wiq_exact_mean_by_class, tis_exact_mean_by_class, niq_exact_mean_by_class, nis_exact_mean_by_class


# ------------------------------------------- GENERATING PLOT -------------------------------------------
def plot_mean_90percentile_with_CI(mu_prime_list, mean, mean_ci, percentile90, percentile90_ci, plot_type, exact_mean=None, exact_percentile=None):
    x = np.linspace(0, 1, 11)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(plot_type)
    fig.set_size_inches(12, 6)

    # Plotting Expected Values
    y1_sim, y2_sim, y3_sim = mean[0], mean[1], mean[2]
    ci1_lower, ci1_upper = [mean_ci[0][i][0] for i in range(len(mean_ci[0]))], [mean_ci[0][i][1] for i in
                                                                                range(len(mean_ci[0]))]
    ci2_lower, ci2_upper = [mean_ci[1][i][0] for i in range(len(mean_ci[1]))], [mean_ci[1][i][1] for i in
                                                                                range(len(mean_ci[0]))]
    ci3_lower, ci3_upper = [mean_ci[2][i][0] for i in range(len(mean_ci[2]))], [mean_ci[2][i][1] for i in
                                                                                range(len(mean_ci[0]))]
    # Simulation
    ax1.plot(x, y1_sim, color='b', label='Simulation: mu_prime={}'.format(mu_prime_list[0]), linestyle='-')
    ax1.fill_between(x, ci1_lower, ci1_upper, color='b', alpha=0.1)
    ax1.plot(x, y2_sim, color='g', label='Simulation: mu_prime={}'.format(mu_prime_list[1]), linestyle='-')
    ax1.fill_between(x, ci2_lower, ci2_upper, color='g', alpha=0.1)
    ax1.plot(x, y3_sim, color='r', label='Simulation: mu_prime={}'.format(mu_prime_list[2]), linestyle='-')
    ax1.fill_between(x, ci3_lower, ci3_upper, color='r', alpha=0.1)

    if exact_mean is not None:
        # Exact Analysis
        y1_exact, y2_exact, y3_exact = exact_mean[0], exact_mean[1], exact_mean[2]
        ax1.plot(x, y1_exact, color='b', label='Exact Analysis: mu_prime={}'.format(mu_prime_list[0]), linestyle='--')
        ax1.plot(x, y2_exact, color='g', label='Exact Analysis: mu_prime={}'.format(mu_prime_list[1]), linestyle='--')
        ax1.plot(x, y3_exact, color='r', label='Exact Analysis: mu_prime={}'.format(mu_prime_list[2]), linestyle='--')

    ax1.set(xlabel = 'P(speedup)', ylabel = 'Expected Value')
    # ax1.legend(('mu_prime=2', 'mu_prime=2.5', 'mu_prime=3'))
    ax1.legend()

    # Plotting 90th Percentiles
    y90p1_sim, y90p2_sim, y90p3_sim = percentile90[0], percentile90[1], percentile90[2]
    ci90p1_lower, ci90p1_upper = [percentile90_ci[0][i][0] for i in range(len(percentile90_ci[0]))], [
        percentile90_ci[0][i][1] for i in range(len(percentile90_ci[0]))]
    ci90p2_lower, ci90p2_upper = [percentile90_ci[1][i][0] for i in range(len(percentile90_ci[1]))], [
        percentile90_ci[1][i][1] for i in range(len(percentile90_ci[0]))]
    ci90p3_lower, ci90p3_upper = [percentile90_ci[2][i][0] for i in range(len(percentile90_ci[2]))], [
        percentile90_ci[2][i][1] for i in range(len(percentile90_ci[0]))]

    ax2.plot(x, y90p1_sim, color='b', label='Simulation: mu_prime={}'.format(mu_prime_list[0]), linestyle='-')
    ax2.fill_between(x, ci90p1_lower, ci90p1_upper, color='b', alpha=0.1)
    ax2.plot(x, y90p2_sim, color='g', label='Simulation: mu_prime={}'.format(mu_prime_list[1]), linestyle='-')
    ax2.fill_between(x, ci90p2_lower, ci90p2_upper, color='g', alpha=0.1)
    ax2.plot(x, y90p3_sim, color='r', label='Simulation: mu_prime={}'.format(mu_prime_list[2]), linestyle='-')
    ax2.fill_between(x, ci90p3_lower, ci90p3_upper, color='r', alpha=0.1)

    if exact_percentile is not None:
        # Exact Analysis
        y90p1_exact, y90p2_exact, y90p3_exact = exact_percentile[0], exact_percentile[1], exact_percentile[2]
        ax2.plot(x, y90p1_exact, color='b', label='Exact Analysis: mu_prime={}'.format(mu_prime_list[0]), linestyle='--')
        ax2.plot(x, y90p2_exact, color='g', label='Exact Analysis: mu_prime={}'.format(mu_prime_list[1]), linestyle='--')
        ax2.plot(x, y90p3_exact, color='r', label='Exact Analysis: mu_prime={}'.format(mu_prime_list[2]), linestyle='--')

    ax2.set(xlabel='P(speedup)', ylabel='90th Percentile')
    # ax2.legend(('mu_prime=2', 'mu_prime=2.5', 'mu_prime=3'))
    ax2.legend()

    save_path = os.path.join(os.getcwd(), plot_type + '.png')
    plt.savefig(save_path, dpi=600)


if __name__ == "__main__":
    # simulation_FCFS_single_class(nCustomer=100, nReplications=5, confidence_level=0.95, generate_data=True)
    # simulation_FCFS_single_class(nCustomer=5000, nReplications=30, confidence_level=0.95, generate_data=False)
    # simulation_priority_two_classes(nCustomer=100, nReplications=5, confidence_level=0.95, nClasses=2, generate_data=True)
    # simulation_priority_two_classes(nCustomer=5000, nReplications=30, confidence_level=0.95, nClasses=2, generate_data=
    simulation_MGnInfinity_single_class(nCustomer=5000, nReplications=30, confidence_level=0.95, generate_data=False)