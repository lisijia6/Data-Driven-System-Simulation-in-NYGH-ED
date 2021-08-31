import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import os


def get_simulated_los_lists(sheetname, sys_state_list):
    """
    Reads in simulated LOS data from Excel file

    @ params:
        sheetname (str): the name of the sheet to read from the file
        sys_state_list (list): specifies the list of system states to read simulated LOS data from

    @ return:
        los_lists (list): simulated LOS data in a list

    """

    # Initialize the simulated LOS list based on number of system states to plot
    n = len(sys_state_list)
    los_lists = []
    for i in range(n):
        los_lists.append([])

    sys_state_list_temp = sys_state_list.copy()
    if 12 in sys_state_list:
        r = n - 1
        sys_state_list_temp.remove(12)
    else:
        r = n

    # Read in data from file(s)
    for i in range(r):
        filename = 'LOS_Data_Simulated_System_State={}.xlsx'.format(sys_state_list_temp[i])
        print(filename, sheetname)
        df = pd.read_excel(filename, sheet_name=sheetname, engine='openpyxl')
        df = df.iloc[:, 1:]
        los_lists[i] = np.array(df.values.tolist()).reshape(1, -1)[0]

    if 12 in sys_state_list:
        filename = 'LOS_Data_Simulated_System_State=1+2.xlsx'
        print(filename, sheetname)
        df = pd.read_excel(filename, sheet_name=sheetname, engine='openpyxl')
        df = df.iloc[:, 1:]
        los_lists[n - 1] = np.array(df.values.tolist()).reshape(1, -1)[0]

    return los_lists


def get_actual_los_lists(filenames):
    """
    Reads in actual LOS data from CVS files for 3 patient types
    """

    actual_lists = [[], [], []]
    for i, filename in enumerate(filenames):
        print(filename)
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                actual_lists[i].append(round(float(row[0])))
    return actual_lists


def calculate_nbins(x, bwidth):
    """
    Calculates the number of bins based on maximum and minimum values, as well as the bin width
    """

    max_num = int(np.ceil(np.max(x) / bwidth)) * bwidth
    min_num = int(np.floor(np.min(x) / bwidth)) * bwidth
    r = max_num - min_num
    nbins = round(r / bwidth)
    return nbins


def compute_ks_2samp_critical_val(n1, n2, alpha):
    """
    Computes the critical value for two-sample KS test
    """

    c_alpha_dict = {0.1: 1.22, 0.05: 1.36, 0.025: 1.48, 0.01: 1.63, 0.005: 1.73, 0.001: 1.95}
    c_alpha = c_alpha_dict.get(alpha)
    return c_alpha * np.sqrt((n1 + n2) / (n1 * n2))


def plot_los_distributions(actual_los_list, simulated_los_lists, patient_type, sys_state_list, x_lim, bwidth):
    """
    Plots the relative frequency histograms of actual and simulated LOS for 1 patient type and saves the plot.
    In addition, performs two-sample Kolmogorov–Smirnov tests

    @ params:
        actual_los_list (list): list of actual LOS
        simulated_los_lists (list): list of simulated LOS
        patient_type (str): the name of patient type
        sys_state_list (list): specifies the list of system states to read simulated LOS data from
        x_lim (int): the maximum value on the x-axis
        bwidth (int): the bin width of the relative frequency histograms

    @ return:
        df_ks (np.DataFrame): the DataFrame with the results of the K-S test (system state & name, KS statistic,
        KS critical value at alpha = 0.05, KS test p-value)
    """

    # Set up the plot parameters
    sns.set_color_codes("colorblind")
    n = len(sys_state_list)
    fig = None
    if n == 1:
        fig, _ = plt.subplots(1, 1, figsize=(8, 6), sharex='all', sharey='all', clear=True)
    elif n == 2:
        fig, _ = plt.subplots(1, 2, figsize=(16, 6), sharex='all', sharey='all', clear=True)
    elif n <= 4:
        fig, _ = plt.subplots(2, 2, figsize=(16, 12), sharex='all', sharey='all', clear=True)
    elif n <= 6:
        fig, _ = plt.subplots(3, 2, figsize=(16, 16), sharex='all', sharey='all', clear=True)

    plot_name = 'Actual vs. Simulated LOS Relative Frequency ({})'.format(patient_type)
    fig.suptitle(plot_name)

    # Dictionaries of labels and colours
    labels_dict = {0: 'Simulated: General NIS', 1: 'Simulated: NIS by Patient Type', 2: 'Simulated: NIS by Zone',
                   3: 'Simulated: NIS by Patient Type x Zone', 12: 'Simulated: NIS by Patient Type + by Zone'}
    colours_dict = {0: 'b', 1: 'r', 2: 'y', 3: 'm', 12: 'g'}

    # Initialize KS test results DataFrame
    df_ks = pd.DataFrame(columns=['System state', 'System state name', 'KS statistic', 'KS critical value (alpha=0.05)', 'KS test p-value'])

    # Plot histograms
    for i, ax_n in enumerate(fig.get_axes()):
        # Plot histogram of actual LOS
        actual_nbins = calculate_nbins(actual_los_list, bwidth)
        act_hist, act_edges = np.histogram(actual_los_list, actual_nbins,
                                           (0.0, np.ceil(np.max(actual_los_list) / bwidth) * bwidth))
        act_freq = act_hist / float(act_hist.sum())
        act_edges = [round(e) for e in act_edges[:-1]]
        sns.barplot(x=act_edges, y=act_freq, label='Actual LOS', color='k', alpha=0.2, ax=ax_n)
        act_freq = [round(f, 2) for f in act_freq]

        # Plot histogram of simulated LOS
        simulated_nbins = calculate_nbins(simulated_los_lists[i], bwidth)
        sim_hist, sim_edges = np.histogram(simulated_los_lists[i], simulated_nbins,
                                           (0.0, np.ceil(np.max(simulated_los_lists[i]) / bwidth) * bwidth))

        sim_hist = sim_hist[0:len(act_freq)]
        sim_freq = sim_hist / float(sim_hist.sum())

        sim_edges = [round(e) for e in sim_edges[:-1]]
        sim_edges = sim_edges[0:len(act_freq)]

        sns.barplot(x=sim_edges, y=sim_freq, label=labels_dict.get(sys_state_list[i]),
                    color=colours_dict.get(sys_state_list[i]), alpha=0.2, ax=ax_n)
        # act_freq = [round(f, 2) for f in act_freq]

        # Specify plot characteristics
        ax_n.legend(loc="upper right")
        ax_n.label_outer()
        ax_n.set_xticks(np.arange(0, len(sim_edges) + 1, 5))

        # Perform K-S test on actual and simulated LOS data
        ks, ks_pval = stats.ks_2samp(actual_los_list, simulated_los_lists[i], alternative='two-sided', mode='auto')
        ks_2samp_critical_val = compute_ks_2samp_critical_val(n1=len(actual_los_list), n2=len(simulated_los_lists[i]),
                                                              alpha=0.05)
        print('Kolmogorov–Smirnov test ({}):  \nks stat={}, ks crit val={}, ks pval={}'.format(
            labels_dict.get(sys_state_list[i]), ks, ks_2samp_critical_val, ks_pval))

        # Save K-S test results in DataFrame
        df_ks.loc[i] = sys_state_list[i], labels_dict.get(sys_state_list[i]), round(ks, 5), round(ks_2samp_critical_val, 5), ks_pval
        if i == n - 1:
            break

    plt.xlim(0, x_lim)
    save_path = os.path.join(os.getcwd(), "{}.png".format(plot_name))
    plt.savefig(save_path, dpi=300)
    return df_ks


def qq_plot(data1, data2, patient_type, sys_state_list, plot_min=None, plot_max=None):
    """
    Plots the Q-Q plots of quantiles of actual and simulated LOS for 1 patient type and saves the plot

    @ params:
        data1 (list): quantiles from data on the x-axis --> actual LOS
        data2 (list): quantiles from data on the y-axis --> simulated LOS
        patient_type (str): the name of patient type
        simulated_los_lists (list): list of simulated LOS
        plot_min (int, default=None): minimum value for x-axis and y-axis
        plot_max (int, default=None): maximum value for x-axis and y-axis
    """

    # Set up the plot parameters
    sns.set_color_codes("colorblind")
    n = len(sys_state_list)
    fig = None
    if n == 1:
        fig, _ = plt.subplots(1, 1, figsize=(8, 6), sharex='all', sharey='all', clear=True)  # ((ax1))
    elif n == 2:
        fig, _ = plt.subplots(1, 2, figsize=(16, 6), sharex='all', sharey='all', clear=True)  # ((ax1, ax2))
    elif n <= 4:
        fig, _ = plt.subplots(2, 2, figsize=(16, 12), sharex='all', sharey='all', clear=True)  # ((ax1, ax2), (ax3, ax4))
    elif n <= 6:
        fig, _ = plt.subplots(3, 2, figsize=(16, 16), sharex='all', sharey='all', clear=True)  # ((ax1, ax2), (ax3, ax4), (ax5, ax6))

    plot_name = 'Q-Q Plot ({})'.format(patient_type)
    fig.suptitle(plot_name)

    # Dictionary of labels
    labels_dict = {0: 'General NIS', 1: 'NIS by Patient Type', 2: 'NIS by Zone',
                   3: 'NIS by Patient Type x Zone', 12: 'NIS by Patient Type + by Zone'}

    for i, ax_n in enumerate(fig.get_axes()):
        # Computes quantiles and plots them
        quantiles = min(len(data1), len(data2[i]))
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
        x_quantiles = np.quantile(data1, quantiles)
        y_quantiles = np.quantile(data2[i], quantiles)
        ax_n.scatter(x_quantiles, y_quantiles)

        # Finds the max and min values to use for the x-axis and y-axis
        data1_min, data1_max = min(data1), max(data1)
        data2_min, data2_max = min(data2[i]), max(data2[i])
        x_min = np.floor(min(data1_min, data2_min))
        x_max = np.ceil(max(data1_max, data2_max))
        x = np.linspace(x_min, x_max, int(x_max - x_min + 1))
        ax_n.plot(x, x, 'k-')  # Plots a 45-degree line

        ax_n.set_xlabel('Simulated LOS Quantiles: {}'.format(labels_dict.get(sys_state_list[i])))
        ax_n.set_ylabel('Actual LOS Quantiles')

        # Sets the x-axis limits and y-axis limits
        if (plot_min is not None) and (plot_max is not None):
            ax_n.set_xlim(plot_min, plot_max)
            ax_n.set_ylim(plot_min, plot_max)
        else:
            ax_n.set_xlim(x_min, x_max)
            ax_n.set_ylim(x_min, x_max)

        if i == n - 1:
            break

    save_path = os.path.join(os.getcwd(), "{}.png".format(plot_name))
    plt.savefig(save_path, dpi=300)


def main_rel_freq_histograms_qqplots_kstest(system_state_list):
    """
    This function is the main simulation results DataFrame transformation function that reads in the results data,
    parse the results, and save the results data into readable DataFrames. The transformed data will be saved in
    Excel format.

    @ params:
        filename (str): specifies the name of the file to read from
        nruns (int): specifies the number of replications of the simulation that was performed
        sys_state_list (list): specifies the list of system states (used for NIS features selection)

    @ return:
        transformed_results_filename (str): the name of the file that the transformed results data is saved in

    """

    print('Processing actual LOS files...')
    actual_los_filenames = ['LOS_Dist_Actual_Patient_Type=T123A.csv', 'LOS_Dist_Actual_Patient_Type=T123NA.csv', 'LOS_Dist_Actual_Patient_Type=T45.csv']
    actual_los_lists = get_actual_los_lists(actual_los_filenames)

    patient_types_short = ['T123A', 'T123NA', 'T45']
    patient_types_long = ['T123 Admitted', 'T123 Not Admitted', 'T45']

    # TODO: Modify these dictionaries accordingly for better fits for plot
    pt_xlim_dict = {'T123A' : 30, 'T123NA' : 30, 'T45' : 25}
    pt_bwidth_dict = {'T123A' : 60, 'T123NA' : 60, 'T45' : 30}
    pt_qqplotmax_dict = {'T123A' : 2200, 'T123NA' : 2300, 'T45' : 1300}

    df_kstest_results = []

    for i in range(len(patient_types_short)):
        pt_short, pt_long = patient_types_short[i], patient_types_long[i]
        xlim, bin_wid = pt_xlim_dict.get(pt_short), pt_bwidth_dict.get(pt_short)
        qqplot_max = pt_qqplotmax_dict.get(pt_short)

        print('\nProcessing simulated LOS for {}...'.format(patient_types_long[i]))
        simulated_los_lists = get_simulated_los_lists(pt_short, system_state_list)
        print('Plotting simulated LOS distribution(s) and conducting Kolmogorov–Smirnov test(s) for {}...'.format(patient_types_long[i]))
        df_ks = plot_los_distributions(actual_los_lists[i], simulated_los_lists, pt_long, system_state_list, x_lim=xlim, bwidth=bin_wid)
        df_kstest_results.append(df_ks)
        print('Plotting Q-Q plot(s) for {}...'.format(patient_types_long[i]))
        qq_plot(actual_los_lists[i], simulated_los_lists, pt_long, system_state_list, plot_min=0, plot_max=qqplot_max)

    print('Saving KS-test results...')
    with pd.ExcelWriter('00_KS_test_results.xlsx') as writer:
        for i, df in enumerate(df_kstest_results):
            df.to_excel(writer, sheet_name='Patient_Type_{}'.format(patient_types_short[i]))

