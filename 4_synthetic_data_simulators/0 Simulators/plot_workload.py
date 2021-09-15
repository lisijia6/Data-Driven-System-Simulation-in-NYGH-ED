from numpy import *

import matplotlib.pyplot as plt

def plot_K_graphs(quants, line_labels, title, y_label,colors, CI_Flag, ci_index, Lower, Upper):
    #assume they are all same length
    time_ls = linspace(1, len(quants[0]),  len(quants[0]))

    fig, ax = plt.subplots()
    fig.show()


    # -----------------------------------plot CI----------------------------------

    # plot the sampled workload
    for j,q in enumerate(quants):
        ax.plot(time_ls, q, colors[j], label=line_labels[j])
    # plot the sampled workload with CI
    #ax.plot(time_ls, quant2, 'r', label=l2)

    if CI_Flag:
        plot_mean_and_CI(array(quants[ci_index]), array(Lower), array(Upper), color_shading='g', lab = 'confidence_intervals')


    ax.legend(loc='best')
    plt.title(title, )

    plt.xlabel('Time (hr)')
    plt.ylabel(y_label)
    plt.show()

def plot_two_graphs(quant1, quant2, l1, l2, title, y_label):
    time_ls = linspace(1, min(len(quant1), len(quant1)),
                       min(len(quant2), len(quant2)))

    fig, ax = plt.subplots()
    fig.show()


    # -----------------------------------plot CI----------------------------------

    # plot the sampled workload
    ax.plot(time_ls, quant1, 'b', label=l1)
    # plot the sampled workload with CI
    ax.plot(time_ls, quant2, 'r', label=l2)

    ax.legend(loc='best')
    plt.title(title, )

    plt.xlabel('Time (hr)')
    plt.ylabel(y_label)
    plt.show()

def plot_mean_and_CI(mean, lb, ub, color_shading=None, lab = None):
    # plot the shaded range of the confidence intervals
    print(mean.shape[0])
    plt.fill_between(range(1, mean.shape[0]+1), lb, ub,
                     color=color_shading, alpha=.5, label = lab)
    # plot the mean on top
    #plt.plot(mean, color_mean)

def plot_sampled_trace(sampled_workload_list, real_workload_list, Upper, Lower, CI_Flag, label1, label2,title, y_label):
    time_ls = linspace(1, min(len(real_workload_list), len(sampled_workload_list)), min(len(real_workload_list), len(sampled_workload_list)))

    fig, ax = plt.subplots()
    fig.show()

    if CI_Flag:
        plt.axis([1, max(time_ls)+1, 0, max(max(Upper), max(real_workload_list))])
    else:
        plt.axis([1, max(time_ls)+1, 0, max(max(sampled_workload_list), max(real_workload_list))])

    # -----------------------------------plot CI----------------------------------

    # plot the sampled workload
    ax.plot(time_ls, sampled_workload_list, 'b', label=label1)
    # plot the sampled workload with CI
    ax.plot(time_ls, real_workload_list, 'r', label=label2)
    if CI_Flag:
        plot_mean_and_CI(array(sampled_workload_list), array(Lower), array(Upper), color_shading='g', lab = 'queueing_intervals')

    # plot the real workload

    ax.legend(loc='best')
    plt.title(title, )

    plt.xlabel('Time (hr)')
    plt.ylabel(y_label)
    plt.show()

def plot_one_only(workload_list, label):
    time_ls = linspace(1, len(workload_list), len(workload_list))

    fig, ax = plt.subplots()
    fig.show()

    plt.axis([1, max(time_ls)+1, 0, max(workload_list)])
    #plt.axis([0, max(time_ls), 0, max(max(sampled_workload_list), max(real_workload_list))])

    # -----------------------------------plot CI----------------------------------

    # plot the sampled workload
    ax.plot(time_ls, workload_list, 'r', label=label)
    # plot the sampled workload with CI
    #plot_mean_and_CI(np.array(sampled_workload_list), np.array(Lower), np.array(Upper), color_shading='g', lab = 'queueing_intervals')

    # plot the real workload

    ax.legend(loc='best')
    plt.title("Counts vs Time", )

    plt.xlabel('Time (hr)')
    plt.ylabel('Counts')
    plt.show()
