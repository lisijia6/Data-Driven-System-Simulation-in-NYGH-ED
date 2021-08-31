import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_clustered_bar_chart(df, plot_name, labels_list, x_title, y_title):
    """
    Constructs clustered bar chart given data and plot characteristics, and saves the plot to file

    @ params:
        df (pd.DataFrame): the DataFrame used to make clustered bar chart
        plot_name (str): the name of the plot
        labels_list (list): list of names for the rows in the data
        x_title (str): name of the x-axis
        y_tilte (str): name of the y-axis
    """

    plt.figure(figsize=(16, 6))
    plt.set_cmap(cmap='Set3')
    bar_wid = 0.12
    n_vars = len(df)

    # Set heights of bars
    bar_heights_list = []
    for i in range(n_vars):
        bar_heights_list.append(list(df.iloc[i, :].values))

    # Set position on x-axis
    x_position_list = []
    temp = np.arange(len(bar_heights_list[0]))
    x_position_list.append(temp)
    for i in range(1, n_vars):
        positions = [x + bar_wid for x in temp]
        x_position_list.append(positions)
        temp = positions.copy()

    # Make the plot
    for i, x_pos in enumerate(x_position_list):
        plt.bar(x_pos, bar_heights_list[i], label=labels_list[i], width=bar_wid, edgecolor='w')
        for j, v in enumerate(bar_heights_list[i]):
            plt.text(x_pos[j] - 0.08, v + 0.02, str(v), fontsize='xx-small')

    # Add labels
    plt.xlabel(x_title, fontweight='bold')
    plt.xticks([r + bar_wid for r in range(len(bar_heights_list[0]))], list(df.columns))
    plt.ylabel(y_title, fontweight='bold')

    # Create legend and save the plot
    plt.legend()
    plt.title(plot_name)
    save_path = os.path.join(os.getcwd(), "{}.png".format(plot_name))
    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    data = pd.read_excel('00_clustered_bar_example.xlsx', engine='openpyxl', header=0, index_col=0)
    plotname = 'Mean LOS values (T123 Admitted Patients)'
    labels = list(data.index)
    xlabel = 'Interventions'
    ylabel = 'Mean LOS (in minutes)'
    plot_clustered_bar_chart(df=data, plot_name=plotname, labels_list=labels, x_title=xlabel, y_title=ylabel)
