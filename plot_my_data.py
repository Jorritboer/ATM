env_name = "4x4 Frozen Lake Environment (semi-slippery)"
fileName_env = "Lake_standard4_semi-slippery"
has_ideal_line = False

fileName_alg = ["BAM_QMDP", "BAM_QMDP+", "AMRL", "BAM_QMDP2", "BAM_QMDP+2"]
legend_alg = ["ATMQ", "Dyna-ATMQ", "AMRL-Q", "ATMQ2", "Dyna-ATMQ2"]

import numpy as np

# Plotting variables
plot_std = True  # Set wether or not to plot standard deviation (turn off to increase readability)
max_eps = (
    np.inf
)  # Set manually to restrict what episodes to plot (i.e. ignore graph after convergence)
interval = 0.95  # Determines plotted confidence interval (only if plot_std = True)
w1, w2 = 50, 1  # Determine smooting window and order

# Filename details (should stay constant!)
Data_path = "Data/Run2/"
fileName_begin = "AMData"
fileName_end = ".json"

# creating names:
Files_to_read = []
nmbr_files = len(fileName_alg)

for i in range(nmbr_files):
    Files_to_read.append(
        "{}_{}_{}{}".format(fileName_begin, fileName_alg[i], fileName_env, fileName_end)
    )

nmbr_files = len(Files_to_read)

# Imports
import json
import math as m
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
import scipy.stats as sts

timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")

# Plot Naming

Plot_names_title = ["Scalarized Return", "Average steps", "Average measurements"]
Plot_Y_Label = ["Scalarized Return", "Steps", "Measurements"]
Plot_names_file = ["Reward", "Steps", "Measures"]

nmbr_plots = len(Plot_names_title)

# Data to obtain:

nmbr_steps = []
measure_cost = []

avg_reward, std_reward, min_reward, max_reward = [], [], [], []
avg_cum_reward = []

avg_steps, std_steps, min_steps, max_steps = [], [], [], []
avg_measures, std_measures, min_measures, max_measures = [], [], [], []
avg_reward_noCost = []

std_reward_test = []

nmbr_eps = []
nmbr_runs = []
measure_cost = []

# Read data:

for file_name in Files_to_read:
    with open(Data_path + file_name) as file:
        contentDict = json.load(file)

        avg_reward.append(np.average(contentDict["reward_per_eps"], axis=0))
        std_reward.append(np.std(contentDict["reward_per_eps"], axis=0))

        avg_steps.append(np.average(contentDict["steps_per_eps"], axis=0))
        std_steps.append(np.std(contentDict["steps_per_eps"], axis=0))
        avg_measures.append(np.average(contentDict["measurements_per_eps"], axis=0))
        std_measures.append(np.std(contentDict["measurements_per_eps"], axis=0))

        nmbr_eps.append(int(contentDict["parameters"]["nmbr_eps"]))
        nmbr_runs.append(int(contentDict["parameters"]["nmbr_runs"]))
        measure_cost.append(float(contentDict["parameters"]["m_cost"]))

all_data = [
    (avg_reward, std_reward),
    (avg_steps, std_steps),
    (avg_measures, std_measures),
]
eps_to_plot = min(np.min(nmbr_eps), max_eps)


# Make and export plots
for i in range(nmbr_plots):

    plt.ylabel(Plot_Y_Label[i])
    plt.xlabel("Episode")

    x = np.arange(eps_to_plot)

    if has_ideal_line and i == 0:
        plt.plot(
            x,
            np.repeat(ideal_value, eps_to_plot),
            "k--",
            linewidth=1,
            label="optimal value",
        )

    for j in range(nmbr_files):

        y, std = all_data[i][0][j][:eps_to_plot], all_data[i][1][j][:eps_to_plot]
        y, std = savgol_filter(y, w1, w2), savgol_filter(std, w1, w2)
        miny, maxy = sts.norm.interval(
            interval, loc=y, scale=std / np.sqrt(nmbr_runs[j])
        )
        plt.plot(x, y, label=legend_alg[j])
        if plot_std:
            plt.fill_between(x, miny, maxy, alpha=0.1)

    plt.legend()
    # plt.savefig(Data_path+"Plots/Plot_{}_{}.pdf".format(fileName_env, Plot_names_file[i]))

    plt.show()
    plt.clf()
