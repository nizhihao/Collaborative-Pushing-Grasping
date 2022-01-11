#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
#########################################################################
plot_ablation_push.py: compare the different push policy's performance
    x axis: epoch number(0-1200)
    y axis: mean push time

    run the following code:
    cd ./src/drl_push_grasp/
    python plot_ablation_push.py './logs/YOUR-SESSION-DIRECTORY-NAME-HERE-01' './logs/YOUR-SESSION-DIRECTORY-NAME-HERE-02' './logs/YOUR-SESSION-DIRECTORY-NAME-HERE-03'
#########################################################################
'''
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plot options (change me)
interval_size = 120 # Report performance over the last 200 training steps
max_plot_iteration = 1200
# Parse session directories
parser = argparse.ArgumentParser(description='Plot performance of a session over training time.')
parser.add_argument('session_directories', metavar='N', type=str, nargs='+', help='path to session directories for which to plot performance')
args = parser.parse_args()
session_directories = args.session_directories

# Define plot colors (Tableau palette)
colors = [[078.0/255.0,121.0/255.0,167.0/255.0], # blue
          [255.0/255.0,087.0/255.0,089.0/255.0], # red
          [089.0/255.0,169.0/255.0,079.0/255.0], # green
          [237.0/255.0,201.0/255.0,072.0/255.0], # yellow
          [242.0/255.0,142.0/255.0,043.0/255.0], # orange
          [176.0/255.0,122.0/255.0,161.0/255.0], # purple
          [255.0/255.0,157.0/255.0,167.0/255.0], # pink 
          [118.0/255.0,183.0/255.0,178.0/255.0], # cyan
          [156.0/255.0,117.0/255.0,095.0/255.0], # brown
          [186.0/255.0,176.0/255.0,172.0/255.0]] # gray

# Create plot design
plt.ylim((6, 15))
plt.ylabel(' Mean Push Numbers',fontsize=18)
plt.xlim((0, max_plot_iteration))
plt.xlabel('Number of training epoches',fontsize=18)
plt.grid(axis='y', linestyle='-', color=[0.9,0.9,0.9])
plt.tick_params(labelsize=15)
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_color('#000000')
plt.rcParams.update({'font.size': 18})
plt.rcParams['mathtext.default']='regular'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

for session_idx in range(len(session_directories)):
    session_directory = session_directories[session_idx]
    color = colors[session_idx % 10]

    # Get logged data
    transitions_directory = os.path.join(session_directory, 'transitions')
    success_log = np.loadtxt(os.path.join(transitions_directory, 'success.log.txt'), delimiter=' ')
    pushout_log = np.loadtxt(os.path.join(transitions_directory, 'pushout.log.txt'), delimiter=' ')
    executed_action_log = len(success_log)

    temp_success_log = []
    for i in success_log:
        if i[0] not in pushout_log:
            temp_success_log.append([i[0], i[1]])

    success_log = np.array(temp_success_log)
    success_log = success_log[:, 1]
    max_iteration = min(executed_action_log, max_plot_iteration)

    # Initialize plot variables
    mean_push_numbers = []
    std_push_numbers = []
    max_push_numbers = []
    min_push_numbers = []
    for step in range(interval_size - 1, max_iteration + interval_size): 
        push_before_step = step - interval_size + 1
        temp_push_epoch = success_log[push_before_step:step + 1]
        temp_push_epoch = np.array(temp_push_epoch)
        temp_mean = np.mean(temp_push_epoch)
        temp_std = np.std(temp_push_epoch, ddof=1)
        mean_push_numbers.append(temp_mean)
        std_push_numbers.append(temp_std)
        max_push_numbers.append(temp_mean + temp_std/2)
        min_push_numbers.append(temp_mean - temp_std/2)

    # Plot push information
    mean_push_numbers = np.array(mean_push_numbers)
    plt.plot(range(0, len(mean_push_numbers)), mean_push_numbers, color=color, linewidth=3) # color='blue', linewidth=3)
    # plt.fill_between(range(0, len(mean_push_numbers)), max_push_numbers, min_push_numbers, color=color, alpha=0.3)
    legend_str = session_directories[session_idx]
    # remove the prefix of the legend
    # legend_str = legend_str.replace('./src/drl_push_grasp/scripts/', '')
    legend.append(legend_str)

plt.legend(legend, loc='lower left', fontsize=18)
plt.show()


