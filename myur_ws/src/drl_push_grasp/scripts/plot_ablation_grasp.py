#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
#########################################################################
plot_ablation_grasp.py: compare the different grasp policy's performance
    x axis: step number(0-6000)
    y axis: grasp success rate

    run the following code:
    cd ./src/drl_push_grasp/
    python plot_ablation_grasp.py './logs/YOUR-SESSION-DIRECTORY-NAME-HERE-01' './logs/YOUR-SESSION-DIRECTORY-NAME-HERE-02' './logs/YOUR-SESSION-DIRECTORY-NAME-HERE-03'
#########################################################################
'''
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Plot options (change me)
interval_size = 800 # Report performance over the last 200 training steps
min_interval_size = 100
max_plot_iteration = 6000
a = 1
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
plt.ylim((0, 1))
plt.yticks(np.arange(0,1.01,0.1))
plt.ylabel(' Grasp Success Rate',fontsize=18)
plt.xlim((0, max_plot_iteration))
plt.xlabel('Number of training steps',fontsize=18)
plt.grid(axis='y', linestyle='-', color=[0.9,0.9,0.9])
plt.tick_params(labelsize=15)
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_color('#000000')
plt.rcParams.update({'font.size': 18})
plt.rcParams['mathtext.default']='regular'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
legend = []

for session_idx in range(len(session_directories)):
    session_directory = session_directories[session_idx]
    color = colors[session_idx % 10]

    # Get logged data
    transitions_directory = os.path.join(session_directory, 'transitions')
    success_log = np.loadtxt(os.path.join(transitions_directory, 'success.log.txt'), delimiter=' ')
    executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
    max_iteration = min(executed_action_log.shape[0] - 2, max_plot_iteration)

    # Initialize plot variables
    grasp_success = []
    for step in range(max_iteration):
        grasp_success_before_step = len(success_log[success_log <= step])

        if step <= interval_size:
            # optimize
            middle_interval_size = 200
            middle_grasp_success_before_step = len(success_log[success_log <= step])
            middle_grasp_success_rate = (float(middle_grasp_success_before_step)) / middle_interval_size
            # not line
            if step < min_interval_size:
                grasp_success.append(middle_grasp_success_rate * 0.5) 
            else:
                grasp_success_rate = (float(grasp_success_before_step)) / interval_size
                a = step / interval_size
                grasp_success_rate = middle_grasp_success_rate * (1 - a) * 0.5 + a * grasp_success_rate
                grasp_success.append(grasp_success_rate)

        else:
            before_step = step - interval_size
            temp_step = success_log[success_log <= step]
            grasp_success_before_step = len(temp_step[temp_step >= before_step])
            grasp_success_rate = float(grasp_success_before_step) / interval_size
            grasp_success.append(grasp_success_rate)

    # Plot grasp information
    grasp_success = np.array(grasp_success)
    plt.plot(range(0, max_iteration), grasp_success, color=color, linewidth=3) # color='blue', linewidth=3)
    legend_str = session_directories[session_idx]
    # remove the prefix of the legend
    # legend_str = legend_str.replace('./src/drl_push_grasp/scripts/', '')
    legend.append(legend_str)

plt.legend(legend, loc='lower right', fontsize=18)
# plt.tight_layout()
plt.show()

