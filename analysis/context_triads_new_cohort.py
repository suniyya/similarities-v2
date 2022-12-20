import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.util import ranking_to_pairwise_comparisons, read_json_file

from analysis.describe_data import group_trials_by_ref, group_by_overlap

PATH_TO_FILE = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/trial_configurations_new_cohort.xlsx'
sheet_data = pd.read_excel(PATH_TO_FILE, sheet_name='all trials (marked)')
PREPROCESSED_DATA_DIR = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/experiments/{}_exp/' \
                        'subject-data/preprocessed'


def read_in_context_triads(data):
    filtered = sheet_data.loc[data['context-pair'] != 0]
    circle_keys = ['stim' + str(i + 1) for i in range(8)]

    stimuli_in_common = {}
    context_trial_pairs = {}
    for ii in range(len(filtered)):
        row = filtered.iloc[ii]
        if row['context-pair'] not in context_trial_pairs:
            context_trial_pairs[row['context-pair']] = []
        else:
            current_trial_stim_in_ring = [filtered.iloc[ii][key] for key in circle_keys]
            prev_trial_stim_in_ring = context_trial_pairs[row['context-pair']][0].split(':')[1].split('.')
            stimuli_in_common[row['context-pair']] = sorted(list(
                set(current_trial_stim_in_ring).intersection(set(prev_trial_stim_in_ring))))
        trial_str = row['ref'] + ':'
        circle = sorted([filtered.iloc[ii][key] for key in circle_keys])
        for stim in circle:
            trial_str += stim + '.'
        trial_str = trial_str[:-1]
        context_trial_pairs[row['context-pair']].append(trial_str)

    pprint.pprint(context_trial_pairs)
    return context_trial_pairs, stimuli_in_common


def tabulate_subject_context_responses(subject, exp_name, trial_pairs, common_stim):
    # read in subject's json data and filter responses for given trial pairs
    data = read_json_file(subject, PREPROCESSED_DATA_DIR.format(exp_name))
    # filter data
    subset = {}
    for t in trial_pairs.values():
        subset[t[0]] = data[t[0]]
        subset[t[1]] = data[t[1]]
    # iterate over pairs of context trials
    # selectively look at judgment involving repeated triad stim_in_common[pairnum]
    num_pairs = max(trial_pairs.keys())

    ratio_map = np.zeros((6, 6))
    prob_in_context_ab = np.zeros((6, 6))

    for pair_num in range(1, num_pairs + 1):
        ref = trial_pairs[pair_num][0].split(':')[0]
        # first index - pair of context trials, second, trial num of pair
        binary_decision = '{},{}<{},{}'.format(ref, common_stim[pair_num][0],
                                               ref, common_stim[pair_num][1])
        judgment1 = ranking_to_pairwise_comparisons(
            [binary_decision],
            subset[trial_pairs[pair_num][0]],
        )[binary_decision]
        judgment2 = ranking_to_pairwise_comparisons(
            [binary_decision],
            subset[trial_pairs[pair_num][1]],
        )[binary_decision]
        print(pair_num)
        print(binary_decision)
        print(judgment1 / 5)
        print(judgment2 / 5)
        print('No context effect: ',
              str((judgment2 / 5 < 0.5 and judgment1 / 5 < 0.5) or (judgment2 / 5 > 0.5 and judgment1 / 5 > 0.5)))
        print('----------')

        prob_in_context_ab[judgment1, judgment2] += 1

    # PLOT just the 2D histogram . Only 23 entries...
    g = sns.heatmap(prob_in_context_ab, square=True,
                    xticklabels=[0, None, None, None, None, 1], annot=True,
                    yticklabels=[0, None, None, None, None, 1],  # cmap="bwr",
                    cbar=True)
    bottom, top = g.axes.get_ylim()
    plt.title(subject + '-' + exp_name)
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Choice Prob in Context A')
    plt.ylabel('Choice Prob in Context B')
    g.axes.invert_yaxis()
    plt.show()


context_pairs, stim_in_common = read_in_context_triads(sheet_data)
tabulate_subject_context_responses('ZK', 'word', context_pairs, stim_in_common)
