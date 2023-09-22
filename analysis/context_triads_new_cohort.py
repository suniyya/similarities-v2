import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import savemat

from analysis.util import ranking_to_pairwise_comparisons, read_json_file

from analysis.describe_data import group_trials_by_ref, group_by_overlap

PATH_TO_FILE = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/trial_configurations_new_cohort.xlsx'
sheet_data = pd.read_excel(PATH_TO_FILE, sheet_name='all trials (marked)')
PREPROCESSED_DATA_DIR = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/experiments/{}_exp/' \
                        'subject-data/preprocessed'
SUBJECTIDS = {'MC': 'S1', 'BL': 'S2', 'EFV': 'S3', 'SJ': 'S4', 'SAW': 'S5', 'NK': 'S6', 'YCL': 'S7',
              'SA': 'S8', 'JF': 'S9', 'AJ': 'S10', 'SN': 'S11', 'ZK': 'S12', 'CME': 'S13'}


def read_in_context_triads():
    filtered = sheet_data.loc[sheet_data['context-pair'] != 0]
    circle_keys = ['stim' + str(i + 1) for i in range(8)]

    stimuli_in_common = {}
    context_trial_pairs = {}
    choice = {}
    expected = {}
    set_context_trials = set()
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
        set_context_trials.add(trial_str)
        context_trial_pairs[row['context-pair']].append(trial_str)
        choice[row['context-pair']] = row['choice']
        expected[trial_str] = row['expected_prob']

    pprint.pprint(context_trial_pairs)
    return context_trial_pairs, stimuli_in_common, choice, expected, set_context_trials

#
# def tabulate_subject_context_response0(subject, exp_name, trial_pairs, common_stim):
#     # read in subject's json data and filter responses for given trial pairs
#     data = read_json_file(subject, PREPROCESSED_DATA_DIR.format(exp_name))
#     # filter data
#     subset = {}
#     for t in trial_pairs.values():
#         subset[t[0]] = data[t[0]]
#         subset[t[1]] = data[t[1]]
#     # iterate over pairs of context trials
#     # selectively look at judgment involving repeated triad stim_in_common[pairnum]
#     num_pairs = max(trial_pairs.keys())
#
#     ratio_map = np.zeros((6, 6))
#     prob_in_context_ab = np.zeros((6, 6))
#
#     for pair_num in range(1, num_pairs + 1):
#         ref = trial_pairs[pair_num][0].split(':')[0]
#         # first index - pair of context trials, second, trial num of pair
#         binary_decision = '{},{}<{},{}'.format(ref, common_stim[pair_num][0],
#                                                ref, common_stim[pair_num][1])
#         judgment1, rep1 = ranking_to_pairwise_comparisons(
#             [binary_decision],
#             subset[trial_pairs[pair_num][0]],
#         )
#         judgment1 = judgment1[binary_decision]
#         judgment2, rep2 = ranking_to_pairwise_comparisons(
#             [binary_decision],
#             subset[trial_pairs[pair_num][1]],
#         )
#         judgment2 = judgment2[binary_decision]
#         print(pair_num)
#         print(binary_decision)
#         print(judgment1 / 5)
#         print(judgment2 / 5)
#         print('No context effect: ',
#               str((judgment2 / 5 < 0.5 and judgment1 / 5 < 0.5) or (judgment2 / 5 > 0.5 and judgment1 / 5 > 0.5)))
#         print('----------')
#
#         prob_in_context_ab[judgment1, judgment2] += 1
#
#     # PLOT just the 2D histogram . Only 23 entries...
#     g = sns.heatmap(prob_in_context_ab, square=True,
#                     xticklabels=[0, None, None, None, None, 1], annot=True,
#                     yticklabels=[0, None, None, None, None, 1],  # cmap="bwr",
#                     cbar=True)
#     bottom, top = g.axes.get_ylim()
#     plt.title(subject + '-' + exp_name)
#     plt.ylim(bottom + 0.5, top - 0.5)
#     plt.xlabel('Choice Prob in Context A')
#     plt.ylabel('Choice Prob in Context B')
#     g.axes.invert_yaxis()
#     plt.show()


def get_context_choice_probs(subject, exp_name, context_pairs):
    data = read_json_file(subject, PREPROCESSED_DATA_DIR.format(exp_name))
    # if context trials only:
    num_pairs = max(context_pairs.keys())
    context_triads = np.zeros((2, num_pairs))
    context_triads[:] = np.nan
    subset = {}
    for t in context_pairs.values():
        subset[t[0]] = data[t[0]]
        subset[t[1]] = data[t[1]]
    for pair_num in range(1, num_pairs + 1):
        binary_decision = choice[pair_num]
        judgment1, rep1 = ranking_to_pairwise_comparisons(
            [binary_decision],
            subset[context_pairs[pair_num][0]]
        )
        judgment1 = judgment1[binary_decision]
        judgment2, rep2 = ranking_to_pairwise_comparisons(
            [binary_decision],
            subset[context_pairs[pair_num][1]]
        )
        judgment2 = judgment2[binary_decision]
        context_triads[0, pair_num-1] = judgment1
        context_triads[1, pair_num-1] = judgment2
    return context_triads


def write_context_choice_probs_to_file(subject, exp_name, context_pairs, outdir):
    # read in subject's json data and filter responses for given trial pairs
    context_triads = get_context_choice_probs(subject, exp_name, context_pairs)
    outfilename = '{}/{}_{}_context_judgments.mat'.format(outdir, SUBJECTIDS[subject], exp_name)
    object_dict = {SUBJECTIDS[subject]: context_triads}
    savemat(outfilename, object_dict)


def tabulate_subject_context_responses(subject, exp_name, context_pairs):
    # read in subject's json data and filter responses for given trial pairs
    data = read_json_file(subject, PREPROCESSED_DATA_DIR.format(exp_name))
    # filter data
    subset = {}
    for t in context_pairs.values():
        subset[t[0]] = data[t[0]]
        subset[t[1]] = data[t[1]]
    # iterate over pairs of context trials
    # selectively look at judgment involving repeated triad stim_in_common[pairnum]
    num_pairs = max(context_pairs.keys())
    heatmap = np.zeros((6, 6))
    # prob_in_context_ab = np.zeros((6, 6))

    for pair_num in range(1, num_pairs + 1):
        binary_decision = choice[pair_num]
        judgment1, rep1 = ranking_to_pairwise_comparisons(
            [binary_decision],
            subset[context_pairs[pair_num][0]]
        )
        judgment1 = judgment1[binary_decision]
        judgment2, rep2 = ranking_to_pairwise_comparisons(
            [binary_decision],
            subset[context_pairs[pair_num][1]]
        )
        judgment2 = judgment2[binary_decision]
        # x axis -> when expected choice prob was 0 and y axis when exp prob was 1
        if expected[context_pairs[pair_num][0]] == 0:
            x = judgment1
            y = judgment2
        else:
            x = judgment2
            y = judgment1
        heatmap[x, y] += 1
        print(pair_num)
        print(binary_decision)
        print(judgment1 / 5)
        print(judgment2 / 5)
        print('No context effect: ',
              str((judgment2 / 5 < 0.5 and judgment1 / 5 < 0.5) or (judgment2 / 5 > 0.5 and judgment1 / 5 > 0.5)))
        print('----------')

    # PLOT just the 2D histogram . Only 23 entries...
    g = sns.heatmap(heatmap, square=True,
                    xticklabels=[0, None, None, None, None, 1], annot=True,
                    yticklabels=[0, None, None, None, None, 1], cmap="BuPu",
                    cbar=True, vmin=0, vmax=7)
    bottom, top = g.axes.get_ylim()
    plt.title(subject + '-' + exp_name)
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Choice Prob Expected to be 0')
    plt.ylabel('Choice Prob Expected to be 1')
    g.axes.invert_yaxis()
    plt.show()


OUTDIR = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/context-choice-probs/special-trials'
context_pairs, stim_in_common, choice, expected, all_context_trials = read_in_context_triads()
# write_context_choice_probs_to_file('AJ', 'intermediate_texture', context_pairs, OUTDIR)

tabulate_subject_context_responses('CME', 'intermediate_texture', context_pairs)
tabulate_subject_context_responses('CME', 'image', context_pairs)
tabulate_subject_context_responses('CME', 'word', context_pairs)
