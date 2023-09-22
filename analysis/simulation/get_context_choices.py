import json
import numpy as np
from scipy.io import savemat

from analysis.describe_data import read_all_files, group_trials_by_ref, group_by_overlap
from analysis.util import ranking_to_pairwise_comparisons

subjects = ['AJ', 'SN', 'ZK', 'CME']  # ['MC', 'BL', 'EFV', 'SJ', 'SAW', 'YCL', 'SA', 'NK', 'AJ', 'SN', 'ZK', 'CME']
SUBJECTIDS = {'MC': 'S1', 'BL': 'S2', 'EFV': 'S3', 'SJ': 'S4', 'SAW': 'S5', 'NK': 'S6', 'YCL': 'S7',
              'SA': 'S8', 'JF': 'S9', 'AJ': 'S10', 'SN': 'S11', 'ZK': 'S12', 'CME': 'S13'}
EXP = 'image'
DATA_DIR = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/' \
           'experiments/{}_exp/subject-data/preprocessed'.format(EXP)
trials = read_all_files(subjects, DATA_DIR)
subset_trials_file = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/context-choice-probs/special' \
                     '-trials/special-trials.json'

with open(subset_trials_file) as user_file:
    file_contents = user_file.read()
subset_trials = json.loads(file_contents)
trial_list = subset_trials['designed-trials']


def context_choices(data, subject, exp, outdir='.', separate_special=False):
    subject_trials = group_trials_by_ref(data)
    context_trial_pairs = []
    # # iterate over all reference stimuli to check context dependence

    for animal in subject_trials:
        # group together pairs of trials that have 2 stimuli in common
        overlapping_trials = group_by_overlap(animal, subject_trials[animal])
        for d in overlapping_trials:
            d['ref'] = animal
        context_trial_pairs = context_trial_pairs + overlapping_trials

    context_triads = np.zeros((2, len(context_trial_pairs)))
    context_triads[:] = np.nan
    for idx in range(len(context_trial_pairs)):
        overlapping_pair = context_trial_pairs[idx]
        stimuli = sorted(list(overlapping_pair['stimuli']))
        binary_decision = '{},{}<{},{}'.format(
            overlapping_pair['ref'], stimuli[0],
            overlapping_pair['ref'], stimuli[1])
        dist_pair = [binary_decision]
        judgment1, repeat1 = ranking_to_pairwise_comparisons(
            dist_pair,
            data[overlapping_pair['1']]
        )
        judgment1 = judgment1[binary_decision]
        judgment2, repeat2 = ranking_to_pairwise_comparisons(
            dist_pair,
            data[overlapping_pair['2']]
        )
        judgment2 = judgment2[binary_decision]
        print(judgment1)
        print(judgment2)
        context_triads[0, idx] = judgment1
        context_triads[1, idx] = judgment2
        idx += 1

    outfilename = '{}/{}_{}_context_judgments.mat'.format(outdir, SUBJECTIDS[subject], exp)
    object_dict = {SUBJECTIDS[subject]: context_triads}
    savemat(outfilename, object_dict)


# special = filter_trials(trials, 'MC', trial_list, filter_out=False)
# print('MC')
# print(special)
# print(len(special))
# special = filter_trials(trials, 'SN', trial_list, filter_out=False)
# print('SN')
# print(special)
# print(len(special))
# others = filter_trials(trials, 'MC', trial_list, filter_out=True)
# print('MC')
# print(others)
# print(len(others))
# others = filter_trials(trials, 'SN', trial_list, filter_out=True)
# print('SN')
# print(others)
# print(len(others))


for sub in subjects:
    subj_trials = trials[sub]
    context_choices(subj_trials, sub, EXP, outdir='/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/'
                                                  'context-choice-probs/special-trials')
