import json
import logging
import pprint
import random
import numpy as np
import pandas as pd
from sklearn.manifold import smacof
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

from analysis.util import combine_model_npy_files_to_mat, write_choice_probs_to_mat_no_ref,\
    write_choice_probs_to_mat_from_json
import analysis.mds as mds
import analysis.run_mds_seed as rs
import analysis.pairwise_likelihood_analysis as an
from analysis.preprocess_grouping import process_clicks
from analysis.util import ranking_to_pairwise_comparisons, all_distance_pairs, read_in_params, stimulus_names, \
    stimulus_id_to_name, stimulus_name_to_id

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# take processed experiment responses (in json format) from the appropriate folder

SHOW_MDS = False

CONFIG = read_in_params()
num_stim_per_trial = None

if __name__ == '__main__':
    # enter path to subject data (json file)
    PARADIGM = None
    while PARADIGM not in ['image_exp', 'working_mem_exp', 'grouping_exp', 'unconstrained_grouping_exp', 'materials_exp']:
        PARADIGM = input("> Name of experiment (choose from [image_exp, working_mem_exp, grouping_exp, unconstrained_grouping_exp, materials_exp]): ")
    if PARADIGM == 'working_mem_exp':
        ISI = input('Enter the ISI used (ms) (default= 1000')
        if ISI == '':
            paradigm = '-wm1000'
        else:
            paradigm = '-wm' + str(ISI)
    elif PARADIGM == 'grouping_exp':
        paradigm = '-gp'
    elif PARADIGM == 'unconstrained_grouping_exp':
        paradigm = '-gm'
    else:
        paradigm = ''  # used in filename of choices files

    CONDITIONS = input("> Name of condition, e.g., bc6pt9_DIS, bc6pt9 etc. (default= bgca3pt9 ):")
    if CONDITIONS == '' or CONDITIONS == ' ':
        CONDITIONS = 'bgca3pt9'
    if 'face' in CONDITIONS:
        print('NOTE: Select the json file starting with subject initials (not the prefix "final"')

    if 'grouping_exp' not in PARADIGM:
        FILENAME = input(
            "> Name of final json file with subject's preprocessed data: (e.g., final_BL_Sess1_10_bgca3pt9_exp.json)")
        FILEPATH = 'C://Users/jdvicto/Documents/similarities/experiments/{}/subject-data/{}_data/preprocessed/{}'.format(
            PARADIGM, CONDITIONS, FILENAME
        )
        print("Will read data from {}".format(FILEPATH))
    else:
        num_stim_per_trial = input("Number of stimuli in each trial (default is 8): ")
        if num_stim_per_trial == '':
            num_stim_per_trial = 8
        else:
            num_stim_per_trial = int(num_stim_per_trial)  # useful when processing clicks for gp paradigm

    EXP = CONDITIONS
    STIMULI = stimulus_names(EXP)
    NAMES_TO_ID = stimulus_name_to_id(EXP)
    ID_TO_NAME = stimulus_id_to_name(EXP)

    SUBJECT = input("> Subject Initials or ID: ")
    ITERATIONS = 1
    OUTDIR = input("> Output directory (C://Users/jdvicto/Documents/similarities/experiments/{}/subject-data) : "
                   .format(PARADIGM))
    SIGMA = input("> Enter noise parameter (if different from default):")
    if SIGMA != '' and SIGMA != ' ':
        CONFIG['sigma'] = {
            'dist': 0,
            'compare': float(SIGMA)
        }
    if OUTDIR[-1] == '/':
        OUTDIR = OUTDIR[:-1]
    MAX_DIM = input("> Enter max number of dimensions to run for (default=7): ")
    if MAX_DIM != '' and MAX_DIM != ' ':
        CONFIG['model_dimensions'] = [i for i in range(1, int(MAX_DIM) + 1)]
    pprint.pprint(CONFIG)
    ok = input("> Ok to proceed? (y/n)")
    if ok != 'y':
        raise InterruptedError

    sess_nums = None
    for ii in range(ITERATIONS):
        if 'grouping_exp' not in PARADIGM:
            # read json file into dict
            print('Reading data from ', FILEPATH)
            with open(FILEPATH) as file:
                ranking_responses_by_trial = json.load(file)

            # break up ranking responses into pairwise judgments
            pairwise_comparison_responses = {}
            pairwise_comparison_num_repeats = {}
            for config in ranking_responses_by_trial:
                comparisons, num_repeats = ranking_to_pairwise_comparisons(all_distance_pairs(config),
                                                                           ranking_responses_by_trial[config]
                                                                           )
                for key, count in comparisons.items():
                    pairs = key.split('<')
                    stim1, stim2 = pairs[1].split(',')
                    stim3, stim4 = pairs[0].split(',')
                    new_key = ((NAMES_TO_ID[stim1], NAMES_TO_ID[stim2]), (NAMES_TO_ID[stim3], NAMES_TO_ID[stim4]))
                    if new_key not in pairwise_comparison_responses:
                        pairwise_comparison_responses[new_key] = count
                        pairwise_comparison_num_repeats[new_key] = num_repeats[key]
                    else:
                        # if the comparison is repeated in two trials (context design side-effect)
                        pairwise_comparison_responses[new_key] += count
                        pairwise_comparison_num_repeats[new_key] += num_repeats[key]

        else:
            data_dir = 'C://Users/jdvicto/Documents/similarities/experiments/{}/subject-data/{}_data/raw'.format(
                PARADIGM, CONDITIONS
            )
            pairwise_comparison_responses = {}
            pairwise_comparison_num_repeats = {}
            rank_responses, repeats, sess_nums = process_clicks(SUBJECT, PARADIGM, data_dir, num_stim_per_trial)
            for key, count in rank_responses.items():
                pairs = key.split('<')
                stim1, stim2 = pairs[1].split(',')
                stim3, stim4 = pairs[0].split(',')
                new_key = ((NAMES_TO_ID[stim1], NAMES_TO_ID[stim2]), (NAMES_TO_ID[stim3], NAMES_TO_ID[stim4]))
                pairwise_comparison_responses[new_key] = count
                pairwise_comparison_num_repeats[new_key] = repeats[key]

        # get MDS starting coordinates
        D = mds.format_distances(mds.heuristic_distances(
            pairwise_comparison_responses, pairwise_comparison_num_repeats))
        coordinates2d, stress = smacof(D, n_components=2, metric=True, eps=1e-9)
        if SHOW_MDS:
            plt.plot(coordinates2d[:, 0], coordinates2d[:, 1], '.')
            for i, txt in enumerate(range(len(STIMULI))):
                plt.annotate(ID_TO_NAME[txt], (coordinates2d[i, 0], coordinates2d[i, 1]))
            plt.show()

        # only consider a subset of trials
        print('Num unique pairwise comparisons', len(pairwise_comparison_responses))
        if CONFIG['max_trials'] < len(pairwise_comparison_responses):
            indices = random.sample(pairwise_comparison_responses.keys(), CONFIG['max_trials'])
            subset = {key: pairwise_comparison_responses[key] for key in indices}
        else:
            subset = pairwise_comparison_responses

        # initialize results dataframe
        total_num_triads = sum([pairwise_comparison_num_repeats[k] for k in subset.keys()])
        print('Num total triads (or comparisons)', total_num_triads)
        result = {'Model': [], 'Log Likelihood': [], 'number of points': [],
                  'Experiment': [EXP] * (2 + len(CONFIG['model_dimensions'])),
                  'Subject': [SUBJECT] * (2 + len(CONFIG['model_dimensions']))}
        num_trials = len(subset)
        for dim in CONFIG['model_dimensions']:
            LOG.info('#######  {} dimensional model'.format(dim))
            model_name = str(dim) + 'D'
            CONFIG['n_dim'] = dim
            x, ll_nd = rs.points_of_best_fit(subset, pairwise_comparison_num_repeats, CONFIG)
            LOG.info("Points: ")
            print(x)
            outfilename = '{}/{}_{}_anchored_points_sigma_{}_dim_{}'.format(
                OUTDIR,
                SUBJECT, EXP,
                str(CONFIG['sigma']['compare'] + CONFIG['sigma']['dist']),
                dim
            )
            np.save(outfilename, x)

            LOG.info("Distances: ")

            distances = pdist(x)
            ll_nd = -ll_nd / float(total_num_triads)
            LOG.info('####### LL: {}'.format(np.round(ll_nd, 4)))
            result['Model'].append(model_name)
            result['Log Likelihood'].append(ll_nd)
            result['number of points'].append(CONFIG['num_stimuli'])

        LOG.info('#######  Random and best model')
        ll_best = an.best_model_ll(
            subset, pairwise_comparison_num_repeats)[0] / float(total_num_triads)
        result['Model'].append('best')
        result['Log Likelihood'].append(ll_best)
        result['number of points'].append(CONFIG['num_stimuli'])
        ll_random = an.random_choice_ll(
            subset, pairwise_comparison_num_repeats)[0] / float(total_num_triads)
        result['Model'].append('random')
        result['Log Likelihood'].append(ll_random)
        result['number of points'].append(CONFIG['num_stimuli'])
        data_frame = pd.DataFrame(result)
        sigma = CONFIG['sigma']['compare'] + CONFIG['sigma']['dist']
        data_frame.to_csv('{}/{}-{}-model-likelihoods_with_{}_trials_{}_iterations_sigma_{}_{}pts_anchored_{}.csv'
                          .format(OUTDIR,
                                  SUBJECT,
                                  EXP,
                                  CONFIG['max_trials'],
                                  CONFIG['max_iterations'],
                                  sigma,
                                  CONFIG['num_stimuli'],
                                  ii))

        # read metadata for filenames
        # session info
        if 'grouping_exp' not in PARADIGM:
            temp = FILENAME.split('Sess')[1]
            sessi = temp.split('_')[0]
            sessj = temp.split('_')[1]
            # also write choice probs file
            write_choice_probs_to_mat_from_json(EXP,
                                                FILEPATH,
                                                OUTDIR,
                                                "{}_choices_{}{}_sess0{}_{}".format(EXP[:-1] if EXP[-1] == '9' else EXP,
                                                                                    SUBJECT, paradigm, sessi, sessj))
        else:
            sessi = 1
            sessj = len(set(sess_nums))
            write_choice_probs_to_mat_no_ref(EXP, pairwise_comparison_responses, pairwise_comparison_num_repeats, OUTDIR,
                                      "{}_choices_{}{}_sess0{}_{}".format(
                                          EXP[:-1] if EXP[-1] == '9' else EXP, SUBJECT, paradigm, sessi, sessj))

        # also combine npy files into a mat file
        combine_model_npy_files_to_mat(OUTDIR, SUBJECT, EXP, paradigm, sessi, sessj, OUTDIR,
                                       CONFIG['model_dimensions'][0], CONFIG['model_dimensions'][-1])

