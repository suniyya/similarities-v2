import json
import logging
import pprint
import random
import numpy as np
import pandas as pd
from sklearn.manifold import smacof
from scipy.spatial.distance import pdist

import analysis.model_fitting.mds as mds
import analysis.model_fitting.run_mds_seed as rs
import analysis.model_fitting.pairwise_likelihood_analysis as an
from analysis.util import ranking_to_pairwise_comparisons, all_distance_pairs, read_in_params

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# take processed experiment responses (in json format) from the appropriate folder


def decompose_similarity_judgments(filepath, names_to_id):
    with open(filepath) as file:
        ranking_responses_by_trial = json.load(file)

    # break up ranking responses into pairwise judgments
    pairwise_comparison_responses_by_trial = {}
    for config in ranking_responses_by_trial:
        comparisons = ranking_to_pairwise_comparisons(all_distance_pairs(config),
                                                      ranking_responses_by_trial[config]
                                                      )
        for key, count in comparisons.items():
            pairs = key.split('<')
            stim1, stim2 = pairs[1].split(',')
            stim3, stim4 = pairs[0].split(',')
            new_key = ((names_to_id[stim1], names_to_id[stim2]), (names_to_id[stim3], names_to_id[stim4]))
            if new_key not in pairwise_comparison_responses_by_trial:
                pairwise_comparison_responses_by_trial[new_key] = count
            else:
                # if the comparison is repeated in two trials (context design side-effect)
                pairwise_comparison_responses_by_trial[new_key] += count
                pairwise_comparison_responses_by_trial[new_key] = pairwise_comparison_responses_by_trial[
                                                                      new_key] / 2.0
    return pairwise_comparison_responses_by_trial


if __name__ == '__main__':
    SHOW_MDS = False
    CONFIG, STIMULI, NAMES_TO_ID, ID_TO_NAME = read_in_params()
    ORIGINAL_CURVATURE = CONFIG['curvature']

    # enter path to subject data (json file)
    FILEPATH = input("Path to json file containing subject's preprocessed data"
                     " (e.g., ./sample-materials/subject-data/preprocessed/S7_sample_word_exp.json: ")
    EXP = input("Experiment name (e.g., sample_word): ")
    SUBJECT = input("Subject name or ID (e.g., S7): ")
    ITERATIONS = int(input("Number of iterations - how many times this should analysis be run (e.g. 1) : "))
    OUTDIR = input("Output directory (e.g., ./sample-materials/subject-data) : ")
    SIGMA = input("Enter number or 'y' to use default ({}):".format(str(
        CONFIG['sigmas']['compare'] + CONFIG['sigmas']['dist'])))
    if SIGMA != 'y':
        CONFIG['sigmas'] = {
            'dist': 0,
            'compare': float(SIGMA)
        }
    if OUTDIR[-1] == '/':
        OUTDIR = OUTDIR[:-1]
    pprint.pprint(CONFIG)
    ok = input("Ok to proceed? (y/n)")
    if ok != 'y':
        raise InterruptedError

    for ii in range(ITERATIONS):
        # read json file into dict
        pairwise_comparison_responses_by_trial = decompose_similarity_judgments(FILEPATH)
        # get MDS starting coordinates
        D = mds.format_distances(mds.heuristic_distances(pairwise_comparison_responses_by_trial, 5))
        coordinates2d, stress = smacof(D, n_components=2, metric=True, eps=1e-9)

        # only consider a subset of trials
        if CONFIG['max_trials'] < len(pairwise_comparison_responses_by_trial):
            indices = random.sample(pairwise_comparison_responses_by_trial.keys(), CONFIG['max_trials'])
            subset = {key: pairwise_comparison_responses_by_trial[key] for key in indices}
        else:
            subset = pairwise_comparison_responses_by_trial

        # initialize results dataframe
        result = {'Model': [], 'Log Likelihood': [], 'number of points': [],
                  'Experiment': [EXP] * (2 + len(CONFIG['model_dimensions'])),
                  'Subject': [SUBJECT] * (2 + len(CONFIG['model_dimensions'])),
                  'Curvature': []}

        # MODELING WITH DIFFERENT EUCLIDEAN MODELS ###################################################
        num_trials = len(subset)
        for dim in CONFIG['model_dimensions']:
            LOG.info('#######  {} dimensional geometry'.format(dim))
            model_name = str(dim) + 'D'
            CONFIG['n_dim'] = dim
            x, ll_nd, fmin_costs = rs.points_of_best_fit(subset, CONFIG)
            LOG.info("Points: ")
            print(x)
            outfilename = '{}/{}_{}_anchored_points_sigma_{}_dim_{}'.format(
                OUTDIR,
                SUBJECT, EXP,
                str(CONFIG['sigmas']['compare'] + CONFIG['sigmas']['dist']),
                dim
            )
            np.save(outfilename, x)
            LOG.info("Distances: ")
            distances = pdist(x)
            ll_nd = -ll_nd / float(num_trials * CONFIG['num_repeats'])
            result['Model'].append(model_name)
            result['Log Likelihood'].append(ll_nd)
            result['number of points'].append(CONFIG['num_stimuli'])
            result['Curvature'].append('')
        # the ii for loop can be taken out later. just need it for a plot
        #   plt.plot(fmin_costs)
        # plt.show()

        if CONFIG['hyperbolic']:
            # HYPERBOLIC MODELS #########################################################################
            hyp_dim = 2
            # FIRST OPTIMIZE CURVATURE ###############################
            MAX_ITERATIONS = CONFIG['max_iterations']
            INITIAL_ITERATIONS = 2000
            CONFIG['max_iterations'] = INITIAL_ITERATIONS
            max_ll = -np.inf
            c_max_ll = None
            hyp_start = None
            for c in np.arange(0.0001, CONFIG['curvature']+0.1, 0.05):
                CONFIG['curvature'] = c
                LOG.info('Fitting hyperbolic model with parameters: ')
                LOG.info('######################################### Hyperbolic model with curvature {}'.format(
                    str(CONFIG['curvature'])))
                CONFIG['n_dim'] = hyp_dim
                start, ll_nd, fmin_costs = rs.hyperbolic_points_of_best_fit(subset, CONFIG)
                ll_nd = -ll_nd / float(num_trials * CONFIG['num_repeats'])
                LOG.info('######################################## Log likelihood: {}'.format(str(ll_nd)))
                if ll_nd > max_ll:
                    max_ll = ll_nd
                    c_max_ll = c
                    hyp_start = start
                    LOG.info("#### Best curvature so far: " + str(c))
            # NEXT OPTIMIZE COORDINATES ##############################
            CONFIG['curvature'] = c_max_ll
            CONFIG['max_iterations'] = MAX_ITERATIONS - INITIAL_ITERATIONS
            LOG.info('Fitting hyperbolic model with parameters: ')
            LOG.info('################################### Hyperbolic model being optimized with curvature {}'.format(
                str(CONFIG['curvature'])))
            solution, ll_nd, fmin_costs = rs.hyperbolic_points_of_best_fit(subset, CONFIG, hyp_start)
            ll_nd = -ll_nd / float(num_trials * CONFIG['num_repeats'])
            result['Model'].append(str(hyp_dim) + 'D-hyp')
            result['Log Likelihood'].append(ll_nd)
            result['number of points'].append(CONFIG['num_stimuli'])
            result['Experiment'].append(EXP)
            result['Subject'].append(SUBJECT)
            result['Curvature'].append(np.round(c_max_ll, 2))

        if CONFIG['spherical']:
            # SPHERICAL MODELS #########################################################################
            sph_dim = 2
            # FIRST OPTIMIZE CURVATURE ###############################
            MAX_ITERATIONS = CONFIG['max_iterations']
            INITIAL_ITERATIONS = 2000
            CONFIG['max_iterations'] = INITIAL_ITERATIONS
            max_ll = -np.inf
            c_max_ll = None
            sph_start = None
            for c in np.arange(0.0001, ORIGINAL_CURVATURE + 0.1, 0.05):
                CONFIG['curvature'] = c
                LOG.info('Fitting spherical model with parameters: ')
                LOG.info('######################################### Spherical model with curvature {}'.format(
                    str(CONFIG['curvature'])))
                CONFIG['n_dim'] = sph_dim
                start2, ll_nd, fmin_costs = rs.spherical_points_of_best_fit(subset, CONFIG)
                ll_nd = -ll_nd / float(num_trials * CONFIG['num_repeats'])
                LOG.info('######################################## Log likelihood: {}'.format(str(ll_nd)))
                if ll_nd > max_ll:
                    max_ll = ll_nd
                    sph_start = start2
                    c_max_ll = c
                    LOG.info("#### Best curvature so far: " + str(c))
            # NEXT OPTIMIZE COORDINATES ##############################
            CONFIG['curvature'] = c_max_ll
            CONFIG['max_iterations'] = MAX_ITERATIONS - INITIAL_ITERATIONS
            LOG.info('Fitting spherical model with parameters: ')
            LOG.info('################################### Spherical model being optimized with curvature {}'.format(
                str(CONFIG['curvature'])))
            solution, ll_nd, fmin_costs = rs.spherical_points_of_best_fit(subset, CONFIG, sph_start)
            ll_nd = -ll_nd / float(num_trials * CONFIG['num_repeats'])
            result['Model'].append(str(sph_dim) + 'D-sph')
            result['Log Likelihood'].append(ll_nd)
            result['number of points'].append(CONFIG['num_stimuli'])
            result['Experiment'].append(EXP)
            result['Subject'].append(SUBJECT)
            result['Curvature'].append(np.round(c_max_ll, 2))

        # RANDOM AND BEST MODELS ####################################################################
        LOG.info('#######  Random and best geometry')
        ll_best = an.best_model_ll(
            subset, CONFIG)[0] / float(num_trials * CONFIG['num_repeats'])
        result['Model'].append('best')
        result['Log Likelihood'].append(ll_best)
        result['number of points'].append(CONFIG['num_stimuli'])
        result['Curvature'].append('')
        ll_random = an.random_choice_ll(
            subset, CONFIG)[0] / float(num_trials * CONFIG['num_repeats'])
        result['Model'].append('random')
        result['Log Likelihood'].append(ll_random)
        result['number of points'].append(CONFIG['num_stimuli'])
        result['Curvature'].append('')

        # OUTPUT RESULTS ###############################################################################
        data_frame = pd.DataFrame(result)
        sigma = CONFIG['sigmas']['compare'] + CONFIG['sigmas']['dist']
        data_frame.to_csv('{}/{}-{}-geometry-likelihoods_with_{}_trials_sigma_{}_{}pts_anchored_{}.csv'
                          .format(OUTDIR,
                                  SUBJECT,
                                  EXP,
                                  CONFIG['max_trials'],
                                  sigma,
                                  CONFIG['num_stimuli'],
                                  ii))
