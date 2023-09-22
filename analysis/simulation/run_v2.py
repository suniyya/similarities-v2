import argparse
import logging
import pprint
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
from multiprocessing import Pool
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import analysis.geometry.euclidean as model
import analysis.simulation.experiment as exp
import analysis.simulation.experiment_ranking as exp_ranking
from analysis.simulation.simulate_coord_noise import run_experiment as coord_noise_run_exp
import analysis.model_fitting.pairwise_likelihood_analysis as an
import analysis.model_fitting.run_mds_seed as rs
from analysis.geometry.hyperbolic import loid_map, hyperbolic_distances, sphere_map, spherical_distances
from analysis.util import add_row, judgments_to_arrays

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
global args


def simulate_judgments(parameters, points):
    repeats = parameters['num_repeats'] # need to fix this
    if parameters['true_geometry'] == 'euclidean':
        distances = squareform(pdist(points))
    elif parameters['true_geometry'] == 'hyperbolic':  # if hyperbolic
        # project points to a n-dimensional hyperboloid and compute distances on the hyperboloid
        hyperbolic_points = loid_map(points.T, parameters['curvature'])
        distances = hyperbolic_distances(hyperbolic_points, parameters['curvature'])
    else:
        # project points to a n-dimensional spherical surface
        sphere_points = sphere_map(points.T, 1 / parameters['curvature'])
        distances = spherical_distances(sphere_points, 1 / parameters['curvature'])
        print(distances)

    if parameters['paradigm'] == 'pairwise_comparisons':
        judgments, repeats, points = exp.run_experiment(points, distances, parameters)
    elif parameters['paradigm'] == 'simple_ranking' and parameters['decision_noise_type'] != 'coordinates':
        data = exp_ranking.run_experiment(points, distances, parameters, simple_err_model=True)
        judgments = {}
        repeats = {}
        for trial, responses in data.items():
            pairwise_comparisons, repeats_dict = exp_ranking.ranking_to_pairwise_comparisons(
                exp_ranking.all_distance_pairs(trial),
                responses)
            for pair, judgment in pairwise_comparisons.items():
                if pair not in judgments:
                    judgments[pair] = judgment
                    repeats[pair] = repeats_dict[pair]
                else:
                    judgments[pair] += judgment
                    repeats[pair] += repeats_dict[pair]
                    # judgments[pair] = judgments[pair] / float(2)
    elif parameters['decision_noise_type'] == 'coordinates':
        # simple ranking trials. Noise only at level of coords not dist or comparison.
        print('simulating judgments with coord noise')
        judgments = coord_noise_run_exp(points, parameters)
    else:
        raise NotImplementedError('Invalid paradigm')
    return judgments, repeats, distances


def n_dim_minimization(true_model, dim, judgments, repeats_dict, params, ll_groundtruth):
    LOG.info('################  {} dimensional geometry'.format(dim))
    model_name = str(dim) + 'D'
    params['n_dim'] = dim
    x, ll_nd = rs.points_of_best_fit(judgments, repeats_dict, params)
    ll_nd = -ll_nd / float(len(judgments) * params['num_repeats'])
    new_row = {'Model': model_name, 'LL': ll_nd, 'Relative LL': ll_nd - ll_groundtruth,
               'Num Points': params['num_stimuli'], 'True Model': true_model}
    return new_row


def main(true_dim, parameters, min_model=1, max_model=5, show=False):
    LOG.info('################  True Dimension: {}'.format(true_dim))
    # SAMPLE STIMULI FROM A GEOMETRIC SPACE ###################################################
    space = model.EuclideanSpace(true_dim)
    if parameters['true_geometry'] == 'euclidean':
        true_model = str(true_dim) + 'D'
    elif parameters['true_geometry'] == 'hyperbolic':
        true_model = str(true_dim) + 'D-hyp-c' + str(parameters['curvature'])
    else:
        true_model = str(true_dim) + 'D-sph-c' + str(parameters['curvature'])
    points = space.get_samples(parameters['num_stimuli'], parameters['spread'], parameters['sampling_method'])
    print(points)

    # SIMULATE RANK JUDGMENTS #################################################################
    # based on input paradigm etc.
    pairwise_judgments, repeats_dict, interstim_distances = simulate_judgments(parameters, points)
    # randomly filter trials is max_trials is less than total comparisons
    if parameters['max_trials'] is not None and len(pairwise_judgments) > parameters['max_trials']:
        pairwise_judgments = dict(random.sample(pairwise_judgments.items(), parameters['max_trials']))
    trial_pairs_a, trial_pairs_b, judgment_counts, judgment_repeats = judgments_to_arrays(pairwise_judgments, repeats_dict)
    LOG.info('################  Number of binary comparisons: {}'.format(len(pairwise_judgments)))
    # initialize results dataframe
    result = {'Model': [], 'LL': [], 'Relative LL': [], 'Num Points': [], 'True Model': []}
    num_trials = len(pairwise_judgments)
    print('distances: ', interstim_distances)
    print('Mean distance', np.mean(interstim_distances))

    # GROUND TRUTH MODEL #######################################################################
    LOG.info('################  Ground truth geometry')
    parameters['n_dim'] = true_dim
    # calculate log-likelihood, is_bad flag
    probs = an.find_probabilities(
        interstim_distances, trial_pairs_a, trial_pairs_b,
        np.sqrt(parameters['sigmas']['dist'] ** 2 + parameters['sigmas']['compare'] ** 2), parameters['no_noise'])
    ll_groundtruth = an.calculate_ll(
        judgment_counts, probs, judgment_repeats)[0] / float(
        num_trials * parameters['num_repeats'])
    new_row = {'Model': 'ground truth', 'LL': ll_groundtruth, 'Relative LL': 0,
               'Num Points': parameters['num_stimuli'], 'True Model': true_model}
    result = add_row(new_row, result)
    # breakpoint()
    # RANDOM AND BEST MODELS ####################################################################
    LOG.info('################  Random and best geometry')
    ll_best = an.best_model_ll(
        pairwise_judgments, repeats_dict)[0] / float(num_trials * parameters['num_repeats'])
    new_row = {'Model': 'best', 'LL': ll_best, 'Relative LL': ll_best - ll_groundtruth,
               'Num Points': parameters['num_stimuli'], 'True Model': true_model}
    result = add_row(new_row, result)
    ll_random = an.random_choice_ll(
        pairwise_judgments, repeats_dict)[0] / float(num_trials * parameters['num_repeats'])
    new_row = {'Model': 'random', 'LL': ll_random, 'Relative LL': ll_random - ll_groundtruth,
               'Num Points': parameters['num_stimuli'], 'True Model': true_model}
    result = add_row(new_row, result)

    # MODELING WITH DIFFERENT EUCLIDEAN MODELS ###################################################
    #  use a different noise param for fitting than for simulating
    # parameters['sigmas'] = {'compare': 0.02 + parameters['sigmas']['compare'], 'dist': 0}
    # for dim in range(min_model, max_model + 1):
    #     new_row = n_dim_minimization(true_model, dim, pairwise_judgments, repeats_dict, parameters, ll_groundtruth)
    #     result = add_row(new_row, result)

    # MODELING WITH HYPERBOLIC MODEL OF CORRECT DIMENSION ########################################
    # also test correct hyperbolic model if true_geometry is hyperbolic
    if parameters['true_geometry'] == 'hyperbolic' or CONFIG['hyperbolic']:
        true_curvature = parameters['curvature']
        for c in [true_curvature]:  # np.arange(0.5 * true_curvature,  1 * true_curvature, 0.1):
            parameters['curvature'] = c
            LOG.info('Fitting hyperbolic model with parameters: ')
            pprint.pprint(parameters)
            LOG.info('################  {} dimensional hyperbolic geometry'.format(true_dim))
            model_name = str(true_dim) + 'D-hyp-c' + str(np.round(c, 2))
            parameters['n_dim'] = true_dim
            x, ll_nd = rs.hyperbolic_points_of_best_fit(pairwise_judgments, repeats_dict, parameters)
            ll_nd = -ll_nd / float(num_trials * parameters['num_repeats'])
            new_row = {'Model': model_name, 'LL': ll_nd, 'Relative LL': ll_nd - ll_groundtruth,
                       'Num Points': parameters['num_stimuli'], 'True Model': true_model}
            result = add_row(new_row, result)

    # MODELING WITH SPHERICAL MODEL OF CORRECT DIMENSION ########################################
    # also test correct hyperbolic model if true_geometry is hyperbolic
    if parameters['true_geometry'] == 'spherical':
        # breakpoint()
        true_curvature = parameters['curvature']
        for c in [true_curvature]:  # np.arange(0.1, 5 + true_curvature, 1):
            parameters['curvature'] = c
            LOG.info('Fitting spherical model with parameters: ')
            pprint.pprint(parameters)
            LOG.info('################  {} dimensional spherical geometry'.format(true_dim))
            model_name = str(true_dim) + 'D-sph-c' + str(np.round(c, 2))
            parameters['n_dim'] = true_dim
            x, ll_nd = rs.spherical_points_of_best_fit(pairwise_judgments, repeats_dict, parameters)
            ll_nd = -ll_nd / float(num_trials * parameters['num_repeats'])
            new_row = {'Model': model_name, 'LL': ll_nd, 'Relative LL': ll_nd - ll_groundtruth,
                       'Num Points': parameters['num_stimuli'], 'True Model': true_model}
            result = add_row(new_row, result)
            pprint.pprint(x)
            # PLOT ORIGINAL POINTS AND PROJECTED
            # fig = plt.figure()
            # x = x.T
            # center = np.mean(x, 1)
            # center = center.reshape((parameters['n_dim'], 1)) * np.ones((1, parameters['num_stimuli']))
            # x = x - center
            # ax = fig.add_subplot(projection='3d')
            # plt.plot(x[0, :], x[1, :], 'ko')
            # Y = sphere_map(x, c)
            # ax.scatter(Y[1, :], Y[2, :], Y[0, :], 'bo')
            # ax.scatter(points.T[0, :], points.T[1, :])
            # plt.show()
    # OUTPUT RESULTS ###############################################################################
    data_frame = pd.DataFrame(result)
    timestamp = time.asctime().replace(" ", '.')
    data_frame.to_csv(
        './simulation_{}_{}-likelihoods_{}_trials_{}_stimuli_sampled-from_{}_std_{}_{}_repeats_{}.csv'.format(
            parameters['true_geometry'],
            parameters['paradigm'],
            str(num_trials),
            parameters['num_stimuli'],
            parameters['sampling_method'],
            str(parameters['spread']),
            str(parameters['num_repeats']),
            timestamp))


# helper
def helper(ii):
    LOG.info('################  Iteration: {}'.format(ii))
    start = time.perf_counter()
    for n in range(args.min_true_dim, args.max_true_dim + 1):
        params = {
            'num_stimuli': args.num_stimuli,
            'max_trials': CONFIG['max_trials'],
            'n_dim': n,
            'no_noise': False,
            'sampling_method': args.sampling,
            'num_repeats': args.repetitions,
            'sigmas': {'compare': CONFIG['sigmas'], 'dist': 0},
            'paradigm': args.paradigm,
            'decision_noise_type': args.decision_noise,
            'true_geometry': args.geometry,
            'curvature': CONFIG['curvature'],
            'spread': args.sampling_radius,
            # 'dist_metric': 'euclidean',
            'epsilon': float(CONFIG['epsilon']),
            'fatol': float(CONFIG['fatol']),
            'tolerance': float(CONFIG['tolerance']),
            'max_iterations': int(CONFIG['max_iterations']),
            'learning_rate': float(CONFIG['learning_rate'])
        }
        pprint.pprint(params)
        np.random.seed()
        main(n, params, args.min_model_dim, args.max_model_dim)
    end = time.perf_counter()
    print('Time taken: {} minutes'.format((end - start) / 60.0))


if __name__ == '__main__':
    # define valid args
    with open('./analysis/config.yaml', "r") as stream:
        CONFIG = yaml.safe_load(stream)
    print(CONFIG)
    # ask for and parse user input
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--iterations", type=int, default=20)
    parser.add_argument("-n", "--num_stimuli", type=int, default=37)
    parser.add_argument("-g", "--geometry", choices=["euclidean", "hyperbolic", "spherical"], default="euclidean")
    parser.add_argument("-d1", "--min_model_dim", choices=[1, 2, 3, 4, 5], type=int, default=1)
    parser.add_argument("-d2", "--max_model_dim", choices=[1, 2, 3, 4, 5], type=int, default=1)
    parser.add_argument("-t1", "--min_true_dim", choices=[2, 3, 4, 5], type=int, default=1)
    parser.add_argument("-t2", "--max_true_dim", choices=[2, 3, 4, 5], type=int, default=1)
    parser.add_argument("-r", "--repetitions", help="Number of times to repeat a trial", type=int, default=5)
    parser.add_argument("-s", "--sampling", choices=["uniform", "spherical_shell", "gaussian"],
                        default="gaussian"),
    parser.add_argument("-p", "--paradigm", choices=["pairwise_comparisons", "simple_ranking"],
                        default="simple_ranking")
    parser.add_argument("-dn", "--decision_noise", choices=["comparison", "coordinates"],
                        default="comparison")
    parser.add_argument("-sr", "--sampling_radius", type=int, default=1)
    args = parser.parse_args()

    LOG.info("Running simulation with arguments: {}".format(args))
    if args.decision_noise == 'coordinates' and args.geometry != 'euclidean':
        raise NotImplementedError('Coordinate level noise has only been implemented for Euclidean distance metric!')

    pool = Pool(1)
    iterations = range(args.iterations)
    pool.map(helper, iterations)
    pool.close()
    pool.join()
