""" We test a new decision model with the purpose of conducting a simulation later to see if data generated with a
decision model involving only noise at the level of coordinates and not comparisons of distances can be accounted for
by a decision model that only accounts for comparison noise and not coordinate level ambiguity.

In each trial, we have a set of 8 points and a reference point.
Before calculating the distances, we jitter the points slightly by adding some noise to each coordinate value.
Distances are calculated and compared without any additional noise.

Like in simple ranking, we simply order these 8 distances from smallest to largest and those are our clicks.

"""

import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform

from analysis.simulation.experiment import create_trials
import analysis.simulation.experiment_ranking as exp_ranking


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def simulate_coord_noise_rank_judgments(trials, points, num_repeats, sigmas):
    """
    :param sigmas: default in DEFAULT
    :param num_repeats: default in DEFAULT
    :param trials: List of tuples containing trial configuration e.g.
                    (ref, (stim1, stim2, stim3, ..., stim8))]
    @param sigmas:
    @param num_repeats:
    @param points:
    @param trials:
    @param stimuli: coordinates of stimuli
    :return:responses (Dict) key=trial, value=clicked_list
    """
    num_stimuli = len(points)
    n_dim = len(points[0])
    noise_std = np.sqrt(sigmas['dist'] ** 2 + sigmas['compare'] ** 2)
    click_responses = {}
    for trial in trials:
        repeats = []
        ref, circle = trial[0], trial[1]
        num_clicks = len(trial[1])
        for _ in range(num_repeats):
            clicked = {}
            # jitter all points then calculate and compare distances
            # do so for each trial presentation
            # temp = points + np.random.normal(0, noise_std, size=(num_stimuli, n_dim))
            temp = np.zeros((num_stimuli, n_dim))
            temp[1:, :] = points[1:, :] + np.random.normal(0, noise_std, size=(num_stimuli-1, n_dim))
            temp[0, :] = points[0, :] + np.random.normal(0, 3*noise_std, size=(1, n_dim))
            all_distances = squareform(pdist(temp))
            distances = [all_distances[ref, _i] for _i in circle]
            # click stimuli in order of similarity, implicitly comparing similarities each time
            sorted_distances = sorted(distances)
            click_indices = [distances.index(v) for v in sorted_distances]
            click_values = [circle[_i] for _i in click_indices]
            for _j in range(num_clicks):
                clicked['s{}'.format(click_values[_j])] = _j
            repeats.append(clicked)
        click_responses[trial] = repeats
    return click_responses


def run_experiment(stimuli, args, trials=None):
    """
    Simulate an experiment where subjects are asked to rank stimuli in order of similarity to a changing
    central reference stimulus. The total number of trials is 222 and number of unique stimuli is 37. They make their
    decision based on the comparison of distances between pairs of points (subject to internal noise in comparing and
    computing distance). Noise level is controlled by args.
    :param stimuli: 37 points with coordinates provided
    :param simple_err_model: False if the error geometry involves noise before each click
    :param args dict, as in default (see DEFAULT)
    :param trials: if None, create configuration of full 222 using 37 stimuli, else used passed in list of trials
    @param args:
    @param trials:
    @param stimuli:
    @param distances: distances between stimuli
    """
    if trials is None:
        # prepare trial configuration of 222 trials using 37 stimuli as in real experiment
        trials = create_trials(stimuli, paradigm="ranking")
    LOG.info('##  Ranking paradigm trials created: %s', len(trials))
    LOG.info('##  Number of repeats per trial: %s', args['num_repeats'])
    data = simulate_coord_noise_rank_judgments(trials, stimuli, args['num_repeats'], args['sigmas'])
    LOG.info('##  Ranking judgments obtained')

    judgments = {}
    for trial, responses in data.items():
        pairwise_comparisons = exp_ranking.ranking_to_pairwise_comparisons(exp_ranking.all_distance_pairs(trial),
                                                                           responses)
        for pair, judgment in pairwise_comparisons.items():
            if pair not in judgments:
                judgments[pair] = judgment
            else:
                judgments[pair] += judgment
                judgments[pair] = judgments[pair] / float(2)
    LOG.info('##  Ranking judgments converted into choice probabilities and comparisons of pairs of dists')
    return judgments
