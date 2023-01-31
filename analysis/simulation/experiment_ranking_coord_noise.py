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

from analysis.simulation.experiment import compare_similarity

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def simulate_judgments(trial_pairs, all_distances, sigmas, num_repeats=5, no_noise=False, verbose=False):
    """At each 'trial',
    one of the pairs will be chosen as the one that is 'more different.
    Given a list of trials, simulate what a subject would do when asked to choose the pair
    that is more different than the other.
    :param trial_pairs: contains configuration of stimuli for each trial
    @param all_distances: n by n distance matrix where n is the number of stimuli
    :param num_repeats: number of times to repeat a trial (default = 1)
    :param sigmas: a dictionary containing sd of noise sources (default above)
    :param no_noise: boolean denoting whether or not the subject's judgments are subject
           to error due to noise in processing steps/ estimation (default=False)
    :param verbose: boolean denoting whether the returned counts dict should have keys
           that are more readable.
    :return count, num_repeats: a tuple containing a counts dict (for each trial) and the
                    num_repeats arg

    """
    counts = {}  # initialize counts for data to be returned
    # the counts dictionary will hold judgements
    # key: "d_i,j > d_k,l" or "i,j>k,l"
    # value: number of times ij was judged to be greater than kl (int)

    # two ways to record judgments, verbose is for convenience, the other one 'make_key' is for
    # better for sequential processing
    count_key = make_verbose if verbose else make_key

    for top, bottom in trial_pairs:
        # create a readable key
        key = count_key(top, bottom)
        # get stimuli for the trial - top pair and bottom pair
        pair1 = (top[0], top[1])
        pair2 = (bottom[0], bottom[1])
        # record fraction of times pair1 was judged to be more different than pair2
        counts[key] = compare_similarity(pair1,
                                         pair2,
                                         all_distances,
                                         sigmas,
                                         num_repeats,
                                         no_noise
                                         )[0]


def simulate_coord_noise_rank_judgments(trials, stimuli, num_repeats, sigmas):
    """
    :param sigmas: default in DEFAULT
    :param num_repeats: default in DEFAULT
    :param trials: List of tuples containing trial configuration e.g.
                    (ref, (stim1, stim2, stim3, ..., stim8))]
    @param trials:
    @param stimuli: coordinates of stimuli
    :return:responses (Dict) key=trial, value=clicked_list
    """
    click_responses = {}
    for trial in trials:
        repeats = []
        ref, circle = trial[0], trial[1]
        num_clicks = len(trial[1])
        for _ in range(num_repeats):
            clicked = {}
            # calculate trial distances, assuming some noise in calculation
            # next, add more noise at the level of comparison and 'click' the stimulus with the shortest distance to ref
            noise_std = np.sqrt(sigmas['dist'] ** 2 + sigmas['compare'] ** 2)
            distances = [
                interstimulus_distance[ref, _i] + np.random.normal(0, noise_std) for _i in circle
            ]
            # click stimuli in order of similarity, implicitly comparing similarities each time
            sorted_distances = sorted(distances)
            click_indices = [distances.index(v) for v in sorted_distances]
            click_values = [circle[_i] for _i in click_indices]
            for _j in range(num_clicks):
                clicked['s{}'.format(click_values[_j])] = _j
            repeats.append(clicked)
        click_responses[trial] = repeats
    return click_responses
