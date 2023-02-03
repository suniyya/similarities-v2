"""
Some utilities to help with processing psychophysical data files
"""
import yaml
import json
from scipy.io import savemat
import glob
import numpy as np
from itertools import combinations


def stimulus_names():
    # Read in parameters from config file
    with open('./analysis/config.yaml', "r") as stream:
        data = yaml.safe_load(stream)
        stimulus_list_file = data['path_to_stimulus_list']
    stimuli = open(stimulus_list_file).read().split('\n')
    return stimuli


def stimulus_name_to_id():
    stimuli = stimulus_names()
    names_to_id = dict(zip(stimuli, range(len(stimuli))))
    return names_to_id


def stimulus_id_to_name():
    stimuli = stimulus_names()
    id_to_name = dict(zip(range(len(stimuli)), stimuli))
    return id_to_name


def read_json_file(subject, preprocessed_data_dir):
    files = glob.glob('{}/{}_*.json'.format(preprocessed_data_dir, subject))
    if len(files) == 0:
        raise FileNotFoundError
    elif len(files) > 1:
        raise ValueError("Multiple files exist for this subject. There should be only one json file for a subject.")
    else:
        filepath = files[0]
    with open(filepath) as file:
        contents = json.load(file)
    return contents


def all_distance_pairs(trial_key):
    trial = trial_key.split(':')
    ref = trial[0]
    pairs = list(combinations(trial[1].split('.'), 2))

    def helper(x):
        return '{},{}<{},{}'.format(ref, x[0], ref, x[1])

    return list(map(helper, pairs))


def ranking_to_pairwise_comparisons(distance_pairs, ranked_stimuli):
    """ Convert ranking data to comparisons of pairs of pairs of stimuli

    @param distance_pairs:
    :type distance_pairs: list
    :param ranked_stimuli:
    :type ranked_stimuli: list
    """
    # ranked_stimuli is a list of lists. each list is a 'repeat'
    rank = {}
    comparisons = {}
    num_repeats = {}
    for stimulus_list in ranked_stimuli:
        for index in range(len(stimulus_list)):
            rank[stimulus_list[index]] = index
        for pair in distance_pairs:
            dists = pair.split('<')
            stim1 = dists[0].split(',')[1]
            stim2 = dists[1].split(',')[1]
            if pair not in comparisons:
                comparisons[pair] = 1 if rank[stim1] < rank[stim2] else 0
                num_repeats[pair] = 1
            else:
                num_repeats[pair] += 1
                if rank[stim1] < rank[stim2]:
                    comparisons[pair] += 1
    return comparisons, num_repeats


def judgments_to_arrays(judgments_dict, repeats):
    """Instead of having trials be a dictionary with keys made of tuples,
    convert judgment keys and values into numpy arrays for faster operations """
    # the indices of the stimuli for each trial's "first" pair of stimuli
    first_pair = np.array([np.array(trial[0]) for trial in judgments_dict.keys()])
    # the indices of the stimuli for each trial's "second" pair of stimuli
    second_pair = np.array([np.array(trial[-1]) for trial in judgments_dict.keys()])
    comparison_counts = np.array([v for k, v in judgments_dict.items()], dtype='float')
    comparison_repeats = np.array([repeats[k] for k, v in judgments_dict.items()], dtype='float')
    return first_pair, second_pair, comparison_counts, comparison_repeats


# def reformat_key(comparison_trial_key):
#     names_to_ids = stimulus_name_to_id()
#     stim_pair1, stim_pair2 = comparison_trial_key.split('<')
#     stim1, stim2 = stim_pair1.split(',')
#     stim3, stim4 = stim_pair2.split(',')
#     return (names_to_ids[stim3], names_to_ids[stim4]), '>', (names_to_ids[stim1], names_to_ids[stim2])


# Should remove from use  - averages context trial probs - should not...
# def json_to_choice_probabilities(json_contents, to_index=False):
#     choice_probabilities = {}
#     for trial, rankings in json_contents.items():
#         pairs = all_distance_pairs(trial)
#         within_trial_comparisons = ranking_to_pairwise_comparisons(pairs, rankings)
#         for choice in pairs:
#             reformatted_choice = reformat_key(choice) if to_index else choice
#             if reformatted_choice not in choice_probabilities:
#                 choice_probabilities[reformatted_choice] = within_trial_comparisons[choice]/5.0
#             else:   # should only be some cases where a trial is repeated once - only once
#                 choice_probabilities[reformatted_choice] = (within_trial_comparisons[choice]/5.0 +
#                                                             choice_probabilities[reformatted_choice])/2.0
#     return choice_probabilities


def write_npy(outfilename, array):
    np.save(outfilename, array)


def read_npy(filename):
    return np.load(filename)


def add_row(fields, table):
    for fieldname, value in fields.items():
        table[fieldname].append(value)
    return table


def read_in_params():
    # Read in parameters from config file
    with open('./analysis/config.yaml', "r") as stream:
        user_config = yaml.safe_load(stream)
        sigma_compare = float(user_config['sigmas'])
        total_noise = {'compare': sigma_compare, 'dist': 0}  # because downstream processing expects a key 'dist'
        user_config['sigmas'] = total_noise
    # Fix type of all inputs
    user_config['num_stimuli'] = int(user_config['num_stimuli'])
    user_config['overlap'] = int(user_config['overlap'])
    user_config['num_stimuli_per_trial'] = int(user_config['num_stimuli_per_trial'])
    user_config['path_to_stimulus_list'] = str(user_config['path_to_stimulus_list'])
    user_config['max_trials'] = int(user_config['max_trials'])
    user_config['model_dimensions'] = [int(number) for number in list(user_config['model_dimensions'])]
    user_config['num_repeats'] = int(user_config['num_repeats'])
    user_config['epsilon'] = float(user_config['epsilon'])
    user_config['curvature'] = float(user_config['curvature'])
    user_config['minimization'] = str(user_config['minimization'])
    user_config['tolerance'] = float(user_config['tolerance'])
    user_config['max_iterations'] = int(user_config['max_iterations'])
    user_config['learning_rate'] = float(user_config['learning_rate'])
    user_config['n_dim'] = None
    user_config['no_noise'] = False
    user_config['verbose'] = False
    return (user_config,
            stimulus_names(),
            stimulus_name_to_id(),
            stimulus_id_to_name())


def combine_model_npy_files_to_mat(directory, subject, outdir='.', min_dim=1, max_dim=7):
    """
    @param directory: input dir - dir in which is a domain dir then a subject dir
    @param subject:
    @param outdir:
    @param min_dim:
    @param max_dim:
    @return:
    """
    domains = ['texture', 'intermediate_texture', 'intermediate_object', 'image', 'word',
               'texture_grayscale', 'texture_color']
    data = {'stim_labels': stimulus_names()}
    for domain in domains:
        data[domain] = {}
        for d in range(min_dim, max_dim + 1):
            model_files = glob.glob("{}/{}/{}/{}_{}_anchored_points_sigma_*_dim_{}.npy".format(
                directory, domain, subject, subject, domain, d
            ))
            if len(model_files) > 0:
                model_file = model_files[0]
                data[domain]["dim{}".format(d)] = np.array(np.load(model_file))
    savemat("{}/{}.mat".format(outdir, subject), data)


def json_to_pairwise_choice_probs(filepath):
    names_to_id = stimulus_name_to_id()
    with open(filepath) as file:
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
            new_key = ((names_to_id[stim1], names_to_id[stim2]), (names_to_id[stim3], names_to_id[stim4]))
            if new_key not in pairwise_comparison_responses:
                pairwise_comparison_responses[new_key] = count
                pairwise_comparison_num_repeats[new_key] = num_repeats[key]
            else:
                # if the comparison is repeated in two trials (context design side-effect)
                pairwise_comparison_responses[new_key] += count
                pairwise_comparison_num_repeats[new_key] += num_repeats[key]
    return pairwise_comparison_responses, pairwise_comparison_num_repeats
