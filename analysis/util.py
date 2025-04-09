"""
Some utilities to help with processing psychophysical data files
"""
from scipy.io import savemat
import os
import yaml
import json
import glob
import pprint
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import pdist

CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'
# Read in parameters from config file
with open(CONFIG_PATH, "r") as stream:
    data = yaml.safe_load(stream)
    STIMULUS_LIST_FILE = data['path_to_stimulus_list']


def stimulus_names(cond):
    if cond is None:
        stimuli = open(STIMULUS_LIST_FILE).read().split('\n')
    else:
        if 'DIS' in cond.upper():
            cond = cond[:-4]
        elif '_' in cond:
            cond = cond.split('_')[0]
            print(cond)
        path = "{}/{}_stimuli.txt".format("./texture-exp-materials/", cond)
        if not os.path.isfile(path):
            path = "{}/{}_stimuli.txt".format("./face-exp-materials/", cond)
        stimuli = open(path).read().split('\n')
    if stimuli[-1] == '' or stimuli[-1] == ' ':
        stimuli = stimuli[:-1]
    print('Num stimuli: ', len(stimuli))
    return stimuli


def stimulus_name_to_id(cond):
    stimuli = stimulus_names(cond)
    names_to_id = dict(zip(stimuli, range(len(stimuli))))
    return names_to_id


def stimulus_id_to_name(cond):
    stimuli = stimulus_names(cond)
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


def bias_dict(use_all=False):
    path_to_bias_files = './analysis/bias-estimation/simulation_simple_ranking*.csv'
    if use_all:
        path_to_bias_files2 = './analysis/bias-estimation/*/simulation_simple_ranking*.csv'
    else:
        path_to_bias_files2 = ''
    # access simulation results and from these read out bias from RMS dist: sigma.
    sim_files = glob.glob(path_to_bias_files)
    sim_files2 = glob.glob(path_to_bias_files2)
    sim_files = sim_files + sim_files2
    df = pd.concat([pd.read_csv(f) for f in sim_files])
    return df


def read_out_median_bias(bias_df, dim, rms_ratio, tolerance=0.75, samples=40):
    # For a given value of RMS distance to sigma, read out the median bias between geometrically unconstrained
    # "best" model LL and the ground truth LL
    # for figure 5 variant, used tolerance of 0.5 for rms >= 0.5, tol =0.2 for rms <0.5 and samples =40
    biases_df = bias_df[bias_df['True Model'] == str(dim) + 'D']
    tol_val = tolerance
    df_temp = biases_df[biases_df['RMS:Sigma'].between(rms_ratio - tol_val, rms_ratio + tol_val)]
    if len(df_temp) < samples:
        print('WARNING: FEW SAMPLES TO ESTIMATE BIAS FOR RATIO ', np.round(rms_ratio, 2), dim)
        raise ValueError
    # print('Num samples ', len(df_temp), 'rms_ratio: ', rms_ratio)
    median_bias = np.quantile(df_temp['Best LL - Ground Truth LL'].sample(n=samples, random_state=942), 0.5)
    return median_bias


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
    with open(CONFIG_PATH, "r") as stream:
        user_config = yaml.safe_load(stream)
        sigma_compare = float(user_config['sigma'])
        total_noise = {'compare': sigma_compare, 'dist': 0}  # because downstream processing expects a key 'dist'
        user_config['sigma'] = total_noise
    # Fix type of all inputs
    user_config['num_stimuli'] = int(user_config['num_stimuli'])
    user_config['overlap'] = int(user_config['overlap'])
    user_config['num_stimuli_per_trial'] = int(user_config['num_stimuli_per_trial'])
    user_config['path_to_stimulus_list'] = str(user_config['path_to_stimulus_list'])
    user_config['max_trials'] = int(user_config['max_trials'])
    user_config['model_dimensions'] = [int(number) for number in list(user_config['model_dimensions'])]
    user_config['epsilon'] = float(user_config['epsilon'])
    user_config['minimization'] = str(user_config['minimization'])
    user_config['tolerance'] = float(user_config['tolerance'])
    user_config['max_iterations'] = int(user_config['max_iterations'])
    user_config['learning_rate'] = float(user_config['learning_rate'])
    user_config['n_dim'] = None
    user_config['no_noise'] = False
    user_config['verbose'] = False
    return user_config


def write_stimlist_from_image_folder(path_to_images, stimlist_filename, outdir='.'):
    # specifically written for JV's image files and their naming scheme
    files = glob.glob(path_to_images+'/*')
    unique_names = set()
    for name in files:
        unique_names.add(name.split('\\')[1].split('_')[0])
        # unique_names.add(name.split('\\')[1].split('.png')[0]) # used this for faces
    unique_names = sorted(list(unique_names))
    print(unique_names)

    f = open(stimlist_filename, 'w')
    final_str = '\n'.join(unique_names)
    print(final_str)
    print(len(unique_names))
    if final_str[-1] == '\n' or final_str[-1] == ' ':
        print('removing newline at end')
        final_str = final_str[0:-1]
    f.write(final_str)
    f.close()


def combine_model_npy_files_to_mat(directory, domain, subject, outdir='.', min_dim=1, max_dim=7):
    """
    Edited on Aug 3, 2023
    Add LL and biases too
    @param directory: input dir - dir in which is a domain dir then a subject dir
    @param subject:
    @param outdir:
    @param min_dim:
    @param max_dim:
    @return:
    """

    data = {'stim_labels': stimulus_names(domain)}
    bias_df = bias_dict()  # for LL bias estimation
    rms_dists_by_dim = {}
    for d in range(min_dim, max_dim + 1):
        model_files = glob.glob("{}/{}_{}_anchored_points_sigma_*_dim_{}.npy".format(
            directory, subject, domain, d
        ))
        print(model_files)
        # enter coordinates for each model dimension
        if len(model_files) > 0:
            model_file = model_files[0]
            points = np.array(np.load(model_file))
            data["dim{}".format(d)] = points
            distances = pdist(points)
            rms_dists_by_dim[d] = np.sqrt(np.mean([d ** 2 for d in distances]))
    # open LL file
    ll_file = glob.glob("{}/{}*{}*likelihoods*.csv".format(directory, subject, domain))
    print(ll_file)
    if len(ll_file) == 0:
        pass  # what does pass do?
    lls = pd.read_csv(ll_file[0])

    data['rawLLs'] = []  # enter raw log-likelihoods
    data['debiasedRelativeLL'] = []
    data['biasEstimate'] = []
    best_index = lls.index[lls['Model'] == 'best']
    best_LL = lls.iloc[best_index]['Log Likelihood'].values[0]
    data['bestModelLL'] = best_LL
    data['metadata'] = ("README\n\nrawLLs[i] is the raw model LL for model with i dimensions\n"
                        "biasEstimate[i] is the median bias estimated for the i-dimensional model, \n"
                        "  based on the RMS distance: sigma\n\n"
                        "debiasedRelativeLL = (rawLLs + biasEstimate) - bestModelLL\n"
                        "--------------------------------------------------------------------------")
    temp = {'bias': {}, 'debiasedLL': {}, 'rawLL': {}}
    for idx, row in lls.iterrows():
        model = 'dim' + str(row['Model'][:-1]) if row['Model'][-1] == 'D' else row['Model']
        if model[0:3] == 'dim':
            # get bias for each model LL
            dim = int(model[3:])
            temp['rawLL'][dim] = row['Log Likelihood']
            bias = read_out_median_bias(
                bias_df, dim, rms_dists_by_dim[dim], tolerance=0.5, samples=70)
            temp['bias'][dim] = bias
            # record debiased model LLs
            temp['debiasedLL'][dim] = row['Log Likelihood'] - (best_LL - bias)
    data['biasEstimate'] = [temp['bias'][key] for key in range(min_dim, max_dim + 1)]
    data['rawLLs'] = [temp['rawLL'][key] for key in range(min_dim, max_dim + 1)]
    data['debiasedRelativeLL'] = [temp['debiasedLL'][key] for key in range(min_dim, max_dim + 1)]
    print(data['biasEstimate'])
    pprint.pprint(data['stim_labels'])
    pprint.pprint(data['rawLLs'])
    # added sess01_10 in filename - default behavior - should change!
    savemat("{}/{}_coords_{}_sess.mat".format(outdir, domain[:-1] if domain[-1] == '9' else domain, subject), data)


# def combine_curvature_model_npy_files_to_mat(directory, domain, subject, sigma, outdir='.'):
#     """
#     BUGGY and untested - DiSTANCES NOT CALCULATED CORRECTLY
#     Created on Sept 25,'23
#     Add LL and biases too
#     @param directory: input dir - dir in which is a domain dir then a subject dir
#     @param subject:
#     @param outdir:
#     @return:
#     """
#
#     data = {'stim_labels': stimulus_names(domain)}
#     bias_df = bias_dict()  # for LL bias estimation
#     rms_dists_by_dim = {}
#
#     # read the csv file containing all likelihoods and details
#     # open LL file
#     ll_file = glob.glob("{}/curvature*{}-{}-combined_likelihoods.csv".format(directory, subject, domain))
#     if len(ll_file) == 0:
#         pass  # what does pass do?
#     lls = pd.read_csv(ll_file[0])
#     # for each entry find and read the corresponding file of coordinates
#     # read in row['Lambda-Mu'] = d - if d < 0 -> hyperbolic_model, else spherical_model, lambda_-d or mu_d,
#     # also read in row['Sigma']
#     # if row['Lambda-Mu'] = 0, look for ...lambda_0.npy or mu_0.npy
#     for idx, row in lls.iterrows():
#         curv_val = row['Lambda-Mu']
#         dim = int(row['Dimension'])
#         curv_type = 'lambda' if curv_val < 0 else 'mu'
#         model_type = 'hyperbolic' if curv_type == 'lambda' else 'spherical'
#         model_files = glob.glob("{}/{}_{}_{}_model_coords_sigma_{}_dim_{}_{}_{}.npy".format(
#             directory, subject, domain, model_type, sigma, dim, curv_type, curv_val)
#         )
#         # enter coordinates for each model dimension
#         if len(model_files) > 0:
#             model_file = model_files[0]
#             points = np.array(np.load(model_file))
#             data[curv_val] = points
#             # calculate distances correctly...
#             distances = pdist(points)
#             if not rms_dists_by_dim[dim]:
#                 rms_dists_by_dim[dim] = {}
#             rms_dists_by_dim[dim][curv_val] = np.sqrt(np.mean([d ** 2 for d in distances]))
#
#     data['rawLLs'] = []  # enter raw log-likelihoods
#     data['debiasedRelativeLL'] = []
#     data['biasEstimate'] = []
#     best_index = lls.index[lls['Model'] == 'best']
#     best_LL = lls.iloc[best_index]['Log Likelihood'].values[0]
#     data['bestModelLL'] = best_LL
#     data['metadata'] = ("README\n\nrawLLs[i] is the raw model LL for model with i dimensions\n"
#                         "biasEstimate[i] is the median bias estimated for the i-dimensional model, \n"
#                         "  based on the RMS distance: sigma\n\n"
#                         "debiasedRelativeLL = (rawLLs + biasEstimate) - bestModelLL\n"
#                         "--------------------------------------------------------------------------")
#     temp = {'bias': {}, 'debiasedLL': {}, 'rawLL': {}}
#     for idx, row in lls.iterrows():
#         model = 'dim' + str(row['Model'][:-1]) if row['Model'][-1] == 'D' else row['Model']
#         if model[0:3] == 'dim':
#             # get bias for each model LL
#             dim = int(model[3:])
#             temp['rawLL'][dim] = row['Log Likelihood']
#             bias = read_out_median_bias(
#                 bias_df, dim, rms_dists_by_dim[dim], tolerance=0.5, samples=70)
#             temp['bias'][dim] = bias
#             # record debiased model LLs
#             temp['debiasedLL'][dim] = row['Log Likelihood'] - (best_LL - bias)
#     data['biasEstimate'] = [temp['bias'][key] for key in range(min_dim, max_dim + 1)]
#     data['rawLLs'] = [temp['rawLL'][key] for key in range(min_dim, max_dim + 1)]
#     data['debiasedRelativeLL'] = [temp['debiasedLL'][key] for key in range(min_dim, max_dim + 1)]
#     savemat("{}/{}_coords-curvature_{}.mat".format(outdir,  domain[:-1] if domain[-1] =='9' else domain, subject), data)


def json_to_pairwise_choice_probs(cond, filepath):
    names_to_id = stimulus_name_to_id(cond)
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


def write_choice_probs_to_mat_from_json(cond, filepath, outdir, outfilename, include_names=False):
    """
        In output mat file, in responses matrix, ref, s1 and s2 go from 1-37 or 1-25 in JV's experiments.
        Because Matlab has 1-indexing this is better - allows indexing stim_list more natural.
        @param include_names:
        @param filepath: path to exp.json file with rank judgments
        @param outdir: directory to write mat file in
        @param outfilename: name of output file. May include subject and/ or condition and/ or num_sessions information
        @return:
        """
    responses_dict, n_repeats = json_to_pairwise_choice_probs(cond, filepath)
    write_choice_probs_to_mat(cond, responses_dict, n_repeats, outdir, outfilename, include_names=include_names)


def write_choice_probs_to_mat(cond, responses_dict, n_repeats, outdir, outfilename, include_names=False):
    first_pair, second_pair, comparison_counts, comparison_repeats = judgments_to_arrays(responses_dict, n_repeats)
    stim_list = stimulus_names(cond)
    responses_col_names = ['ref', 's1', 's2', 'N(D(ref, s1) > D(ref, s2))', 'N_Repeats(D(ref, s1) > D(ref, s2))']
    num_comparisons = len(first_pair)
    # hold (ref, s1, s2) tuples with labels instead of numbers
    ref_name = []
    s1_name = []
    s2_name = []

    responses = np.zeros((num_comparisons, len(responses_col_names)))

    for i in range(num_comparisons):
        ref = [s for s in first_pair[i] if s in second_pair[i]]
        if len(ref) != 1:
            raise ValueError('Expected one element in common. Just one ref')
        responses[i, 0] = ref[0] + 1
        s1 = [s for s in first_pair[i] if s != ref[0]][0]
        responses[i, 1] = s1 + 1
        s2 = [s for s in second_pair[i] if s != ref[0]][0]
        responses[i, 2] = s2 + 1
        responses[i, 3] = comparison_counts[i]
        responses[i, 4] = comparison_repeats[i]
        # record names of ref, s1 and s2 for the curre
        # nt comparison trial
        ref_name.append(stim_list[ref[0]])
        s1_name.append(stim_list[s1])
        s2_name.append(stim_list[s2])

    data = {
        'stim_list': stim_list,
        'responses_colnames': responses_col_names,
        'responses': responses
    }
    if include_names:
        data['ref_name'] = ref_name
        data['s1_name'] = s1_name
        data['s2_name'] = s2_name
    savemat("{}/{}.mat".format(outdir, outfilename), data)


def write_choice_probs_to_mat_no_ref(cond, responses_dict, n_repeats, outdir, outfilename, include_names=False):
    first_pair, second_pair, comparison_counts, comparison_repeats = judgments_to_arrays(responses_dict, n_repeats)
    stim_list = stimulus_names(cond)
    responses_col_names = ['s1', 's2', 's3', 's4', 'N(D(s1, s2) > D(s3, s4))', 'N_Repeats(D(s1, s2) > D(s3, s4))']
    num_comparisons = len(first_pair)
    # hold (s1, s2, s3, s4) tuples with labels instead of numbers
    s1_name = []
    s2_name = []
    s3_name = []
    s4_name = []

    responses = np.zeros((num_comparisons, len(responses_col_names)))

    for i in range(num_comparisons):
        s1, s2 = first_pair[i]
        s3, s4 = second_pair[i]
        responses[i, 0] = s1 + 1
        responses[i, 1] = s2 + 1
        responses[i, 2] = s3 + 1
        responses[i, 3] = s4 + 1
        responses[i, 4] = comparison_counts[i]
        responses[i, 5] = comparison_repeats[i]
        # record names of stimuli
        # nt comparison trial
        s1_name.append(stim_list[s1])
        s2_name.append(stim_list[s2])
        s3_name.append(stim_list[s3])
        s4_name.append(stim_list[s4])

    data = {
        'stim_list': stim_list,
        'responses_colnames': responses_col_names,
        'responses': responses
    }
    if include_names:
        data['s1_name'] = s1_name
        data['s2_name'] = s2_name
        data['s3_name'] = s3_name
        data['s4_name'] = s4_name
    savemat("{}/{}.mat".format(outdir, outfilename), data)


if __name__ == '__main__':
    conditions = ['dgea3pt']
    outdir = "." #"./experiments/grouping_exp/subject-data/old_coords_with_missing_lls"
    subjects = ['ZK']
    for condition in conditions:

        for subject in subjects:
            directory = "/Users/suniyya/Documents/lc809/similarities-homePC/experiments/working_mem_exp/subject-data/{}_data".format(
                condition)
            combine_model_npy_files_to_mat(directory, condition, subject, outdir=outdir, min_dim=1, max_dim=7)
# write_choice_probs_to_mat(condition,
#                           "{}/final_{}_Sess1_10_{}_exp.json".format(directory, subject, condition),
#                           outdir,
#                           "{}_choices_{}_sess01_10".format(condition[:-1], subject))

# write_stimlist_from_image_folder("C:/Users/jdvicto/Documents/similarities/texture-exp-materials/bc6pt9", "bc6pt9_stimuli.txt")
# import os
# from stat import S_IREAD, S_IRGRP, S_IROTH
# filename = "C:/Users/jdvicto/Documents/similarities/texture-exp-materials/dgea3pt_stimuli.txt"
# write_stimlist_from_image_folder("C:/Users/jdvicto/Documents/similarities/texture-exp-materials/dgea3pt",
#                                  filename)
# os.chmod(filename, S_IREAD | S_IRGRP | S_IROTH)
#

# filename = "X_stimuli.txt"
# write_stimlist_from_image_folder("C:/Users/jdvicto/Documents/similarities/face-exp-materials/X", filename, outdir='.')



