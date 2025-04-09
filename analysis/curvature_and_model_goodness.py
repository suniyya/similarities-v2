"""
Here, given a data set, generate surrogate datasets that are drawn from the original choice probablities.
Then, for each surrogate dataset, essentially calculate LLs for a range of curvatures and (chosen/ adjusted)
sigma values.

This should yield a scatterplot of model LLs by curvature values. We start by doing so for the 2D case and we can
progress to higher dimensions as needed.
######## DEPRECATED - DO NOT USE ###########

The inputs are
- a range of curvature values
- dimensions for which to apply the analysis. Minimum dimension is 2.
- path to json files for individual datasets.
- optionally, can and probably should pass in a legend for when multiple curves are drawn
- number of iterations (surrogates)
- max number of iterations to run the model_fitting pipeline with.
"""

import glob
import pprint
import copy
import logging
import numpy as np
from scipy.spatial.distance import squareform, pdist
from analysis import run_mds_seed as rs
import pandas as pd
from analysis.geometry.hyperbolic import spherical_distances, sphere_map, hyperbolic_distances, loid_map
import analysis.util as util

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

CONFIG = util.read_in_params()
# disable unneeded params
CONFIG['curvature'] = None
CONFIG['spherical'] = None
CONFIG['hyperbolic'] = None


def rms_dist_curvature(path_to_data_dir, condition, dimensions, num_dists=666):
    # Copied from bias_estimate.py and edited on July 3, 2023
    # !calculate distances based on curvature param value!
    path_to_npy_files = '{}/*sigma_1.0*dim_{}*.npy'
    rms_sigma = {}
    for dimension in dimensions:
        npy_files = glob.glob(path_to_npy_files.format(path_to_data_dir, dimension))
        for npy_file in npy_files:
            # get filename and use metdata to populate subject, domain, curvature vals
            segments = npy_file.split('/')[-1].split('_')
            subname = segments[0]
            domain = condition
            curvature_param = float(segments[-1].split('.npy')[0])
            points = np.load(npy_file)
            # intialize dicts to ensure no key error
            if subname not in rms_sigma:
                rms_sigma[subname] = {}
            if domain not in rms_sigma[subname]:
                rms_sigma[subname][domain] = {}
            if dimension not in rms_sigma[subname][domain]:
                rms_sigma[subname][domain][dimension] = {}
            # read in each dataset, from points of different dimensions calculate the RMS of distances
            # save the ratio of RMS distance to sigma. We know this value rises with model dimension
            # and varies across subjects
            # and experiments
            if curvature_param == 0:
                distances = squareform(pdist(points))
            elif curvature_param < 0:
                distances = hyperbolic_distances(loid_map(points.T, abs(curvature_param)), abs(curvature_param))
            else:
                distances = spherical_distances(sphere_map(points.T, 1 / curvature_param), 1 / curvature_param)
            num_stim = distances.shape[0]

            unique_pairs_dists = []
            for i in range(num_stim):
                for j in range(i):
                    unique_pairs_dists.append(distances[i, j])

            if len(unique_pairs_dists) != num_dists:
                print("ERROR wrong num dists")
            rms = np.sqrt(np.mean([d ** 2 for d in distances]))

            # record rms if not previously computed.
            rms_sigma[subname][domain][dimension][curvature_param] = rms

        # pprint.pprint(rms_sigma)
    return rms_sigma


def get_best_model_lls(path_to_ll_csv_dir, condition):
    # geometric-modeling
    path_to_ll_csv_files = '{}/*{}-model-likelihood*.csv'.format(path_to_ll_csv_dir, condition)
    best_ll_dict = {}
    files = glob.glob(path_to_ll_csv_files)
    for file in files:
        filename = file.split('/')[-1]
        segments = filename.split('-')
        subname = segments[0]
        domain = segments[1]
        results = pd.read_csv(file)
        best_index = results.index[results['Model'] == 'best']
        best_ll = results.iloc[best_index]['Log Likelihood']
        if subname not in best_ll_dict:
            best_ll_dict[subname] = {}
        best_ll_dict[subname][domain] = best_ll
    return best_ll_dict


def run(args):
    judgments, repeats, CONFIG, subject, domain, dim, OUTDIR = args

    degree_curvature = [0, 0.25/2, 0.5/2, 0.75/2, 0.5, 1.25/2, 1.5/2, 1.75/2, 1, 2.5/2, 1.5, 3.5/2, 2, 4.5/2, 2.5]
    degree_curvature_h = [0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75, -2, -2.5, -3, -3.5, -4, -4.5, -5]

    def fit_model(similarity_judgments, num_repeats_dict, curvature, params, dim, start_points):
        params_copy = copy.deepcopy(params)

        total_num_triads = sum([num_repeats_dict[k] for k in similarity_judgments.keys()])
        noise = np.sqrt(params['sigma']['compare'] ** 2 + params['sigma']['dist'] ** 2)
        params_copy['n_dim'] = dim  # ensure correct model is tested
        if curvature == 0:
            # fit Euclidean model
            x, ll = rs.points_of_best_fit(similarity_judgments, repeats, params_copy, start_points)
            ll = -1 * ll / total_num_triads
        elif curvature > 0:
            # fit spherical model
            curvature_val = curvature
            params_copy['curvature'] = curvature_val
            x, ll = rs.spherical_points_of_best_fit(similarity_judgments, repeats, params_copy, start_points)
            ll = -1 * ll / total_num_triads
        else:
            # fit hyperbolic model
            curvature_val = -1 * curvature
            params_copy['curvature'] = curvature_val
            x, ll = rs.hyperbolic_points_of_best_fit(similarity_judgments, repeats, params_copy, start_points)
            ll = -1 * ll / total_num_triads
        return ll, curvature, noise, x

    results = {'Log Likelihood': [], 'Lambda-Mu': [], 'Curvature of Space': [], 'Sigma': [], 'Dimension': [], 'Subject': [],
               'Domain': []}
    start_euclidean = None
    for _c in range(len(degree_curvature)):
        c = degree_curvature[_c]
        if _c == 0:
            start = None
        log_likelihood, curvature_val, sigma, coords = fit_model(judgments, repeats, c, CONFIG, dim, start)
        if c == 0:
            start_euclidean = coords
        LOG.info("Log Likelihood: {}".format(log_likelihood))
        LOG.info("Sph fit points: ")
        outfilename = '{}/{}_{}_spherical_model_coords_sigma_{}_dim_{}_mu_{}'.format(
            OUTDIR,
            subject, domain,
            str(sigma),
            dim,
            curvature_val
        )
        np.save(outfilename, coords)
        start = coords
        # write to pandas file
        results['Lambda-Mu'].append(curvature_val)
        results['Curvature of Space'].append(curvature_val * 2) # 2 mu
        results['Log Likelihood'].append(log_likelihood)
        results['Sigma'].append(sigma)
        results['Dimension'].append(dim)
        results['Subject'].append(subject)
        results['Domain'].append(domain)
    for _c in range(len(degree_curvature_h)):
        c = degree_curvature_h[_c]
        if _c == 0:
            start = start_euclidean
        log_likelihood, curvature_val, sigma, coords = fit_model(judgments, repeats, c, CONFIG, dim, start)
        LOG.info("Log Likelihood: {}".format(log_likelihood))
        LOG.info("Hyp fit points: ")
        outfilename = '{}/{}_{}_hyperbolic_model_coords_sigma_{}_dim_{}_lambda_{}'.format(
            OUTDIR,
            subject, domain,
            str(sigma),
            dim,
            curvature_val
        )
        np.save(outfilename, coords)
        start = coords
        # write to pandas file
        results['Lambda-Mu'].append(curvature_val)
        results['Curvature of Space'].append(curvature_val)  # curv = lambda, for hyp case
        results['Log Likelihood'].append(log_likelihood)
        results['Sigma'].append(sigma)
        results['Dimension'].append(dim)
        results['Subject'].append(subject)
        results['Domain'].append(domain)
    # # write df
    df = pd.DataFrame(results)
    return df


# NOTE: ########################################
# Values of lambda and mu are hard-coded inside run function
# curvature was lambda^2 and mu^2 = 1/R^2 according to definition of Gaussian curvature...
# now corrected to lambda for hyp and 2mu for sph


if __name__ == '__main__':
    print(CONFIG)

    domain = input('Domain: ')
    SUBJECTS = input('Subjects (separated by spaces): ').split(' ')
    print(SUBJECTS)
    proceed = input('If subjects correct, press "y" to proceed')
    if proceed != 'y':
        raise IOError

    OUTDIR = input('Output directory for LLs and coordinates: ')
    PARADIGM = input('What was the data collection paradigm? (Options: "brightness", "unconstrained_grouping", '
                     '"similarity", "grouping"): ')  # does not include dis atm  only sim implemented

    STIMULI = util.stimulus_names(domain)
    NAMES_TO_ID = util.stimulus_name_to_id(domain)
    ID_TO_NAME = util.stimulus_id_to_name(domain)
    num_stim = 25

    # simply for file naming
    # would need to be checked  for running other than similarity
    # extended to make sure it works correctly with dis data - preprocess_dis check to see if choices flipped
    PARADIGM_LABELS = {'similarity': '', 'brightness': '-br', 'grouping': '-gp', 'unconstrained_grouping': '-gm'}

    for subject in SUBJECTS:
        print(subject)

        if not (PARADIGM == 'similarity' or PARADIGM == 'brightness'):
            raise ImplementationError
            # data_dir = '/Users/suniyya/Documents/' \
            #            'psg-texture-data/{}_data/raw'.format(domain, subject, domain)
            # judgments, repeats = util.csv_to_pairwise_choice_probs(data_dir, subject, num_stim_per_trial=8)
        else:
            INPUT_DATA = './experiments/image_exp/subject-data/{}_data/preprocessed/final_{}_Sess1_10_{}_exp.json'.format(domain, subject, domain)
            judgments, repeats = util.json_to_pairwise_choice_probs(domain, INPUT_DATA)

        dfs_for_curv_vals = []
        for DIM in CONFIG['model_dimensions']:
            subj_outdir = OUTDIR + '/likelihoods/' + subject
            ARGS = (judgments, repeats, CONFIG, subject, domain, DIM, subj_outdir)
            result = run(ARGS)
            dfs_for_curv_vals.append(result)
            print(result)

        total_df = pd.concat(dfs_for_curv_vals)
        print(total_df)
        total_df.to_csv('{}/curvature_and_LL_{}-{}-combined_likelihoods.csv'.format(subj_outdir, subject, domain))


        num_stim_pairs = int(num_stim * (num_stim - 1) / 2)
        rms_dict = rms_dist_curvature(subj_outdir, domain, CONFIG['model_dimensions'], num_dists=num_stim_pairs)
        bias_df = util.bias_dict(use_all=True)

        files = glob.glob('{}/curvature*likelihood*.csv'.format(subj_outdir))
        df2 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=True)

        best_dict = get_best_model_lls(subj_outdir, domain)
        ################################
        # add an expected bias column
        df2['Bias Estimate'] = None
        df2['Corrected LL Relative to Best Model'] = None
        print(df2)
        for index, row in df2.iterrows():
            dim = row['Dimension']
            domain = row['Domain']
            subject = row['Subject']
            curv_param = row['Lambda-Mu']

            # print(subject, domain, dim, curv_param)
            print(rms_dict)
            print(curv_param)
            print(domain)
            print(dim)
            rms_dist = rms_dict[subject][domain][dim][float(curv_param)]
            bias = util.read_out_median_bias(bias_df, dim, rms_dist)
            df2.loc[[index], ['Bias Estimate']] = bias
            df2.loc[[index], ['Corrected LL Relative to Best Model']] = - best_dict[subject][domain.split('-')[0]].values[
                0] + (bias + row['Log Likelihood'])

        print(df2.head())
        df2['Corrected LLs'] = df2['Log Likelihood'] + df2['Bias Estimate']
        df2['Task'] = PARADIGM
        domain_label = domain.split('_')[0]
        domain_label = domain[:-1] if domain[-1] == '9' else domain
        paradigm_label = PARADIGM_LABELS[PARADIGM]
        sessnum = None
        if paradigm_label == '-gp':
            sessnum = 20
        else:
            sessnum = 10
        df2.to_csv('{}/{}_curve_{}{}_sess01_{}.csv'.format(subj_outdir, domain_label, subject, paradigm_label, sessnum),
                   sep=',', index=False, encoding='utf-8')


