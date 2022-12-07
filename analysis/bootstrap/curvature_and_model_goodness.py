"""
Here, given a data set, generate surrogate datasets that are drawn from the original choice probablities.
Then, for each surrogate dataset, essentially calculate LLs for a range of curvatures and (chosen/ adjusted)
sigma values.

This should yield a scatterplot of model LLs by curvature values. We start by doing so for the 2D case and we can
progress to higher dimensions as needed.

The inputs are
- a range of curvature values
- dimensions for which to apply the analysis. Minimum dimension is 2.
- path to json files for individual datasets.
- optionally, can and probably should pass in a legend for when multiple curves are drawn
- number of iterations (surrogates)
- max number of iterations to run the model_fitting pipeline with.
"""

import time
import functools
import ipyparallel as ipp
import pandas as pd

from analysis.model_fitting.model_fitting import decompose_similarity_judgments
from analysis.util import read_in_params


CONFIG, STIMULI, NAMES_TO_ID, ID_TO_NAME = read_in_params()
# disable unneeded params
CONFIG['curvature'] = None
CONFIG['spherical'] = None
CONFIG['hyperbolic'] = None


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


def run(args):
    import copy
    import time
    import logging
    import numpy as np
    import pandas as pd
    from analysis.model_fitting import run_mds_seed as rs

    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)
    num_iterations, judgments, CONFIG, subject, domain, dim, OUTDIR = args
    print(num_iterations)
    degree_curvature = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    degree_curvature_h = [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1]

    def sample_judgments(original_judgments, num_repeats):
        """
        Simulate judgments based on empirical choice probabilities
        :param original_judgments:
        :param num_repeats:
        :return:
        """
        sample = {}
        for trial, count in original_judgments.items():
            sample[trial] = 0
            prob = float(count) / num_repeats
            for j in range(num_repeats):
                random_draw = np.random.uniform(0, 1)
                if random_draw < prob:
                    sample[trial] += 1
        return sample

    def fit_model(similarity_judgments, curvature, params, dim, start_points):
        params_copy = copy.deepcopy(params)
        num_judgments = len(similarity_judgments)
        noise = np.sqrt(params['sigmas']['compare']**2 + params['sigmas']['dist']**2)
        params_copy['n_dim'] = dim  # ensure correct model is tested
        if curvature == 0:
            # fit Euclidean model
            x, ll, fmin_costs = rs.points_of_best_fit(similarity_judgments, params_copy, start_points)
            ll = -1 * ll / (params['num_repeats'] * num_judgments)
        elif curvature > 0:
            # fit spherical model
            curvature_val = curvature
            params_copy['curvature'] = curvature_val
            x, ll, fmin_costs = rs.spherical_points_of_best_fit(similarity_judgments, params_copy, start_points)
            ll = -1 * ll / (params['num_repeats'] * num_judgments)
        else:
            # fit hyperbolic model
            curvature_val = -1 * curvature
            params_copy['curvature'] = curvature_val
            x, ll, fmin_costs = rs.hyperbolic_points_of_best_fit(similarity_judgments, params_copy, start_points)
            ll = -1 * ll / (params['num_repeats'] * num_judgments)
        return ll, curvature, noise, x

    def produce_surrogate_data(judgments_orig, params, batch_size=1):
        """
        Return a collection of surrogate judgments based on real data
        @param judgments_orig:  real data
        @param batch_size: size of surrogate datasets to make in a go
        @param params: read in from Config file
        @return:
        """
        batch = []
        for i in range(batch_size):
            new_judgments = sample_judgments(judgments_orig, params['num_repeats'])
            batch.append(new_judgments)
        return batch

    surrogate_datasets = produce_surrogate_data(judgments, CONFIG, 1)
    results = {'LL': [], 'Lambda-Mu': [], 'Curvature of Space': [], 'Sigma': [], 'Dimension': [], 'Subject': [],
               'Domain': []}
    for data in surrogate_datasets:
        for _c in range(len(degree_curvature)):
            c = degree_curvature[_c]
            if _c == 0:
                start = None
            log_likelihood, curvature_val, sigma, coords = fit_model(data, c, CONFIG, dim, start)
            LOG.info("LL: {}".format(log_likelihood))
            LOG.info("Sph fit points: ")
            print(coords)
            outfilename = '{}/{}_{}_spherical_model_coords_sigma_{}_dim_{}_mu_{}'.format(
                OUTDIR,
                subject, domain,
                str(CONFIG['sigma']['compare'] + CONFIG['sigma']['dist']),
                dim,
                curvature_val
            )
            np.save(outfilename, coords)
            start = coords
            # write to pandas file
            results['Lambda-Mu'].append(curvature_val)
            results['Curvature of Space'].append(curvature_val ** 2)
            results['LL'].append(log_likelihood)
            results['Sigma'].append(sigma)
            results['Dimension'].append(dim)
            results['Subject'].append(subject)
            results['Domain'].append(domain)
        for _c in range(len(degree_curvature_h)):
            c = degree_curvature_h[_c]
            if _c == 0:
                start = None
            log_likelihood, curvature_val, sigma, coords = fit_model(data, c, CONFIG, dim, start)
            LOG.info("LL: {}".format(log_likelihood))
            LOG.info("Hyp fit points: ")
            print(coords)
            outfilename = '{}/{}_{}_hyperbolic_model_coords_sigma_{}_dim_{}_lambda_{}'.format(
                OUTDIR,
                subject, domain,
                str(CONFIG['sigma']['compare'] + CONFIG['sigma']['dist']),
                dim,
                curvature_val
            )
            np.save(outfilename, coords)
            start = coords
            # write to pandas file
            results['Lambda-Mu'].append(curvature_val)
            results['Curvature of Space'].append(curvature_val ** 2)
            results['LL'].append(log_likelihood)
            results['Sigma'].append(sigma)
            results['Dimension'].append(dim)
            results['Subject'].append(subject)
            results['Domain'].append(domain)
    # # write df
    df = pd.DataFrame(results)
    return df


@timer
def run_in_parallel(operation, n_iter, workers):
    return workers.map_sync(operation, n_iter)


# NOTE: ########################################
# Values of lambda and mu are hard-coded inside run function
# curvature is lambda^2 and mu^2 = 1/R^2 according to definition of Gaussian curvature...

print(CONFIG)

domain = input('Domain: ')
SUBJECTS = input('Subjects (separated by spaces): ').split(' ')
print(SUBJECTS)
proceed = input('If subjects correct, press "y" to proceed')
if proceed != 'y':
    raise IOError

DIM = int(input('Number of Dimensions: '))
OUTDIR = input('Output directory for LLs and coordinates: ')
NUM_SURROGATES = int(input('Number of surrogate data sets to make and fit: '))

client_ids = ipp.Client()
pool = client_ids[:]

for subject in SUBJECTS:
    print(subject)
    INPUT_DATA = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/' \
                 'experiments/{}_exp/subject-data/preprocessed/{}_{}_exp.json'.format(domain, subject, domain)
    judgments = decompose_similarity_judgments(INPUT_DATA, NAMES_TO_ID)

    ARGS = []

    for i in range(NUM_SURROGATES):
        ARGS.append((i, judgments, CONFIG, subject, domain, DIM, OUTDIR))

    result = run_in_parallel(run, ARGS, pool)

    total_df = pd.concat(result)
    print(total_df)
    total_df.to_csv('{}/curvature_and_LL_{}-{}-{}_combined.csv'.format(OUTDIR, subject, domain, DIM))
