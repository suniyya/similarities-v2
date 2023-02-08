import pandas as pd

from analysis.model_fitting.pairwise_likelihood_analysis import log_likelihood_of_choice_probs

DATA_DIR = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/geometric-modeling/euclidean'


def write_new_lls(subject, domain, sigma, num_pts=37):
    lls = {'Model': [], 'Log Likelihood': [], 'number of points': [], 'Experiment': [], 'Subject': []}
    choice_prob_json = JSON_PATH.format(domain, subject, domain)
    current_ll = None
    for dim in range(1, 8):
        path_pts = '{}/{}/{}/{}_{}_anchored_points_sigma_{}_dim_{}.npy'.format(
            DATA_DIR, domain, subject, subject, domain, sigma, dim)
        current_ll = log_likelihood_of_choice_probs(choice_prob_json, path_pts, sigma)
        lls['Model'].append(str(dim) + 'D')
        lls['Log Likelihood'].append(current_ll[0])
        lls['number of points'].append(num_pts)
        lls['Experiment'].append(domain)
        lls['Subject'].append(subject)
    lls['Model'].append('random')
    lls['Log Likelihood'].append(current_ll[1])
    lls['number of points'].append(num_pts)
    lls['Experiment'].append(domain)
    lls['Subject'].append(subject)
    lls['Model'].append('best')
    lls['Log Likelihood'].append(current_ll[2])
    lls['number of points'].append(num_pts)
    lls['Experiment'].append(domain)
    lls['Subject'].append(subject)
    df = pd.DataFrame(lls)
    df.to_csv('{}/{}/{}_{}_log_likelihoods_sigma_{}.csv'.format(
        DATA_DIR, domain, subject, domain, sigma))


if __name__ == '__main__':
    JSON_PATH = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/experiments/' \
                '{}_exp/subject-data/preprocessed/{}_{}_exp.json'
    SUBJECT1 = input('Subject whose choice probs you want to predict : ')
    DOMAIN1 = input('Domain (of choice probs to predict) : ')
    SUBJECT2 = input('Subject whose model coordinates you are using to predict : ')
    DOMAIN2 = input('Domain (of model coordinates you are using to predict) : ')
    JSON_FILE = JSON_PATH.format(DOMAIN1, SUBJECT1, DOMAIN1)
    DIM = input('Dimensions of coordinates (1, 2, 3, 4, 5, 6, 7?): ')
    if DIM == '':
        DIM = [1, 2, 3, 4, 5, 6, 7]
    else:
        DIM = [DIM]
    SIGMA = float(input('Value of noise parameter during modeling (usually 0.18 or 1): '))

    for d in DIM:
        PATH_NPY = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/geometric-modeling/euclidean' \
                   '/{}/{}/{}_{}_anchored_points_sigma_{}_dim_{}.npy'.format(
                    DOMAIN2, SUBJECT2, SUBJECT2, DOMAIN2, SIGMA, d)
        print(JSON_FILE)
        print(PATH_NPY)
        LL = log_likelihood_of_choice_probs(JSON_FILE, PATH_NPY, SIGMA)
        print(LL)
