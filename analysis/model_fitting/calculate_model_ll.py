from analysis.model_fitting.pairwise_likelihood_analysis import log_likelihood_of_choice_probs

if __name__ == '__main__':
    JSON_PATH = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/experiments/' \
                '{}_exp/subject-data/preprocessed/{}_{}_exp.json'
    SUBJECT1 = input('Subject whose choice probs you want to predict : ')
    DOMAIN1 = input('Domain (of choice probs to predict) : ')
    SUBJECT2 = input('Subject whose model coordinates you are using to predict : ')
    DOMAIN2 = input('Domain (of model coordinates you are using to predict) : ')
    JSON_FILE = JSON_PATH.format(DOMAIN1, SUBJECT1, DOMAIN1)
    DIM = input('Dimensions of coordinates (1, 2, 3, 4, 5, 6, 7?): ')
    SIGMA = float(input('Value of noise parameter during modeling (usuall 0.18 or 1): '))
    PATH_NPY = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/geometric-modeling/euclidean' \
               '/{}/{}/{}_{}_anchored_points_sigma_{}_dim_{}.npy'.format(
                DOMAIN2, SUBJECT2, SUBJECT2, DOMAIN2, SIGMA, DIM)
    print(JSON_FILE)
    print(PATH_NPY)
    LL = log_likelihood_of_choice_probs(JSON_FILE, PATH_NPY, SIGMA)
    print(LL)
