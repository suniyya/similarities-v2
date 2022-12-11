import numpy as np
import hyperbolic as hyp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    DIRECTORY = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1' \
                '/geometric-modeling/noneuclidean-models-curvature-1to1-sigma-1'
    DOMAIN = input('Domain: ')
    SUBJECT = input('Subject: ')

    LAMBDA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for hyp_param in LAMBDA:
        FILE_NPY = '{}/{}/{}/{}_{}_hyperbolic_model_coords_sigma_1.0_dim_2_lambda_-{}.npy'.format(
            DIRECTORY, DOMAIN, SUBJECT, SUBJECT, DOMAIN, hyp_param
        )
        # points = np.load(FILE_NPY).T
        # indented blockk is to test. trinagle maps onto somethng... else.
            # X1s = [0.05*i for i in range(100)] + [-0.05*i for i in range(100)]
            # Y1s = [-1 for _ in range(200)]
            # X2s = [50 for _ in range(100)]
            # Y2s = [[0.05*i for i in range(50)] + [-0.05*i for i in range(50)]]
            # Xs = X1s+X2s
            # Ys = Y1s+Y2s
            # points = np.array([Xs, Ys])
        loid_proj = hyp.loid_map(points, hyp_param)
        Z = hyp.loid_to_poincare_map(loid_proj)
        print('Coordinates (d by n)')
        print(Z.shape)
        disc = plt.Circle((0, 0), 1, fill=False)
        f = plt.figure()
        plt.title(SUBJECT + ' ' + DOMAIN + ' lambda=' + str(hyp_param))
        plt.plot(Z[0, :], Z[1, :], 'b.')
        ax = plt.gca()
        ax.add_patch(disc)
        plt.axis('square')
        plt.show()
