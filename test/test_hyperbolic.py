import unittest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import analysis.geometry.euclidean as model
from analysis.geometry.hyperbolic import loid_map, hyperbolic_distances, spherical_distances, sphere_map, \
    sphere_map_deprecated, spherical_distances_deprecated


class CurvatureHyperbolic(unittest.TestCase):
    def test_origin_correctly_mapped_to_hyperboloid_orgin(self):
        X = np.array([[0], [0]])
        Y = loid_map(X, 1)
        self.assertEqual(len(Y), len(X) + 1)
        self.assertEqual(Y[0], 1)  # add assertion here
        self.assertEqual(Y[1], 0)
        self.assertEqual(Y[2], 0)

    def test_visually_if_square_projected_correctly(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        square_euclidean = np.array([[-1, -1, -1, -1, -1, -1, -1, -1,
                                      -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1,
                                      0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75],
                                     [1, 0.75, 0.5, 0, -0.25, -0.5, -0.75, -1,
                                      -1, -1, -1, -1, -1, -1, -1, -1,
                                      -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
                                      1, 1, 1, 1, 1, 1, 1]])
        plt.plot(square_euclidean[0, :], square_euclidean[1, :], 'o')
        for _i in [0.2, 1, 2]:
            for _j in [1, 1.5]:
                temp = _j * square_euclidean
                Y = loid_map(temp, _i)
                plt.plot(temp[0, :], temp[1, :], 'bo')
                ax.scatter(Y[1, :], Y[2, :], Y[0, :], color=[0.5, 0.5, 0.7])
            plt.show()
        self.assertTrue(True)

    def test_visually_if_random_points_projected_to_sphere_of_correct_radius(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        rand_dots = 1*np.array([np.random.standard_normal(100), np.random.standard_normal(100)])
        # rand_dots = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1,
        #                        -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
        #                        1, 1, 1, 1, 1, 1, 1, 1,
        #                        0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75],
        #                       [0, 1, 0.75, 0.5, 0, -0.25, -0.5, -0.75, -1,
        #                        -1, -1, -1, -1, -1, -1, -1, -1,
        #                        -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
        #                        1, 1, 1, 1, 1, 1, 1]])
        rand_dots = rand_dots
        plt.plot(rand_dots[0, :], rand_dots[1, :], 'bo')
        for _i in [1 / 2]:  # [1/0.2, 1, 2
            # Y = sphere_map_deprecated(rand_dots, _i)
            # ax.scatter(Y[1, :], Y[2, :], Y[0, :], color=[0.5, 0.5, 0.7])
            Y2 = sphere_map(rand_dots, _i)
            ax.scatter(Y2[1, :], Y2[2, :], Y2[0, :], color=[0.7, 0.5, 0.5])
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.show()
        # max_range = np.array([Y[1, :].max() - Y[1, :].min(), Y[2, :].max() - Y[2, :].min(), Y[0, :].max() - Y[0, :].min()]).max() / 2.0
        # mid_x = (Y[1, :].max() + Y[1, :].min()) * 0.5
        # mid_y = (Y[2, :].max() + Y[2, :].min()) * 0.5
        # mid_z = (Y[0, :].max() + Y[0, :].min()) * 0.5
        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # ax.set_xlim(-15, 15)
        # ax.set_ylim(-15, 15)
        # ax.set_zlim(-15, 15)
        self.assertTrue(True)

    def test_if_hyperbolic_distance_equals_euclidean_for_near_zero_curvature(self):
        square_euclidean = np.array([[-1, -1, -1, -1, -1, -0.5, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, -0.5],
                                     [1, 0.5, 0, -0.5, -1, -1, -1, -1, -1, -0.5, 0, 0.5, 1, 1, 1, 1]])
        X = squareform(pdist(square_euclidean.T))
        Y = hyperbolic_distances(loid_map(square_euclidean, 0.01), 0.01)
        diffs = 0
        total = 0
        for i in range(Y.shape[0]):
            for j in range(i):
                diffs += np.abs(Y[i, j] - X[i, j])
                total += 1
        self.assertLessEqual(diffs / total, 0.005)

    def test_if_hyperbolic_distance_equals_euclidean_for_points_near_orgin(self):
        random_euclidean = np.random.rand(2, 10)
        random_euclidean = 0.02 * random_euclidean
        center = np.mean(random_euclidean, 1)
        center = center.reshape((2, 1)) * np.ones((1, 10))
        random_euclidean = random_euclidean - center
        X = squareform(pdist(random_euclidean.T))
        lambda_val = [0.5, 1, 2]
        for val in lambda_val:
            Y = hyperbolic_distances(loid_map(random_euclidean, val), val)
            ratios = []
            for i in range(Y.shape[0]):
                for j in range(i):
                    ratios.append(Y[i, j] / X[i, j])
            print(val)
            self.assertAlmostEqual(np.mean(ratios), 1, delta=0.03)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.plot(random_euclidean[0, :], random_euclidean[1, :], 'o')
        for _i in [0.5, 1, 2, 3]:
            Y = loid_map(random_euclidean, _i)
            if Y is not None:
                ax.scatter(Y[1, :], Y[2, :], Y[0, :])
        ax.set_xlim(-2, 1)
        ax.set_ylim(-2, 1)
        # plt.show()

    def test_if_spherical_distance_equals_euclidean_for_points_near_orgin(self):
        random_euclidean = np.random.rand(2, 50)
        random_euclidean = 0.02 * random_euclidean
        center = np.mean(random_euclidean, 1)
        center = center.reshape((2, 1)) * np.ones((1, 50))
        random_euclidean = random_euclidean - center
        X = squareform(pdist(random_euclidean.T))
        radii = [500]
        for radius in radii:
            Y = spherical_distances(sphere_map(random_euclidean, radius), radius)
            ratios = []
            for i in range(Y.shape[0]):
                for j in range(i):
                    ratios.append(Y[i, j] / X[i, j])
            self.assertAlmostEqual(np.mean(ratios), 1, delta=0.03)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.plot(random_euclidean[0, :], random_euclidean[1, :], 'o')
        for _i in [0.5, 1, 2, 3]:
            Y = sphere_map(random_euclidean, _i)
            if Y is not None:
                ax.scatter(Y[1, :], Y[2, :], Y[0, :])
        ax.set_xlim(-2, 1)
        ax.set_ylim(-2, 1)
        plt.show()

    def test_inner_product_is_minus_1_for_points_on_hyperboloid(self):
        points2D = np.random.rand(10, 10)
        pointsH = loid_map(points2D.T, 1)
        H = np.eye(pointsH.shape[0])
        H[0, 0] = -1
        inner_product = np.round(pointsH.T @ H @ pointsH, 5)
        # inner prod of vecs with  themselves should be -1
        for i in range(len(inner_product)):
            self.assertEqual(inner_product[i, i], -1)
        self.assertTrue(np.all(inner_product <= -1))

    def test_old_spherical_map_and_dist_give_same_answer_as_old_method(self):
        # rand_dots = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1,
        #                        -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
        #                        1, 1, 1, 1, 1, 1, 1, 1,
        #                        0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75],
        #                       [0, 1, 0.75, 0.5, 0, -0.25, -0.5, -0.75, -1,
        #                        -1, -1, -1, -1, -1, -1, -1, -1,
        #                        -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
        #                        1, 1, 1, 1, 1, 1, 1]])
        space = model.EuclideanSpace(2)
        rand_dots = space.get_samples(10, 1, 'gaussian')
        rand_dots = rand_dots.T
        radii = [0.1, 0.5, 1, 5]
        for radius in radii:
            print('curvature', 1 / radius)
            Y = sphere_map_deprecated(rand_dots, radius)
            Y2 = sphere_map(rand_dots, radius)
            D_new = spherical_distances(Y2, radius)
            D_old = spherical_distances_deprecated(Y, radius)
            print(np.mean(np.mean(D_old)))
            print(np.mean(np.mean(D_new)))
            self.assertLess(sum(sum(D_new - D_old)), 0.00015)

    def test_hyperbolic_map_and_dist_as_curvature_increases(self):
        # rand_dots = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1,
        #                        -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
        #                        1, 1, 1, 1, 1, 1, 1, 1,
        #                        0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75],
        #                       [0, 1, 0.75, 0.5, 0, -0.25, -0.5, -0.75, -1,
        #                        -1, -1, -1, -1, -1, -1, -1, -1,
        #                        -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
        #                        1, 1, 1, 1, 1, 1, 1]])
        space = model.EuclideanSpace(2)
        rand_dots = space.get_samples(1000, 10, 'uniform')
        rand_dots = rand_dots.T
        curvs = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4]
        for curv in curvs:
            print('curvature', curv)
            Y = loid_map(rand_dots, curv)
            # Y2 = sphere_map(rand_dots, radius)
            D_new = hyperbolic_distances(Y, curv)
            print('Mean dist:', np.mean(np.mean(D_new)), '.  Max dist:', np.max(np.max(D_new)))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
