import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.util import stimulus_names
from sklearn.decomposition import PCA

from matplotlib.cbook import get_sample_data
import matplotlib.ticker as ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def var_explained(filename):
    """
    Run PCA.
    Stretch axes so as to make the variance along each axis the same.
    Do so by dividing the values of the coordinate by the standard deviation of values along that axis.
    This way points along each axis will have unit variance. This does not affect the radial
    distribution of points, only their distance from the origin in different directions.
    """
    points = np.load(filename)
    n_components = points.shape[1]
    means = []
    varis = []
    message = "Dim,Mean,Variance,Fraction of Total Variance\n"
    pca = PCA(n_components=n_components)
    # obtain the 5 PC directions and project data onto that space
    temp = pca.fit_transform(points)
    # normalize each axis by its standard deviation to make sd across each axis the same
    for i in range(n_components):
        varis.append((np.std(temp[:, i]))**2)
        means.append(np.mean(temp[:, i]))
    sum_var = sum(varis)
    for i in range(n_components):
        message += "{},{},{},{}\n".format(i+1, np.round(means[i], 5),
                                                                 np.round(varis[i], 5),
                                                                 np.round((varis[i] / sum_var), 5))
    print(message)
    return means, varis


def do_pca(points, dim=5):
    """
    Run PCA.
    Stretch axes so as to make the variance along each axis the same.
    Do so by dividing the values of the coordinate by the standard deviation of values along that axis.
    This way points along each axis will have unit variance. This does not affect the radial
    distribution of points, only their distance from the origin in different directions.
    """
    n_components = dim
    pca = PCA(n_components=n_components)
    # obtain the 5 PC directions and project data onto that space
    temp = pca.fit_transform(points)
    # do not normalize each axis by its standard deviation to make sd across each axis the same
    points = temp
    return points


def scatterplots_2d_annotated(subject_name, subject_exp_data, condition, pc1=1, pc2=2, outdir="./experiments/image_exp/subject-data"):
    sns.set_style('darkgrid')
    stimuli = stimulus_names(condition)
    fig, ax = plt.subplots()
    plt.scatter(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1], c="#31505A", marker='.')
    # add labels to points
    label_idx = 0
    for x, y in zip(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1]):
        plt.annotate(stimuli[label_idx],  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 1.5),  # distance from text to points (x,y)
                     size=9.5,
                     ha='center')  # horizontal alignment can be left, right or center
        label_idx += 1
    plt.xlabel('Principal Component {}'.format(pc1))
    plt.ylabel('Principal Component {}'.format(pc2))
    plt.title(subject_name)
    plt.axis('square')
    plt.savefig('{}/{}_{}_scatterplot_pc_{}{}.png'.format(outdir, subject_name, condition, pc1, pc2),
                bbox_inches='tight', pad_inches=0.25)
    plt.show()


def scatterplots_2d_image_annotated(subject_name, subject_exp_data, imagesource, pc1=1, pc2=2, outdir="./experiments/image_exp/subject-data"):
    sns.set_style('darkgrid')
    stimuli = stimulus_names(imagesource)
    if 'face' in imagesource:
        domain = 'face'
    else:
        domain = 'texture'
    fig, ax = plt.subplots()
    plt.scatter(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1], c="#31505A", marker='.')
    # add labels to points
    label_idx = 0
    for x, y in zip(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1]):
        image_path = glob.glob('C:/Users/jdvicto/Documents/similarities/{}-exp-materials/{}/{}*'
                               .format(domain, imagesource, stimuli[label_idx]))[0]
        with get_sample_data(image_path) as file:
            image = plt.imread(file)
        imagebox = OffsetImage(image, zoom=1/6)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (x, y),
                            xycoords='data',
                            boxcoords='offset points',
                            pad=0.1)

        ax.add_artist(ab)
        label_idx += 1
    plt.xlabel('Principal Component {}'.format(pc1))
    plt.ylabel('Principal Component {}'.format(pc2))
    plt.title(subject_name)
    plt.axis('square')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig('{}/{}_{}_scatterplot_pc_{}{}_im.png'.format(outdir, subject_name, imagesource, pc1, pc2),
                bbox_inches='tight', pad_inches=0.25)
    plt.show()


if __name__ == '__main__':
    PATH_TO_NPY_FILE = input("Path to npy file containing 5D coordinates "
                             "(.\experiments\image_exp\subject-data/tvpm3pt9_data/BL_tvpm3pt9_anchored_points_sigma_1.0_dim_5.npy) : ")
    NAME = input("Subject name or ID (e.g., S7): ")
    CONDITION = input("Condition/ experiment name (e.g., bgca3pt9): ")

    data0 = np.load(PATH_TO_NPY_FILE)
    DIM = PATH_TO_NPY_FILE.split('.npy')[0].split('dim_')[-1]
    data = do_pca(data0, int(DIM))
    scatterplots_2d_image_annotated(NAME, data, CONDITION)
    scatterplots_2d_image_annotated(NAME, data, CONDITION, 1, 3)
    scatterplots_2d_image_annotated(NAME, data, CONDITION, 1, 4)
    scatterplots_2d_image_annotated(NAME, data, CONDITION, 2, 3)
    scatterplots_2d_image_annotated(NAME, data, CONDITION, 2, 4)
    scatterplots_2d_image_annotated(NAME, data, CONDITION, 3, 4)
    scatterplots_2d_annotated(NAME, data, CONDITION)
    scatterplots_2d_annotated(NAME, data, CONDITION, 1, 3)
    scatterplots_2d_annotated(NAME, data, CONDITION, 2, 3)
    scatterplots_2d_annotated(NAME, data, CONDITION, 1, 4)
    scatterplots_2d_annotated(NAME, data, CONDITION, 2, 4)
    scatterplots_2d_annotated(NAME, data, CONDITION, 3, 4)

    # var_explained(PATH_TO_NPY_FILE)