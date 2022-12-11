"""
Quantify the differences between configurations of points across domains (for a given subject),
or within a domain, across subjects.

Parameters:
@ num_stimuli:  number of points/ stimuli
@ num_dim: dimension of the space from which to draw points
@ magnitude: magnitude of each of the points
"""

import glob
import numpy as np
import seaborn as sns
import analysis.util as utils
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

# from analysis.model_fitting.rough_common_models import subjects

MODELS_DIRECTORY = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/geometric-modeling/euclidean'
SUBJECTS = ['MC', 'BL', 'EFV', 'SJ', 'SAW', 'YCL', 'SA']
SUBJECT_IDS = {'MC': 'S1', 'BL': 'S2', 'EFV': 'S3', 'SJ': 'S4', 'SAW': 'S5',
               'YCL': 'S7', 'SA': 'S8', 'NK': 'S6', 'JF': 'S9'}
DOMAINS = ['texture', 'intermediate_texture', 'intermediate_object', 'image', 'word']
DOMAIN_LABELS = {'texture': 'texture', 'intermediate_texture': 'texture-like',
                 'intermediate_object': 'image-like', 'image': 'image', 'word': 'word'}
DOMAIN_NAMES = ['texture', 'texture-like', 'image-like', 'image', 'word']
DIMENSIONS = [2, 3, 7]
NUM_DOMAINS = len(DOMAINS)
NUM_SUBJECTS = len(SUBJECTS)


def proc_distances_by_subject():
    for subject in SUBJECTS:
        proc_dists_sub = np.zeros((NUM_DOMAINS, NUM_DOMAINS))
        f, axes = plt.subplots(1, len(DIMENSIONS)+1,
                               gridspec_kw={'width_ratios': [1, 1, 1, 0.08]}, figsize=(10, 6))
        axes[0].get_shared_y_axes().join(axes[1], axes[2])
        is_cbar = [False]*len(DIMENSIONS)
        is_cbar[-1] = True
        for d in range(len(DIMENSIONS)):
            dim = DIMENSIONS[d]
            domain_files = []
            for domain in DOMAINS:
                domain_files = domain_files + glob.glob('{}/{}/{}/{}_*dim_{}.npy'.format(
                    MODELS_DIRECTORY, domain, subject, subject, dim))
            for i in range(len(DOMAINS)):
                for j in range(i):
                    points_i = np.load(domain_files[i])
                    points_j = np.load(domain_files[j])
                    m1, m2, difference = procrustes(points_i, points_j)
                    proc_dists_sub[i, j] = difference

            # sns.color_palette("vlag", as_cmap=True)
            mask = np.zeros_like(proc_dists_sub)
            mask[np.triu_indices_from(proc_dists_sub)] = True
            with sns.axes_style("white"):
                g = sns.heatmap(np.array(proc_dists_sub), mask=mask, square=True, vmax=1, ax=axes[d], cbar=is_cbar[d],
                                cbar_kws={"shrink": 0.4}, cbar_ax=axes[-1])
                g.set_title(subject + '- Dim: ' + str(dim), fontsize=10)
                if d == 0:
                    g.set_yticklabels(DOMAIN_NAMES, rotation=360)
                else:
                    g.set_yticklabels([None]*NUM_DOMAINS)
                g.set_xticklabels([DOMAIN_NAMES[0], None, None, DOMAIN_NAMES[-2], None], rotation=90)
        plt.show()


def proc_dist_by_domain():
    for domain in DOMAINS:
        if domain in ['image', 'intermediate_object']:
            subjects = SUBJECTS + ['NK']
        else:
            subjects = SUBJECTS + ['JF']
        proc_dists_domain = np.zeros((NUM_SUBJECTS+1, NUM_SUBJECTS+1))
        f, axes = plt.subplots(1, len(DIMENSIONS) + 1,
                               gridspec_kw={'width_ratios': [1, 1, 1, 0.08]}, figsize=(10, 6))
        axes[0].get_shared_y_axes().join(axes[1], axes[2])
        is_cbar = [False] * len(DIMENSIONS)
        is_cbar[-1] = True
        for d in range(len(DIMENSIONS)):
            dim = DIMENSIONS[d]
            subject_files = []
            for subject in subjects:
                subject_files = subject_files + glob.glob('{}/{}/{}/{}_*dim_{}.npy'.format(
                    MODELS_DIRECTORY, domain, subject, subject, dim))
            for i in range(NUM_SUBJECTS+1):
                for j in range(i):
                    points_i = np.load(subject_files[i])
                    points_j = np.load(subject_files[j])
                    m1, m2, difference = procrustes(points_i, points_j)
                    proc_dists_domain[i, j] = difference

            # sns.color_palette("vlag", as_cmap=True)
            mask = np.zeros_like(proc_dists_domain)
            mask[np.triu_indices_from(proc_dists_domain)] = True
            with sns.axes_style("white"):
                g = sns.heatmap(np.array(proc_dists_domain), mask=mask, square=True, vmax=1, ax=axes[d], cbar=is_cbar[d],
                                cbar_kws={"shrink": 0.4}, cbar_ax=axes[-1])
                g.set_title(domain + '- Dim: ' + str(dim), fontsize=10)
                if d == 0:
                    g.set_yticklabels([SUBJECT_IDS[k] for k in subjects], rotation=360)
                else:
                    g.set_yticklabels([None] * (NUM_SUBJECTS+1))
                g.set_xticklabels([SUBJECT_IDS[k] for k in subjects], rotation=90)
        plt.show()


def errors_after_proc_alignment_2d(set1, set2, label1=None, label2=None):
    """
    Assumes points are 2 dimensional.
    After two sets of points have been aligned by procrustes alignment, display the aligned points,
    with lines between corresponding points to show errors.
    @param set1: matrix n by 2
    @param set2: matrix n by 2
    @return: None - plot scatterplot
    """
    plt.plot(set1[:, 0], set1[:, 1], 'c.')
    plt.plot(set2[:, 0], set2[:, 1], 'm.')
    for _i in range(len(set1)):
        plt.arrow(set1[_i, 0], set1[_i, 1], set2[_i, 0]-set1[_i, 0], set2[_i, 1]-set1[_i, 1],
                  length_includes_head=True, head_width=0.01, color=[0, 0, 0], alpha=0.5)
    if label1 is not None:
        plt.legend([label1, label2], bbox_to_anchor=(1.04, 1), loc="upper center")


def proc_errors_transition_plot(subject, dim=2):
    """Plot original and next domain points after Procrustes alignment.
    Assumes you are passing in the raw 2D model coordinates. No PCA, no higher dimensions. """
    if dim != 2:
        raise NotImplementedError('Only done for 2 dimensions atm')
    subject_files = []
    stimuli = utils.stimulus_names()
    for domain in DOMAINS:
        subject_files = subject_files + glob.glob('{}/{}/{}/{}_*dim_{}.npy'.format(
            MODELS_DIRECTORY, domain, subject, subject, dim))
    plt.figure(figsize=(20, 5), dpi=80)
    for d in range(len(subject_files)-1):
        m1, m2, diff = procrustes(np.load(subject_files[d]), np.load(subject_files[d+1]))
        # m1, m2, diff = procrustes(np.load(subject_files[d+1]), np.load(subject_files[d]))
        plt.subplot(1, len(DOMAINS)-1, d+1)
        errors_after_proc_alignment_2d(m1, m2)
        # add labels to points
        label_idx = 0
        for x, y in zip(m1[:, 0], m1[:, 1]):
            plt.annotate(stimuli[label_idx],  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 1.5),  # distance from text to points (x,y)
                         size=8,
                         ha='center')  # horizontal alignment can be left, right or center
            label_idx += 1
        print(diff)
        plt.ylim([-0.3, 0.4])
        plt.xlim([-0.3, 0.4])
        plt.gca().set_title(DOMAIN_LABELS[DOMAINS[d]] + '->' + DOMAIN_LABELS[DOMAINS[d+1]], fontsize=10)
        plt.axis('equal')
    plt.suptitle(subject)
    plt.show()


proc_errors_transition_plot('YCL', 2)

# if dim == 3:
#     # 3D scatterplots
#     plt.title(subject)
#     fig = plt.figure(figsize=(12, 12))
#     plotNum = 1
#     domain_colors = {'texture': 'red', 'intermediate_texture': 'yellow',
#                      'intermediate_object': 'green', 'image': 'blue', 'word': 'magenta'}
#     for d in range(len(DOMAINS) - 1):
#         # =============
#         # First Subplot
#         # =============
#         # set up the axes for the first plot
#         ax = fig.add_subplot(4, 2, plotNum, projection='3d')
#         # coordinates of chains A & C (before alignment)
#         points_i = np.load(domain_files[d])
#         points_j = np.load(domain_files[d + 1])
#         m1, m2, difference = procrustes(points_i, points_j)
#         ax.scatter(xs=points_i[:, 0], ys=points_i[:, 1], zs=points_i[:, 2],
#                    marker="o", color=domain_colors[DOMAINS[d]], s=11, label=DOMAINS[d])
#         ax.scatter(xs=points_j[:, 0], ys=points_j[:, 1], zs=points_j[:, 2],
#                    marker="o", color=domain_colors[DOMAINS[d + 1]], s=11, label=DOMAINS[d + 1])
#         ax.set_xlabel("X", fontsize=8)
#         ax.set_ylabel("Y", fontsize=8)
#         ax.set_zlabel("Z", fontsize=8)
#         ax.legend(fontsize=5, loc="best")
#         plotNum += 1
#         # ==============
#         # Second Subplot
#         # ==============
#         # set up the axes for the second plot
#         ax = fig.add_subplot(4, 2, plotNum, projection='3d')
#         # coordinates of chains A & C after translation and rotation
#         ax.scatter(xs=m1[:, 0], ys=m1[:, 1], zs=m1[:, 2],
#                    marker="o", color=domain_colors[DOMAINS[d]], s=11, label=DOMAINS[d])
#         ax.scatter(xs=m2[:, 0], ys=m2[:, 1], zs=m2[:, 2],
#                    marker="o", color=domain_colors[DOMAINS[d + 1]], s=11, label=DOMAINS[d + 1])
#         ax.set_xlabel("X", fontsize=8)
#         ax.set_ylabel("Y", fontsize=8)
#         ax.set_zlabel("Z", fontsize=8)
#         ax.legend(fontsize=5, loc="best")
#         plotNum += 1
#     plt.show()