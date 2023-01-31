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
from scipy.io import loadmat
import seaborn as sns
import analysis.util as utils
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

# from analysis.model_fitting.rough_common_models import subjects

MODELS_DIRECTORY = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/geometric-modeling/euclidean'
SUBJECTS = ['MC', 'SJ', 'SAW', 'YCL', 'AJ', 'SN', 'ZK']  # , 'BL', 'SA', 'EFV']
SUBJECT_IDS = {'MC': 'S1', 'BL': 'S2', 'EFV': 'S3', 'SJ': 'S4', 'SAW': 'S5',
               'YCL': 'S7', 'SA': 'S8', 'NK': 'S6', 'JF': 'S9', 'AJ': 'S10', 'SN': 'S11', 'ZK': 'S12',
               'consensus': 'consensus'}
DOMAINS = ['texture', 'intermediate_texture', 'intermediate_object', 'image', 'word']
DOMAIN_LABELS = {'texture': 'texture', 'intermediate_texture': 'texture-like',
                 'intermediate_object': 'image-like', 'image': 'image', 'word': 'word'}
DOMAIN_NAMES = ['texture', 'texture-like', 'image-like', 'image', 'word']
DIMENSIONS = [3, 5, 7]
NUM_DOMAINS = len(DOMAINS)


def get_consensus_pts(domain):
    path = '{}/consensus/consensus_{}.mat'.format(MODELS_DIRECTORY, domain)
    domain_keys = {'texture': 'consensus_tex', 'intermediate_texture': 'consensus_texlike',
                   'intermediate_object': 'consensus_imlike', 'image': 'consensus_im', 'word': 'consensus_word'}
    points = loadmat(path)
    return points[domain_keys[domain]]


def proc_distances_consensus():
    proc_dists_sub = np.zeros((NUM_DOMAINS, NUM_DOMAINS))
    f, axes = plt.subplots(1, len(DIMENSIONS) + 1,
                           figsize=(10, 6))  # gridspec_kw={'width_ratios': [1, 1, 1, 0.08]}, )
    axes[0].get_shared_y_axes().join(axes[1], axes[2])
    is_cbar = [False] * len(DIMENSIONS)
    is_cbar[-1] = True
    for d in range(len(DIMENSIONS)):
        dim = DIMENSIONS[d]
        domain_files = []
        for domain in DOMAINS:
            domain_files.append(get_consensus_pts(domain))
        for i in range(len(DOMAINS)):
            for j in range(i):
                m1, m2, difference = procrustes(domain_files[i], domain_files[j])
                proc_dists_sub[i, j] = difference

        # sns.color_palette("vlag", as_cmap=True)
        mask = np.zeros_like(proc_dists_sub)
        mask[np.triu_indices_from(proc_dists_sub)] = True
        with sns.axes_style("white"):
            g = sns.heatmap(np.array(proc_dists_sub), mask=mask, square=True, vmax=1, ax=axes[d], cbar=is_cbar[d],
                            cbar_kws={"shrink": 0.4}, cbar_ax=axes[-1], cmap='Purples')
            g.set_title('Consensus' + '- Dim: ' + str(dim), fontsize=10)
            if d == 0:
                g.set_yticklabels(DOMAIN_NAMES, rotation=360)
            else:
                g.set_yticklabels([None] * NUM_DOMAINS)
            g.set_xticklabels([DOMAIN_NAMES[0], None, None, DOMAIN_NAMES[-2], None], rotation=90)
    plt.show()


def proc_distances_by_subject():
    for subject in SUBJECTS:
        proc_dists_sub = np.zeros((NUM_DOMAINS, NUM_DOMAINS))
        f, axes = plt.subplots(1, len(DIMENSIONS) + 1,
                               figsize=(10, 6))  # gridspec_kw={'width_ratios': [1, 1, 1, 0.08]}, )
        axes[0].get_shared_y_axes().join(axes[1], axes[2])
        is_cbar = [False] * len(DIMENSIONS)
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
                                cbar_kws={"shrink": 0.4}, cbar_ax=axes[-1], cmap='Purples')
                g.set_title(subject + '- Dim: ' + str(dim), fontsize=10)
                if d == 0:
                    g.set_yticklabels(DOMAIN_NAMES, rotation=360)
                else:
                    g.set_yticklabels([None] * NUM_DOMAINS)
                g.set_xticklabels([DOMAIN_NAMES[0], None, None, DOMAIN_NAMES[-2], None], rotation=90)
        plt.show()


def proc_dist_by_domain(dim, include_consensus=False):
    if include_consensus:
        subjects = SUBJECTS + ['consensus']
    else:
        subjects = SUBJECTS
    num_subjects = len(subjects)
    proc_dists_domain = np.zeros((num_subjects, num_subjects))
    for domain in DOMAINS:
        f, axes = plt.subplots(1, 1,
                               figsize=(6, 3.5))  # gridspec_kw={'width_ratios': [1, 1, 1, 0.08]},
        subject_points = {}
        for subject in subjects:
            if subject == 'consensus':
                subject_points['consensus'] = get_consensus_pts(domain)
            else:
                subject_files = glob.glob('{}/{}/{}/{}_*dim_{}.npy'.format(
                    MODELS_DIRECTORY, domain, subject, subject, dim))
                if len(subject_files) > 1:
                    raise ValueError('More than 1 npy file...')
                subject_points[subject] = np.load(subject_files[0])
        for i in range(num_subjects):
            for j in range(i):
                points_i = subject_points[subjects[i]]
                points_j = subject_points[subjects[j]]
                m1, m2, difference = procrustes(points_i, points_j)
                proc_dists_domain[i, j] = difference

        # sns.color_palette("vlag", as_cmap=True)
        mask = np.zeros_like(proc_dists_domain)
        mask[np.triu_indices_from(proc_dists_domain)] = True
        with sns.axes_style("white"):
            g = sns.heatmap(np.array(proc_dists_domain), mask=mask, square=True, vmax=1, cmap='Purples')
            g.set_title(domain + '- Dim: ' + str(dim), fontsize=10)
            g.set_yticklabels([SUBJECT_IDS[k] for k in subjects], rotation=360)
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
        plt.arrow(set1[_i, 0], set1[_i, 1], set2[_i, 0] - set1[_i, 0], set2[_i, 1] - set1[_i, 1],
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
    for d in range(len(subject_files) - 1):
        m1, m2, diff = procrustes(np.load(subject_files[d]), np.load(subject_files[d + 1]))
        # m1, m2, diff = procrustes(np.load(subject_files[d+1]), np.load(subject_files[d]))
        plt.subplot(1, len(DOMAINS) - 1, d + 1)
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
        plt.gca().set_title(DOMAIN_LABELS[DOMAINS[d]] + '->' + DOMAIN_LABELS[DOMAINS[d + 1]], fontsize=10)
        plt.axis('equal')
    plt.suptitle(subject)
    plt.show()


# proc_distances_by_subject()
# proc_distances_consensus()
# proc_dist_by_domain(5, True)

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
