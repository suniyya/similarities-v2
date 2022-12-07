"""
This is a helper script to generate randomized trial and block configurations for the perceptual spaces
fMRI study. In each group of blocks (let's call this 1 round), each of the 37 stimuli from alll 5 domains are
shown to a subject at least once while they perform a one-back memory task.

The structure is as follows:
- 8 repeats over the course of the experiment
- Each repeat has 5 rounds.
- Each round has 5 blocks.
- Each block contains 9 stimuli, one of which is repeated. This stimulus two times in succession for the one-back
test. The position of the repeating stimulus varies randomly across blocks. In each block, the stimuli are from
the same domain.
Each stimulus appears for 1 second and is followed by a blank screen (with a fixation cross) for 3 seconds.

Over the course of a full round, 8 stimuli from each of the domains will have been shown to the subject.
Over the course of a full repeat, all 37 stimuli from all domains will have been shown to the subject at least once.
8 repeats will yield a minimum of 8 presentations per stimulus per domain.

Our key constraints:
- each block must have one repeat
- the repeated stimulus must appear right after the original - no intervening stimuli.
- the groupings of stimuli into blocks is randomized
- the order of blocks within a round is randomized, such that, the next round cannot start with the domain
  that was immediately seen or one before this. For example, if round 1 ends with domain X, the first or
  second domain of round 2 cannot be X, and if domain Y was the second last domain in round 1, round 2 cannot
  begin with domain Y. That is, the same domain should only be shown again after a gap of two blocks.
- each round must have exactly one block from each domain - this is just a choice for simplicity and to keep
  things as consistent as possible across conditions.
- all stimuli from all domains should have been shown at least once by the end of a repeat and some are shown
  twice.
"""
import os
import analysis.util as utils
import numpy as np


def group_stimuli_into_equal_blocks(stim_list, num_blocks=5, block_size=8):
    num_repeated = num_blocks * block_size - len(stim_list)
    if num_repeated > 3:
        print('Are you sure you have the correct parameters? Stimulus list should have 37 stimuli.\n'
              '5 blocks of 8.')
    repeated = list(np.random.choice(stim_list, num_repeated, replace=False))
    all_stimuli = stim_list + repeated
    np.random.shuffle(all_stimuli)
    blocks = [all_stimuli[i * block_size:i * block_size + block_size] for i in range(num_blocks)]
    # ensure one stimulus is always repeated per block for one-back
    for block in blocks:
        idx = np.random.choice(range(block_size))
        to_insert = block[idx]
        block.insert(idx+1, to_insert)
    return blocks


def consecutive_round_orders(num_domains=5, num_rounds=5):
    orders = [[None]*num_domains for _ in range(num_rounds)]
    temp = list(range(num_domains))
    np.random.shuffle(temp)
    orders[0] = temp
    options = set(orders[0])
    for i in range(num_rounds-1):
        # ensure that the first domain of one round is not among the last two of the preceding round
        orders[i+1][0] = np.random.choice(orders[i][0:num_domains-2])
        choose_from = options - {orders[i][-1], orders[i + 1][0]}
        orders[i+1][1] = np.random.choice(list(choose_from))
        remaining = list(options - {orders[i+1][0], orders[i+1][1]})
        np.random.shuffle(remaining)
        orders[i+1][2:] = remaining
    return orders


def write_parent_file(rounds, directory):
    domains = ['texture', 'texture-like', 'image-like', 'image', 'word']
    image_sources = ['animal_textures/textures_big_checks', 'animal_intermediates/texture-like-opaque',
                     'animal_intermediates/image-like-opaque',
                     'animal_images', 'None']
    lines = ['round,domain,condFile,imageSource\n']
    for r in range(len(rounds)):
        for domain_idx in rounds[r]:
            cond_file = '{}/round_{}_{}_block_order.csv'.format(directory, r+1, domains[domain_idx])
            lines.append('{},{},{},{}\n'.format(r+1, domains[domain_idx], cond_file, image_sources[domain_idx]))
    f = open('{}/domain_block_order.csv'.format(directory), 'w')
    f.writelines(lines)
    return


def write_block_files(filename, trials, domain):
    f = open(filename, 'w')
    lines = ['name,imageFile,correct\n']
    for i in range(len(trials)):
        if domain != 'word':
            im_name = './stimuli/{}/{}.png'.format(domain, trials[i])
        else:
            im_name = None
        correct = 0
        if i > 0:
            correct = int(trials[i] == trials[i-1])
        lines.append('{},{},{}\n'.format(trials[i], im_name, correct))
    f.writelines(lines)
    return


if __name__ == '__main__':
    DOMAINS = ['texture', 'texture-like', 'image-like', 'image', 'word']
    NUM_REPEATS = 8
    PATH = '/Users/suniyya/Dropbox/Research/Thesis_Work/fMRI_Aim2/paradigm/components/' \
           'perceptualSpaces/subjects/YCL'

    stimuli = utils.stimulus_names()
    # group stimuli into blocks of 8 - 37 - into 40, with 3 repeated.
    # run separately for each domain
    # x['texture'] = [block1, block2, block3, block4, block5]
    # ...
    # x['image'] = [block1, block2, block3, block4, block5]
    for rep in range(NUM_REPEATS):
        directory = 'repeat_' + str(rep+1)
        path = os.path.join(PATH, directory)
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.mkdir(path)
        stimuli_in_blocks = {}
        for d in range(len(DOMAINS)):
            stimuli_in_blocks[DOMAINS[d]] = group_stimuli_into_equal_blocks(stimuli)
        # shuffle block orders for 5 rounds of 5 domain blocks each
        domain_order_by_round = consecutive_round_orders()
        print(domain_order_by_round)

        for roundNum in range(len(domain_order_by_round)):
            for domainIdx in domain_order_by_round[roundNum]:
                cond_filename = '{}/round_{}_{}_block_order.csv'.format(path, roundNum+1, DOMAINS[domainIdx])
                write_block_files(cond_filename, stimuli_in_blocks[DOMAINS[domainIdx]][roundNum], DOMAINS[domainIdx])
        write_parent_file(domain_order_by_round, path)
