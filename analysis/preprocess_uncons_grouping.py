"""
 My raw data looks like a csv file with headers includes
 stim1, stim2, ..., stim8, stim clicked, participant, session etc.
 There is no ref in this paradigm. All pairs of stimuli are compared to each other, and chosen in
 order of similarity, i.e., the pair that seems most like each other are picked first (two sequential clicks) and so on.

 Preprocessing of raw clicks into comparisons of pairs of distances happens here

 Returns two dicts: one for repeats of a comparison and one for counts (judgments in a direction)
 Keys should be in this format: 'am400,bm300<bm200,gp400'

 ### WARNING: this removes string separated by underscores from stim names, assumes it is the example number. #########
 ### Please remove underscores from stim names/ image filenames if they are to be treated individually ################
"""

import csv
import ast
from itertools import combinations
from analysis.preprocess import get_response_files


def get_remaining_pairs(all_pairs):
    """
    Given a list of pairs, return all pairs left after you take out the first pair
    For example, given all_pairs = [(s1, s2), (s3, s4), (s5, s6)]
    s3, s4, s5, s6
    return [(s3, s4), (s3, s5), (s3, s6), (s4, s5), (s4, s6), (s5, s6)]
    :return: a list of pairs of stimuli not including the first pair
    """
    stimuli = []
    for i in range(len(all_pairs)):
        stimuli.append(all_pairs[i][0])
        stimuli.append(all_pairs[i][1])
    comparison_pairs = list(combinations(stimuli, 2))
    comparison_pairs.remove((all_pairs[0]))
    return comparison_pairs


def process_clicks_uncons_gp(subject, data_directory):
    """
    Read in raw csv response files and return pairwise choices and number of repeats in separate dictionaries
    :param subject: subject initials
    :param data_directory: path to subject-data/condition/raw
    :param  num_stim_per_trial: should be 8 if there are 8 stimuli in a circle
    :return: judgment_counts, repeats
    """
    # initialize output vars
    repeats = {}
    judgment_counts = {}
    ses_nums = []

    # read in csv files
    subject_files = get_response_files(subject, data_directory)
    print('Processing', str(len(subject_files)), 'files for UNCONSTRAINED GROUPING exp')
    for file in subject_files:
        num_trials = 0
        ses_num = None
        with open(file, 'r', encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)
            # not skipping line, bcz not passing in fieldnames treats first line as header...
            # can use next() to skip header line BUT ONLY IF YOU PASSED IN FIELDNAMES!!
            for row in reader:
                num_trials += 1
                if ses_num is None:
                    ses_num = row['session']
                # for each trial, compile a dictionary of choices and judgments ensure that for a comparison of the form
                # s1,s2 < s3,s4, s1 and s2 are in alphabetical order and s3 and s4 are in alphabetical order. This
                # imposes an order on each unique pairwise comparison, so there is no risk for another entry reading
                # s2-s1 < s3-s4 etc.
                clicks = ast.literal_eval(row['clicks'])

                for i in range(len(clicks)):
                    idx_a, idx_b = clicks[i]
                    # remove text after underscore
                    ref_pair = sorted([row[idx_a].split('_')[0], row[idx_b].split('_')[0]])
                    comparison_pairs = get_remaining_pairs(clicks[i:])

                    # for each pair of stimuli (s_m, s_n) that are clicked we have several comparison pairs: (s_i, s_j),
                    # (s_r, s_t) ... sort the stimuli within each pair => e.g., (s_n, s_m), (s_i, s_j), (s_t, s_r).
                    # next when recording the counts, write the key based on the alphabetical order again, meaning
                    # if (s_n, s_m) is being compared to (s_i, s_j), the key should be (s_i, s_j) < (s_n, s_m) if
                    # (s_i, s_j) came before (s_n, s_m) in sorted([(s_n, s_m), (s_i, s_j)])
                    # this will help keep track of repeats of a comparison regardless of which pair is clicked.
                    for pair in comparison_pairs:
                        idx_i, idx_j = pair
                        # rm text after underscore
                        comp_pair = sorted([row[idx_i].split('_')[0], row[idx_j].split('_')[0]])
                        pairs = sorted([sorted((comp_pair[0], comp_pair[1])), sorted((ref_pair[0], ref_pair[1]))])
                        key = '{},{}<{},{}'.format(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
                        if key not in repeats:
                            repeats[key] = 1
                            # if ref pair is listed first, then count judgment as choosing ref pair to be more similar
                            # do nothing otherwise, if ref pair is listed second, that implies another pair was chosen
                            # before the ref pair, i.e., the pair that was clicked first. downstream processing will
                            # interpret this as a judgment in the opposite direction
                            if pairs[0][0] == ref_pair[0] and pairs[0][1] == ref_pair[1]:
                                judgment_counts[key] = 1
                            else:
                                judgment_counts[key] = 0
                        else:
                            repeats[key] += 1
                            if pairs[0][0] == ref_pair[0] and pairs[0][1] == ref_pair[1]:
                                judgment_counts[key] += 1

        ses_nums.append(ses_num)
    if len(set(ses_nums)) < 10:
        print("--------------- WARNING: Less than 10 sessions found. Check that data directory has responses.csv files"
              " for all 10 sessions ------------------------")
    return judgment_counts, repeats, ses_nums

# # return a dictionary of repeats and counts for all unique comparisons of pairs of stimuli.
# # write to a json file ...
# session_nums = [int(re.findall(r'\d+', s)[0]) for s in ses_nums]
# sessions_str = 'Sess{}_{}'.format(min(session_nums), max(session_nums))
# with open(f'{output_directory}/{subject}_{sessions_str}_{experiment_name}_judgments.json', 'w') as fp:
#     json.dump(judgment_counts, fp, indent=2)
# with open(f'{output_directory}/{subject}_{sessions_str}_{experiment_name}_repeats.json', 'w') as fp:
#     json.dump(repeats, fp, indent=2)
# print("The name of the new file: " +
#       '{}_{}_{}_exp.json'.format(subject, sessions_str, experiment_name) +
#       " in subject-data/<condition>/preprocessed")