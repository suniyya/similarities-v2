"""
This file is to put preprocessing methods in. My raw data looks like a csv file with headers including
ref, stim1, stim2, ..., stim8, session, participant and clicks.
Version to preprocess dissimilarity responses for analysis.

Analysis will expect similarity judgments, so the raw responses first need to be reversed so they are in terms of the
similarity wrt reference.

This flipping happens in this script, and is reflected in the output filename
"""

import ast
import csv
import json
import re
from analysis.preprocess import get_response_files

if __name__ == '__main__':
    num_stim_per_trial = input("> Enter the number of stimuli per trial, excluding the central reference (default= 8): ")
    PARADIGM = None
    while PARADIGM not in ['image_exp', 'working_mem_exp']:
        PARADIGM = input("> Name of experiment (choose from [image_exp, working_mem_exp]): ")
    experiment_name = input("> Experiment name: (default= bgca3pt9_DIS )")
    if experiment_name == '' or experiment_name == ' ':
        experiment_name = 'bgca3pt9_DIS'

    instructions = input('Was the subject given the non-standard (dissimilarity) instructions? (y/n)')
    if instructions == 'n':
        raise ValueError('Run analysis.preprocess instead')
    suggested_dir = 'C://Users/jdvicto/Documents/similarities/experiments/{}/subject-data/{}_data'.format(PARADIGM, experiment_name)
    directory = input(
        "> Path to subject-data dir of experiment:(default = {}) ".format(suggested_dir))

    # use defaults if no string entered
    if directory == '' or directory == ' ':
        directory = suggested_dir
    if num_stim_per_trial == '' or num_stim_per_trial == ' ':
        num_stim_per_trial = 8
    else:
        num_stim_per_trial = int(num_stim_per_trial)

    output_directory = '{}/preprocessed'.format(directory)
    data_directory = '{}/raw'.format(directory)
    subject_name = input("> Enter Subject ID: ")
    print('Preprocessing data from {} experiment for subjects {}'.format(experiment_name, subject_name))

    for subject in [subject_name]:
        subject_files = get_response_files(subject, data_directory)
        """Removes any exemplar numbers for each stimulus. If each stimulus has multiple example images(as is the case for JV and MC's texture experiments,
        the names cannot be different, because the modeling analysis will consider each image as a separate stimulus point.
        This file takes in similarity judgments in response to images and converts individual image names into stimulus names by removing whatever follows an underscore.
        For example, the image name 'ap600_124' and 'ap600_010' would both be renamed to 'ap600.'
        """
        if 'face' not in experiment_name:
            trials = {}
            session_nums = []
            for file in subject_files:
                session_num = None
                with open(file, 'r', encoding='utf-8-sig') as csv_file:
                    reader = csv.DictReader(csv_file)
                    reader.__next__()  # to skip header line
                    for row in reader:
                        if session_num is None:
                            print(row['session'])
                            session_num = row['session']
                        # eval is usually not secure but here I created the files parsing
                        sequence = ast.literal_eval(row['clicks'])
                        clicks = [row[stim].split('_')[0] for stim in sequence]
                        stim_positions = ['stim' + str(i) for i in range(1, num_stim_per_trial + 1)]

                        circle = '.'.join(sorted(
                            [row[pos].split('_')[0] for pos in stim_positions])
                        )
                        trial_str = '{}:{}'.format(row['ref'].split('_')[0], circle)
                        # reverse order of clicks before appending - for DIS experiment
                        clicks.reverse()
                        if trial_str in trials:
                            trials[trial_str].append(clicks)
                        else:
                            trials[trial_str] = [clicks]
                session_nums.append(session_num)

            # check how many sessions were covered
            print(session_nums)
            session_nums = [int(re.findall(r'\d+', s)[0]) for s in session_nums]
            sessions_str = 'Sess{}_{}'.format(min(session_nums), max(session_nums))

            with open(f'{output_directory}/final_{subject}_{sessions_str}_{experiment_name}_exp.json', 'w') as fp:
                json.dump(trials, fp, indent=2)
            print("The name of the new file: " + "final_{}_{}_{}_exp.json".format(
                subject, sessions_str, experiment_name))

        trials = {}
        session_nums = []
        for file in subject_files:
            session_num = None
            with open(file, 'r', encoding='utf-8-sig') as csv_file:
                reader = csv.DictReader(csv_file)
                reader.__next__()  # to skip header line
                for row in reader:
                    if session_num is None:
                        session_num = row['session']
                    # eval is usually not secure but here I created the files parsing
                    sequence = ast.literal_eval(row['clicks'])
                    clicks = [row[stim] for stim in sequence]
                    circle = '.'.join(sorted(
                        [row['stim' + str(i)] for i in range(1, num_stim_per_trial + 1)])
                    )
                    # reverse order of clicks before appending - for DIS experiment
                    clicks.reverse()
                    trial_str = '{}:{}'.format(row['ref'], circle)
                    if trial_str in trials:
                        trials[trial_str].append(clicks)
                    else:
                        trials[trial_str] = [clicks]
            session_nums.append(session_num)

        session_nums = [int(re.findall(r'\d+', s)[0]) for s in session_nums]
        sessions_str = 'Sess{}_{}'.format(min(session_nums), max(session_nums))
        with open(f'{output_directory}/{subject}_{sessions_str}_{experiment_name}_exp.json', 'w') as fp:
            json.dump(trials, fp, indent=2)

        print("The name of the new file: " +
              '{}_{}_{}_exp.json'.format(subject, sessions_str, experiment_name) +
              " in subject-data/<condition>/preprocessed")
