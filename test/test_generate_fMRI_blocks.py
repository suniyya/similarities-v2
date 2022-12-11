import unittest
import analysis.generate_fMRI_blocks as fmriTask
import glob


class BlockDesignForFMRI(unittest.TestCase):
    def test_each_block_has_one_repeat_in_correct_position(self):
        stim_list = list(range(37))
        blocks = fmriTask.group_stimuli_into_equal_blocks(stim_list)
        for block in blocks:
            self.assertEqual(len(block), len(set(block)) + 1)
            count = 0
            for i in range(len(block) - 1):
                if block[i] == block[i + 1]:
                    count += 1
            self.assertEqual(count, 1)

    def test_stimuli_grouped_into_five_blocks_with_repeats(self):
        stim_list = list(range(37))
        blocks = fmriTask.group_stimuli_into_equal_blocks(stim_list)
        all_stim = []
        for block in blocks:
            all_stim += block
        self.assertEqual(len(set(all_stim)), len(stim_list))
        self.assertEqual(len(all_stim), len(stim_list) + 3 + len(blocks))

    def test_same_domain_is_shown_after_gap_of_at_least_two_domains(self):
        num_domains = 5
        orders = fmriTask.consecutive_round_orders()
        for order in orders:
            self.assertEqual(len(set(order)), num_domains)
        for i in range(len(orders)-1):
            # test when the first or second domain of the i+1th round was last seen in the ith round
            gap1 = num_domains - orders[i].index(orders[i+1][0]) - 1
            gap2 = (num_domains + 1) - orders[i].index(orders[i+1][1]) - 1
            self.assertGreaterEqual(gap1, 2)
            self.assertGreaterEqual(gap2, 2)

    def test_if_all_stimuli_shown_at_least_once_in_one_repeat(self):
        # write files.
        # read individual block files and store into dict by domain.
        # count appearances of each stim per domain
        # unique stim per domain should be 37.
        # should have some repeated more than once
        # test fails if a stimulus is missing, i.e., unique stim < 37 and total stim != 9*5*5
        DOMAINS = ['texture', 'texture-like', 'image-like', 'image', 'word']
        stimuli = list(range(37))
        stimuli_in_blocks = {}
        for d in range(len(DOMAINS)):
            stimuli_in_blocks[DOMAINS[d]] = fmriTask.group_stimuli_into_equal_blocks(stimuli)
        # shuffle block orders for 5 rounds of 5 domain blocks each
        domain_order_by_round = fmriTask.consecutive_round_orders()
        print(domain_order_by_round)
        for roundNum in range(len(domain_order_by_round)):
            for domainIdx in domain_order_by_round[roundNum]:
                cond_filename = 'round_{}_{}_block_order.csv'.format(roundNum, DOMAINS[domainIdx])
                fmriTask.write_block_files(cond_filename,
                                           stimuli_in_blocks[DOMAINS[domainIdx]][roundNum], DOMAINS[domainIdx])

        seen_stimuli = {'texture': {}, 'texture-like': {}, 'image-like': {}, 'image': {}, 'word': {}}
        files = glob.glob('./round*.csv')
        for filename in files:
            domain = filename.split('_block')[0].split('_')[-1]
            f = open(filename, 'r')
            trials = f.readlines()
            for trial in trials[1:]:
                name = trial.split(',')[0]
                if name not in seen_stimuli[domain]:
                    seen_stimuli[domain][name] = 1
                else:
                    seen_stimuli[domain][name] += 1
        total = 0
        for d in DOMAINS:
            self.assertEqual(len(set(seen_stimuli[d].keys())), 37)
            for k, v in seen_stimuli[d].items():
                total += v
        self.assertEqual(total, 5*5*9)


if __name__ == '__main__':
    unittest.main()
