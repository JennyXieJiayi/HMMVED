'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

import os
import numpy as np
import pandas as pd

test_num = 30324
val_num = 30324
total_num = 186637


def split_data(split_idx):
	'''
		get the indexes for the train, val 
		and test data
	'''
	data_idxes = np.random.permutation(total_num)
	test_idxes = data_idxes[0: test_num]
	val_idxes = data_idxes[test_num: test_num + val_num]
	train_idxes = data_idxes[test_num + val_num: ]

	if not os.path.exists("{}".format(split_idx)):
		os.makedirs("{}".format(split_idx))
	pd.DataFrame({"train_idxes" : sorted(train_idxes)}).to_csv("{}/train.txt".format(split_idx) , header=None, index=False)
	pd.DataFrame({"val_idxes" : sorted(val_idxes)}).to_csv("{}/val.txt".format(split_idx) , header=None, index=False)
	pd.DataFrame({"test_idxes": sorted(test_idxes)}).to_csv("{}/test.txt".format(split_idx), header=None, index=False)


def main():
	for i in range(5):
		split_data(i)

if __name__ == '__main__':
	main()
