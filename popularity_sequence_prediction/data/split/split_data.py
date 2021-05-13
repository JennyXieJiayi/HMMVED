'''
Copyright (c) 2019. IIP Lab, Wuhan University
Created by Y. Zhu, J. Xie and J. Yi, All right reserved
'''

import os
import argparse

import numpy as np
import pandas as pd

p_train = 0.8
p_val = 0.2
total_num = 10287


def split_data(total_num, split_idx):
	'''
		get the indexes for the train, val 
		and test data
	'''
	data_idxes = np.random.permutation(total_num)
	train_num = int(total_num * p_train * p_train)
	val_num = int(total_num * p_train * (1 - p_train))
	train_idxes = data_idxes[ :train_num]
	val_idxes = data_idxes[train_num: train_num + val_num]
	test_idxes = data_idxes[train_num + val_num: ]

	if not os.path.exists("{}".format(split_idx)):
		os.makedirs("{}".format(split_idx))
	pd.DataFrame({"train_idxes" : train_idxes}).to_csv("{}/train.txt".format(split_idx) , header=None, index=False)
	pd.DataFrame({"val_idxes" : val_idxes}).to_csv("{}/val.txt".format(split_idx) , header=None, index=False)
	pd.DataFrame({"test_idxes": test_idxes}).to_csv("{}/test.txt".format(split_idx), header=None, index=False)


def main():
	for i in range(5):
		split_data(total_num, i)

if __name__ == '__main__':
	main()
