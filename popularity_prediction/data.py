'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

import os
import glob

import numpy as np
import pandas as pd
from tensorflow import keras


class VariationalEncoderDecoderGen(keras.utils.Sequence):
    '''
        Generate training data, validation, test data
    '''
    def __init__(self,
                 feature_root,
                 modalities,
                 split_root,
                 phase,
                 batch_size,
                 shuffle=True):

        assert phase in ["train", "val", "test", "all"], \
            "phase must be one of train, val, test, all!"

        ### Load the indexes of the videos
        if phase is not "all":
            index_path = os.path.join(split_root, "{}.txt".format(phase))
            phase_idxes = pd.read_table(index_path, header=None).values.squeeze()
        else:
            total_num = -np.inf
            for phase_one in ["train", "val", "test"]:
                index_path = os.path.join(split_root, "{}.txt".format(phase_one))
                total_num = max(total_num, np.max(pd.read_table(index_path, header=None).values.squeeze()))
            phase_idxes = np.arange(total_num + 1)

        ### Load the features from specified modalities
        self.modalities = modalities
        self.features = []

        if modalities == ["user"]:
            feature_path = os.path.join(feature_root, "user.npy")
            self.features.append(np.load(feature_path)[phase_idxes][:, 0].reshape(-1, 1))  # user_idxes
            self.features.append(np.load(feature_path)[phase_idxes][:, 1:])  # user_info
        else:
            for modality in modalities:
                feature_path = os.path.join(feature_root, "{}.npy".format(modality))
                self.features.append(np.load(feature_path)[phase_idxes])

            ### Load the user embedding
            uemb_path = os.path.join(feature_root, "user_emb.npy")
            self.useremb = np.load(uemb_path)[phase_idxes]

        ### Load the groundtruth
        target_path = os.path.join(feature_root, "target.npz")
        self.target = np.load(target_path)["target"][phase_idxes]
        self.target_stats = [np.load(target_path)["mean"], np.load(target_path)["std"]]

        ### Data&Batch information
        self.num_videos  = len(phase_idxes)
        self.video_idxes = np.arange(self.num_videos)
        self.batch_size  = batch_size

        ### Shuffle the video indexes if necessary
        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def on_epoch_end(self): 
        '''
            Shuffle the index after each epoch finished
        '''
        if self.shuffle:
            np.random.shuffle(self.video_idxes)    

    def __len__(self): 
        '''
            The total number of batches
        '''
        self.batch_num = self.num_videos // self.batch_size + 1
        return self.batch_num

    def __getitem__(self, i):
        '''
            Return the batch indexed by i
        '''
        batch_idxes = self.video_idxes[i*self.batch_size:(i+1)*self.batch_size]
        batch_size  = len(batch_idxes)

        batch_features = [np.empty((batch_size, feature.shape[-1]), dtype=np.float32) 
                                for feature in self.features]
        batch_target = np.empty((batch_size, 1), dtype=np.float32)

        if self.modalities == ["user"]:
            for j, idx in enumerate(batch_idxes):
                batch_features[0][j] = self.features[0][idx]
                batch_features[1][j] = self.features[1][idx]
                batch_target[j] = self.target[idx]
        else:
            batch_useremb = np.empty((batch_size, self.useremb.shape[1], self.useremb.shape[-1]), dtype=np.float32)
            for j, idx in enumerate(batch_idxes):
                for k in range(len(self.modalities)):
                    batch_features[k][j] = self.features[k][idx]
                batch_useremb[j] = self.useremb[idx, :]
                batch_target[j] = self.target[idx]
            batch_features = [batch_useremb] + batch_features

        return batch_features, batch_target

    @property
    def mod_shape(self):
        return [[feature.shape[-1]] for feature in self.features]

    @property
    def useremb_shape(self):
        return [] if self.modalities == ["user"] else list(self.useremb.shape[1:])


if __name__ == "__main__":
    '''
        For test purpose ONLY
    '''
    pass