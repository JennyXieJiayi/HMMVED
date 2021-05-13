'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

import os
import numpy as np
import pandas as pd

# from train import train_run
# from predict import test_run


def basic_exp():
    user_lambd_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    video_lambd_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    split_list = np.arange(5)
    for split_idx in split_list:
        for user_lambd in user_lambd_list:
            ### train user prior encoder decoder
            user_model_root = train_run(lambd=user_lambd, split=split_idx, encodertype="user")
            ### evaluate user prior encoder decoder and obtain the learnt user embeddings
            user_model_path = os.path.join(user_model_root, "model_100.h5")
            test_run(model_path=user_model_path, abbr_test_mods="U")
            for video_lambd in video_lambd_list:
                ### train video encoder decoder with hidden stochastic user embeddings as prior
                video_model_root = train_run(lambd=video_lambd, split=split_idx, encodertype="video")
                ### evaluate video encoder decoder
                video_model_path = os.path.join(video_model_root, "model_100.h5")
                test_run(model_path=video_model_path, abbr_test_mods="VAT")


if __name__ == '__main__':
    ### basic experiments
    basic_exp()


