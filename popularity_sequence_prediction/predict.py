'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

from data import *
from train import *
from layers import ProductOfExpertGaussian as POE


### Modality to their short name
mod_rep_dict = {
    "resnet50" :  "V",
    "audiovgg" :  "A",
    "fudannlp" :  "T",
}

### Short name to the modalities
rep_mod_dict = \
    {value: key for key, value in mod_rep_dict.items()}


### Modality to their shape
mod_shape_dict = {
    "resnet50" :  128,
    "audiovgg" :  128,
    "fudannlp" :  20,
}


def ord_rep(rep_str):
    ord_rep = ""
    for i, letter in enumerate(["V", "A", "T"]):
        if letter in rep_str:
            ord_rep += letter
    return ord_rep


def rep2mods(rep_str):
    test_mods = []
    for i, letter in enumerate(["V", "A", "T"]):
        if letter in rep_str:
            test_mods.append(rep_mod_dict[letter])
    return test_mods


def mods2index(mods_list, mod_pos_dict):
    idx_list = [mod_pos_dict[mod] for mod in mods_list]
    return sorted(idx_list)    


def get_model_info(model_path):
    info_dict = {}
    path_list = model_path.split(os.path.sep)
    info_dict["encodertype"] = path_list[-6]
    info_dict["length"] = int(path_list[-5].split("_")[-1])
    info_dict["split"]  = int(path_list[-4])
    info_dict["lambda"] = float(path_list[-3])
    return info_dict


def get_testgen(feature_root, target_root, split_root, test_mods, phase):
    '''
        Get data generator for test
    '''
    test_gen = VariationalEncoderDecoderGen(
        phase         = phase,
        feature_root  = feature_root,
        target_root   = target_root,
        split_root    = split_root,
        modalities    = test_mods,
        batch_size    = 128,
        shuffle       = False, ### You cannot shuffle data in test phase
        concat        = False,        
    )
    return test_gen


def build_test_model(model_path,
                     train_shapes,
                     test_mods,
                     rnn_type,
                     mod_pos_dict,
                     modalities,
                     summary=False):
    model = get_model(train_shapes, rnn_type, modalities, summary=False)
    model.load_weights(model_path)

    if modalities == ["user"]:
        ### Get the input tensor
        abst_in = model.inputs[-1]
        uid_in = model.inputs[0]
        mods_in = model.inputs[1]
        uid_emb = model.get_layer("uid_emb")(uid_in)
        uid_emb = model.get_layer("uid_emb_reshape")(uid_emb)
        concat = layers.Concatenate(axis=-1)([uid_emb, mods_in])
        mean_stds = model.encoders[0](concat)
        mean = mean_stds[0]
        input_space = [uid_in] + [mods_in] + [abst_in]
        preds_seq = model.decoder([mean, abst_in])

        ### Get learnt user embeddings
        test_model = [models.Model(inputs=input_space, outputs=mean_stds)]
        ### Evaluation
        test_model.append(models.Model(inputs=input_space, outputs=preds_seq))
        if summary:
            [test_model[i].summary() for i in range(len(test_model))]
    else:
        ### Get index for each modality
        mod_idxes = mods2index(test_mods, mod_pos_dict)

        ### Get the input tensor indicated by mod_idxes
        uemb_in = model.inputs[0]
        mods_in = [model.inputs[1:-1][i] for i in mod_idxes]
        abst_in = model.inputs[-1]

        ### Build the model for prediction
        encoders  = [model.encoders[i] for i in mod_idxes]
        mean_stds = [encoder(mod_in) for encoder, mod_in in zip(encoders, mods_in)]
        mean, _ = POE()(mean_stds)
        preds_seq = model.decoder([mean, abst_in])
        test_model = models.Model(inputs=[uemb_in]+mods_in+[abst_in], outputs=preds_seq)
        if summary:
            test_model.summary()

    return test_model


def user_predict(model, test_gen, pred_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size
    timesteps = test_gen.timesteps
    # emb_dim = model[0].output_shape[0][-1]

    ### for user-encoder evaluation
    preds = np.empty((num_videos, timesteps), dtype=np.float32)
    truth = np.empty((num_videos, timesteps), dtype=np.float32)

    for i, [features, target] in enumerate(test_gen):
        preds_batch = np.squeeze(model[1].predict(features))
        preds[i * batch_size:(i + 1) * batch_size] = preds_batch.squeeze()
        truth[i * batch_size:(i + 1) * batch_size] = target.squeeze()

    if pred_path is not None:
        print("Prediction has been saved to {}".format(pred_path))
        np.save(pred_path, preds)

    return preds, truth


def uemb_output(model, test_gen, emb_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size
    # timesteps = test_gen.timesteps
    emb_dim = model[0].output_shape[0][-1]

    ### for user embeddings
    uemb_mean = np.empty((num_videos, emb_dim), dtype=np.float32)
    uemb_std = np.empty((num_videos, emb_dim), dtype=np.float32)

    for i, [features, target] in enumerate(test_gen):
        uemb_mean[i * batch_size:(i + 1) * batch_size] = model[0].predict(features)[0].squeeze()
        uemb_std[i * batch_size:(i + 1) * batch_size] = model[0].predict(features)[1].squeeze()

    uemb = np.concatenate((uemb_mean[:, None, :], uemb_std[:, None, :]), axis=1)

    if emb_path is not None:
        print("User embeddings have been saved to {}".format(emb_path))
        np.save(emb_path, uemb)



def predict(test_model, test_gen, save_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size
    timesteps  = test_gen.timesteps

    preds = np.empty([num_videos, timesteps], dtype=np.float32)
    truth = np.empty([num_videos, timesteps], dtype=np.float32)

    for i, [features, targets] in enumerate(test_gen):
        preds[i*batch_size:(i+1)*batch_size] = test_model.predict(features).squeeze()
        truth[i*batch_size:(i+1)*batch_size] = targets.squeeze()

    if save_path is not None:
        print("Prediction saved to {}".format(save_path))
        np.save(save_path, preds)
    return preds, truth


def evaluate(preds, truth, save_path):
    def pearson_corr(preds, truth):
        corr = 0
        num_samples = len(preds)
        cnt_samples = num_samples
        for i in range(num_samples):
            corr_this =  pd.Series(preds[i]).corr(pd.Series(truth[i]))
            if np.isnan(corr_this):
                cnt_samples = cnt_samples-1
                continue
            corr += corr_this
        return corr / cnt_samples

    def spearman_corr(preds, truth):
        corr = 0
        p_val = 0
        num_samples = len(preds)
        cnt_samples = num_samples
        for i in range(num_samples):
            corr_this, p_value_this = spearmanr(pd.Series(preds[i]), pd.Series(truth[i]))
            if np.isnan(corr_this):
                cnt_samples = cnt_samples-1
                continue
            corr += corr_this
            p_val += p_value_this
        return corr / cnt_samples, p_val / cnt_samples

    def nmse(preds, truth):   
        return np.mean(np.square(preds - truth)) / (truth.std()**2)

    nmse = nmse(preds, truth)
    corr = pearson_corr(preds, truth)
    srcc, p_val = spearman_corr(preds, truth)

    table = pd.DataFrame({
            "nmse" : [nmse],
            "corr" : [corr],
            "srcc" : [srcc],
            "srcc_p_val" : [p_val]})
    print("test nmse: {:.4f}".format(nmse))
    print("test corr: {:.4f}".format(corr))
    print("test srcc: {:.4f}, p_value: {:.4f}".format(srcc, p_val))
    table.to_csv(save_path, mode='a', index=False, sep="\t")
    return nmse, corr, srcc


def test_run(model_path, rnn_type="simple", abbr_test_mods="U", device="0"):
    ### Set tensorflow session
    tf.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Save path to the prediction result
    model_info = get_model_info(model_path)
    model_root = os.path.split(model_path)[0]
    test_root = os.path.join(model_root, "test", std_mods(abbr_test_mods))
    if not os.path.exists(test_root):
        os.makedirs(test_root)
    pred_path = os.path.join(test_root, "predict.npy")

    ### Get the test data generator
    feature_root = os.path.join("data")
    split_root = os.path.join(feature_root, "split", str(model_info["split"]))
    target_root = os.path.join(feature_root, "len_{}".format(model_info["length"]))

    ### Get the model for prediction
    if model_info["encodertype"] == "user":
        train_mods = ["user"]
        mod_pos_dict = {"user": 0}
        uemb_path = os.path.join(feature_root, "user_emb.npy")
        test_mods = train_mods
        train_shapes = [[1], [3]] + [[model_info["length"], 1]]
        test_model = build_test_model(model_path, train_shapes, test_mods, rnn_type, mod_pos_dict, train_mods)
        test_gen = get_testgen(feature_root, target_root, split_root, test_mods, phase="test")

        ### Evaluation
        preds, truth = user_predict(test_model, test_gen, pred_path)
        ### User embeddings output
        uemb_gen = get_testgen(feature_root, target_root, split_root, test_mods, phase="all")
        uemb_output(test_model, uemb_gen, uemb_path)

    else:
        train_mods = ["resnet50", "audiovgg", "fudannlp"]
        mod_pos_dict = {mod: train_mods.index(mod) for mod in mod_rep_dict.keys()}
        test_mods = rep2mods(ord_rep(abbr_test_mods))
        train_shapes = [[2, 8]] + [[mod_shape_dict[mod]] for mod in train_mods] + [[model_info["length"], 1]]
        test_model = build_test_model(model_path, train_shapes, test_mods, rnn_type, mod_pos_dict, train_mods)
        test_gen = get_testgen(feature_root, target_root, split_root, test_mods, phase="test")
        preds, truth = predict(test_model, test_gen, pred_path)

    ### Evaluate model with numerous indexes
    eval_path = os.path.join(test_root, "eval.txt")
    nmse, corr, srcc = evaluate(preds, truth, eval_path)

    K.clear_session()
    return nmse, corr, srcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="path where the pre-trained model is stored.")
    parser.add_argument("--rnn_type", type=str, default="simple",
                        help="type of decoder")
    parser.add_argument("--test_mods", type=str, default="U",
                        help="modalities available in the test phase")
    parser.add_argument("--device", type=str, default="0",
                        help="specify the GPU device")
    args = parser.parse_args()
    test_run(model_path=args.model_path, rnn_type=args.rnn_type, abbr_test_mods=args.test_mods, device=args.device)