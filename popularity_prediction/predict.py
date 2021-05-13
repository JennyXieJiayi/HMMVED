'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''


import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

from data import *
from train import *
from layers import ProductOfExpertGaussian as POE


def mod2index(mod_str):
    idx_list = []
    for i, mod in enumerate(["V", "A", "T", "U"]):
        if mod in mod_str:
            idx_list.append(i)
    return idx_list    


def get_model_info(model_path):
    info_dict = {}
    path_list = model_path.split(os.path.sep)
    info_dict["split"] = int(path_list[-5])
    info_dict["lambda"] = float(path_list[-3])
    info_dict["train_mods"] = unfold_mods(path_list[-4])
    return info_dict


def get_testgen(feature_root, pred_mods, split_root, phase):
    '''
        Get data generator for test
    '''
    test_gen = VariationalEncoderDecoderGen(
        feature_root = feature_root,
        modalities = pred_mods,
        split_root = split_root,
        phase = phase,
        batch_size  = 512,
        shuffle = False, ### you can not shuffle data in test phase
    )
    return test_gen


def build_predict_model(model_path,
                        input_shapes,
                        train_mods,
                        pred_mods,
                        summary=True):
    model = get_model(input_shapes, train_mods, summary=False)
    model.load_weights(model_path)

    if unfold_mods(pred_mods) == ["user"]:
        ### Get the input tensor
        uid_in = model.inputs[0]
        mods_in = model.inputs[1]
        uid_emb = model.get_layer("uid_emb")(uid_in)
        uid_emb = model.get_layer("uid_emb_reshape")(uid_emb)
        concat = layers.Concatenate(axis=-1)([uid_emb, mods_in])
        mean_stds = model.encoders[0](concat)
        mean = mean_stds[0]
        input_space = [uid_in] + [mods_in]
        preds = model.decoder(mean)

        ### Get user embeddings
        uemb_model = models.Model(inputs=input_space, outputs=mean_stds)
        ### Evaluate
        upred_model = models.Model(inputs=input_space, outputs=preds)
        pred_model = [uemb_model, upred_model]
        if summary:
            [pred_model[i].summary() for i in range(len(pred_model))]
    else:
        ### Get index for each modality
        pred_mod_idxes = mod2index(pred_mods)

        ### Get the input tensor indicated by mod_idxes
        uemb_in = model.inputs[0]
        mods_in = [model.inputs[1:][i] for i in pred_mod_idxes]

        ### Build the model for prediction
        encoders  = [model.encoders[i] for i in pred_mod_idxes]
        mean_stds = [encoder(mod_in) for encoder, mod_in in zip(encoders, mods_in)]
        mean, _ = POE()(mean_stds)
        preds = model.decoder(mean)
        pred_model = models.Model(inputs=[uemb_in]+mods_in, outputs=preds)
        if summary:
            pred_model.summary()

    return pred_model


def user_predict(model, test_gen, pred_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size

    ### for user-encoder evaluation
    preds = np.empty(num_videos, dtype=np.float32)
    truth = np.empty(num_videos, dtype=np.float32)

    for i, [features, target] in enumerate(test_gen):
        preds_batch = np.squeeze(model[1].predict(features))
        preds[i*batch_size:(i+1)*batch_size] = preds_batch.squeeze()
        truth[i*batch_size:(i+1)*batch_size] = target.squeeze()

    mean, std = test_gen.target_stats
    preds = np.exp(preds * std + mean)
    truth = np.exp(truth * std + mean)

    if pred_path is not None:
        print("Prediction has been saved to {}".format(pred_path))
        np.save(pred_path, preds)
    return preds, truth


def uemb_output(model, test_gen, emb_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size
    emb_dim = model[0].output_shape[0][-1]

    ### for user embeddings
    uemb_mean = np.empty((num_videos, emb_dim), dtype=np.float32)
    uemb_std = np.empty((num_videos, emb_dim), dtype=np.float32)

    for i, [features, target] in enumerate(test_gen):
        uemb_mean[i*batch_size:(i+1)*batch_size] = model[0].predict(features)[0].squeeze()
        uemb_std[i*batch_size:(i+1)*batch_size] = model[0].predict(features)[1].squeeze()

    # uemb = np.concatenate((np.expand_dims(uemb_mean, 1), np.expand_dims(uemb_std, 1)), axis=1)
    uemb = np.concatenate((uemb_mean[:,None,:], uemb_std[:,None,:]), axis=1)
    if emb_path is not None:
        print("User embeddings have been saved to {}".format(emb_path))
        np.save(emb_path, uemb)


def predict(model, test_gen, mod_idxes, save_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size

    preds = np.empty(num_videos, dtype=np.float32)
    truth = np.empty(num_videos, dtype=np.float32)

    for i, [feature, target] in enumerate(test_gen):
        feature = [feature[0]] + [feature[j+1] for j in mod_idxes]
        preds_batch = np.squeeze(model.predict(feature))
        preds[i*batch_size:(i+1)*batch_size] = preds_batch.squeeze()
        truth[i*batch_size:(i+1)*batch_size] = target.squeeze()

    mean, std = test_gen.target_stats
    preds = np.exp(preds * std + mean)
    truth = np.exp(truth * std + mean)

    if save_path is not None:
        print("Prediction saved to {}".format(save_path))
        np.save(save_path, preds)
    return preds, truth


def evaluate(preds, truth, save_path):
    def nmse(preds, truth):
        return np.mean(np.square(preds - truth)) / (truth.std()**2)

    def pearson_corr(preds, truth):
        return pearsonr(preds, truth)

    def spearman_corr(preds, truth):
        return spearmanr(preds, truth)

    nmse = nmse(preds, truth)
    pcc = pearson_corr(preds, truth)
    src = spearman_corr(preds, truth)

    table = pd.DataFrame({
            "nmse" : [nmse],
            "pcc"  : [pcc],
            "src"  : [src]})
    print("test nmse: {}".format(nmse))
    table.to_csv(save_path, index=False, sep="\t")
    return nmse, pcc, src


def test_run(model_path, abbr_test_mods="U", device="0"):
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
    feature_root = os.path.join("dataset")
    split_root = os.path.join(feature_root, "split", str(model_info["split"]))
    test_gen = get_testgen(feature_root, unfold_mods(abbr_test_mods), split_root, phase="test")

    ### Get the model for prediction
    if model_info["train_mods"] == ["user"]:
        input_shapes = test_gen.mod_shape
        pred_model = build_predict_model(model_path, input_shapes, model_info["train_mods"], abbr_test_mods)
        ### Evaluation
        preds, truth = user_predict(pred_model, test_gen, pred_path)
        ### User embeddings output
        uemb_path = os.path.join(feature_root, "user_emb.npy")
        uemb_gen = get_testgen(feature_root, unfold_mods(abbr_test_mods), split_root, phase="all")
        uemb_output(pred_model, uemb_gen, uemb_path)
    else:
        input_shapes = [test_gen.useremb_shape] + test_gen.mod_shape
        pred_model = build_predict_model(model_path, input_shapes, model_info["train_mods"], abbr_test_mods)
        preds, truth = predict(pred_model, test_gen, mod2index(abbr_test_mods), pred_path)

    ### Evaluate model with numerous indexes
    eval_path = os.path.join(test_root, "eval.txt")
    nmse, pcc, src = evaluate(preds, truth, eval_path)

    K.clear_session()
    return preds, truth, nmse, pcc, src


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
            help="path where the pre-trained model is stored.")
    parser.add_argument("--test_mods", type=str, default="VAT",
            help="modalities available in the test phase")
    parser.add_argument("--device", type=str, default="0",
                        help="specify the GPU device")
    args = parser.parse_args()
    test_run(model_path=args.model_path, abbr_test_mods=args.test_mods, device=args.device)