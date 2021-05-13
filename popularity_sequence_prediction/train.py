'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

import os
import time
import argparse
import warnings

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

from utils import PiecewiseSchedule
from model import VariationalEncoderDecoder
from data  import VariationalEncoderDecoderGen
from callbacks import AnnealEveryEpoch, ValidateRecordandSaveBest

warnings.filterwarnings('ignore')

def std_mods(mod_str):
    mod_str = mod_str.upper()
    std_mod = ""
    for mod in ["V","A","T","U"]:
        if mod in mod_str:
            std_mod += mod
    return std_mod


def unfold_mods(mod_str):
    unfolded_mods = []
    mod_str = mod_str.upper()
    for s_mod, l_mod in zip(["V","A","T","U"],
                            ["visual", "aural", "textual", "user"]):
        if s_mod in mod_str:
            unfolded_mods.append(l_mod)
    return unfolded_mods

def get_gen(feature_root, target_root, split_root, batch_size, modalities):
    train_gen = VariationalEncoderDecoderGen(
         phase         = "train",        
         feature_root  = feature_root,
         target_root   = target_root,
         split_root    = split_root,
         modalities    = modalities,
         batch_size    = batch_size,
         shuffle       = True,
         concat        = False,
    )
    val_gen = VariationalEncoderDecoderGen(
         phase         = "val",        
         feature_root  = feature_root,
         target_root   = target_root,
         split_root    = split_root,
         modalities    = modalities,
         batch_size    = batch_size,
         shuffle       = True,
         concat        = False,
    )
    return train_gen, val_gen


def get_model(input_shapes, rnn_type, modalities, summary=True):
    name_args_dict = {
        "resnet50" : {"hidden_sizes":[16, 8], "activation":"relu"},
        "audiovgg" : {"hidden_sizes":[16, 8], "activation":"relu"},        
        "fudannlp" : {"hidden_sizes":[8, 8], "activation":"relu"},
        "user"   : {"hidden_sizes":[8, 8], "activation":"relu", "uindim": 2664, "uembdim": 4},
    }
    enc_args = {name: name_args_dict[name] for name in modalities}
    dec_args = {"emb_size": None, "hid_size": 8, "out_size": 1, "rnn_type": rnn_type}
    model = VariationalEncoderDecoder(enc_args, dec_args, input_shapes)
    if summary:
        model.summary()
    return model


def train(model, epochs, train_gen, val_gen, lambd, rec_path, model_root):
    lr_schedule = PiecewiseSchedule(
        [[0       , 1e-3],
         [epochs/2, 5e-4],
         [epochs  , 1e-4]], outside_value=1e-4)

    ### In our implementation, we keep lambd fixed.
    lambd_schedule = PiecewiseSchedule(
        [[0,      lambd],
         [epochs, lambd]], outside_value=lambd
    )

    callbacks = [
        AnnealEveryEpoch({"lr": lr_schedule, "kl_gauss": lambd_schedule}),
        ValidateRecordandSaveBest(val_gen, rec_path, model_root),
        ModelCheckpoint(filepath=os.path.join(model_root, "model_{epoch:02d}.h5"), period=1)
    ]

    model.compile(optimizer=optimizers.Adam(), loss="mse")
    model.fit_generator(train_gen, workers=4, epochs=epochs, callbacks=callbacks, validation_data=val_gen)


def train_run(lambd=0.3, length=9, batches=128, epochs=100, rnntype="simple", split=0, encodertype="user", device="0"):
    modalities = ["user"] if encodertype == "user" else ["resnet50", "audiovgg", "fudannlp"]

    ### Set tensorflow session
    tf.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Senerate/reading training and validation data
    feature_root = os.path.join("data")
    target_root = os.path.join("data", "len_{}".format(length))
    split_root = os.path.join("data", "split", str(split))

    train_gen, val_gen = get_gen(feature_root, target_root, split_root, batches, modalities)

    ### Get input shapes and set up the model
    input_shapes = train_gen.mod_shape + [train_gen.abstime_shape] if encodertype == "user" else [
                                                                                                     train_gen.useremb_shape] + train_gen.mod_shape + [
                                                                                                     train_gen.abstime_shape]
    model = get_model(input_shapes, rnntype, modalities)

    ### Set up the training process
    save_root = os.path.join("models", encodertype, "len_{}".format(length), str(split), str(lambd),
                             time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_root = os.path.join(save_root)
    rec_path = os.path.join(save_root, "validation_records.txt")
    train(model, epochs, train_gen, val_gen, lambd, rec_path, model_root)

    K.clear_session()
    return save_root

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.3,
                        help="weights of the KL divergence, control the wideness of information bottleneck.")
    parser.add_argument("--length", type=int, default=9,
                        help="resampled timesteps of the popularity sequence")
    parser.add_argument("--batches", type=int, default=128,
                        help="size of training batch")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--rnntype", type=str, default="simple", choices=["simple", "lstm"],
                        help="the type of rnn, vanilla or lstm")
    parser.add_argument("--split", type=int, default=1,
                        help="use which split of train/val/test")
    parser.add_argument("--encodertype", type=str, default="user",
                        help="the type of encoder, user or video")
    parser.add_argument("--device", type=str, default="0",
                        help="use which GPU device")
    args = parser.parse_args()
    train_run(lambd=args.lambd, length=args.length, batches=args.batches, epochs=args.epochs, rnntype=args.rnntype, split=args.split, encodertype=args.encodertype, device=args.device)


