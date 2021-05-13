'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils  import generic_utils
from tensorflow.python.keras.engine import network

from layers import AddGaussianLoss 
from layers import ReparameterizeGaussian
from layers import ProductOfExpertGaussian

import numpy as np

class Encoder(network.Network):
    '''
        Encode video features into hidden 
        representations with mlp
    '''
    def __init__(self,
                 # hidden_sizes,
                 # activation,
                 params,
                 **kwargs):

        super(Encoder, self).__init__(**kwargs)
        self.dense_list = []
        hidden_sizes = params["hidden_sizes"]
        activation = params["activation"]
        for size in hidden_sizes[:-1]:
            self.dense_list.append(
                layers.Dense(size, activation=activation)
            )
        self.dense_mean = layers.Dense(hidden_sizes[-1])
        self.dense_std  = layers.Dense(hidden_sizes[-1])
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2))
        self.exp  = layers.Lambda(lambda x:K.exp(x))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mid = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            mid = dense(layers.Dropout(0.5)(mid))
        mean = self.dense_mean(mid)
        std  = self.exp(self.clip(self.dense_std(mid)))
        self._init_graph_network(x_in, [mean, std])
        super(Encoder, self).build(input_shapes)

class SamplerVideo(network.Network):
    '''
        Sample from variational distribution
    '''
    def __init__(self, **kwargs):
        super(SamplerVideo, self).__init__(**kwargs)
        self.rep_gauss  = ReparameterizeGaussian()
        self.gauss_loss = AddGaussianLoss("video")

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        uemb_in = layers.Input(input_shapes[-1][1:])
        sample  = self.rep_gauss([mean, std])
        kl_loss = self.gauss_loss([mean, std, uemb_in])
        self._init_graph_network([mean, std, uemb_in], [sample, kl_loss])
        super(SamplerVideo, self).build(input_shapes)

    def call(self, inputs):
        inputs  = generic_utils.to_list(inputs)
        [sample, kl_loss], _ = self._run_internal_graph(inputs)
        self.add_loss(kl_loss)
        return sample

class SamplerUser(network.Network):
    '''
        Sample from variational distribution
    '''
    def __init__(self, **kwargs):
        super(SamplerUser, self).__init__(**kwargs)
        self.rep_gauss  = ReparameterizeGaussian()
        self.gauss_loss = AddGaussianLoss("user")

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        sample  = self.rep_gauss([mean, std])
        kl_loss = self.gauss_loss([mean, std])
        self._init_graph_network([mean, std], [sample, kl_loss])
        super(SamplerUser, self).build(input_shapes)

    def call(self, inputs):
        inputs  = generic_utils.to_list(inputs)
        [sample, kl_loss], _ = self._run_internal_graph(inputs)
        self.add_loss(kl_loss)
        return sample

class Decoder(network.Network):
    '''
        Decode sample from sample drawn from
        distribution of hidden representation
    '''
    def __init__(self,
                 emb_size,
                 hid_size,
                 out_size,
                 in_activation="relu",
                 out_activation=None,
                 rnn_type="lstm",
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        assert rnn_type in ["lstm", "simple"]

        ### Whether to embed the abs time before fed into RNN
        self.conv_emb = layers.Conv1D(emb_size, kernel_size=1, \
                activation=in_activation) if emb_size else None
        self.conv_out = layers.Conv1D(out_size, kernel_size=1, \
                activation=out_activation)
        self.rnn_type = rnn_type
        if self. rnn_type == "simple":
            self.rnn = layers.SimpleRNN(units=hid_size, return_sequences=True)
        elif self.rnn_type == "lstm":
            self.rnn = layers.LSTM(units=hid_size, return_sequences=True)
            self.cdense = layers.Dense(hid_size)
            self.hdense = layers.Dense(hid_size)

    def build(self, input_shapes):
        hid_in = layers.Input(input_shapes[0][1:])
        seq_in = layers.Input(input_shapes[1][1:])
        seq_emb  = self.conv_emb(seq_in) if self.conv_emb else seq_in
        if self.rnn_type == "simple":
            pop_pred = self.conv_out(self.rnn(seq_emb, initial_state=hid_in))
        else:
            hhid_in  = self.cdense(hid_in)
            chid_in  = self.hdense(hid_in)
            pop_pred = self.conv_out(self.rnn(seq_emb, initial_state=[hhid_in, chid_in]))
        self._init_graph_network([hid_in, seq_in], pop_pred)
        super(Decoder, self).build(input_shapes)


class VariationalEncoderDecoder(models.Model):
    '''
        Implementation of proposed variational
        encoder-decoder network
    '''
    def __init__(self,
                 enc_args,
                 dec_args,
                 input_shapes,
                 **kwargs):
        super(VariationalEncoderDecoder, self).__init__(**kwargs)
        self.encodertype = "user" if "user" in enc_args.keys() else "video"
        if self.encodertype == "user":
            self.uindim = enc_args['user']['uindim']
            self.uembdim = enc_args['user']['uembdim']
        self.num_feat = len(enc_args)
        self.encoders = [Encoder(enc_arg, name="Encoder_{}".format(i)) \
            for i, enc_arg in enumerate(enc_args.values())]
        self.poe_gaussian = ProductOfExpertGaussian()
        self.decoder = Decoder(**dec_args, name="Decoder")
        self.sampler = SamplerUser(name="Sampler") if self.encodertype == "user" else SamplerVideo(name="Sampler")
        self.build(input_shapes)

    def build(self, input_shapes):
        abst_in = layers.Input(input_shapes[-1], name="abstime")
        if self.encodertype == "user":
            uid_in = layers.Input(input_shapes[0], name="uid")
            mods_in = layers.Input(input_shapes[1], name="uinfo")
            uid_emb = layers.Embedding(self.uindim, self.uembdim, input_length=1, embeddings_initializer='uniform', name="uid_emb")(uid_in)
            uid_emb = layers.Lambda(lambda x: x[:,0,:], name="uid_emb_reshape")(uid_emb)
            concat = layers.Concatenate(axis=-1)([uid_emb, mods_in])
            mean_stds = self.encoders[0](concat)
            mean = mean_stds[0]
            std = mean_stds[1]
            hid_in = self.sampler([mean, std])
            input_space = [uid_in] + [mods_in] + [abst_in]
        else:
            uemb_in = layers.Input(input_shapes[0], name="uemb")
            mods_in = [layers.Input(input_shapes[i+1], name="modality_{}".format(i)) \
                       for i in range(self.num_feat)]
            mean_stds = [encoder(mod_in) for encoder, mod_in in \
                         zip(self.encoders, mods_in)]
            mean, std = self.poe_gaussian(mean_stds)
            hid_in   = self.sampler([mean, std, uemb_in])
            input_space = [uemb_in] + mods_in + [abst_in]

        pop_pred = self.decoder([hid_in, abst_in])
        self._init_graph_network(input_space, pop_pred)
        super(VariationalEncoderDecoder, self).build(input_shapes)

    def call(self, inputs):
        inputs = generic_utils.to_list(inputs)
        pop_pred, _ = self._run_internal_graph(inputs)
        return pop_pred    


if __name__ == "__main__":
    '''
        For test purpose ONLY
    '''
    pass