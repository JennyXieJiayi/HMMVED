'''
Copyright (c) 2021. IIP Lab, Wuhan University
'''

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils  import generic_utils
from tensorflow.python.keras.engine import network

from layers import AddGaussianLoss 
from layers import ReparameterizeGaussian
from layers import ProductOfExpertGaussian


class Encoder(network.Network):
    '''
        Encode features from one specific modality
        into its hidden Gaussian embedding.
    '''
    def __init__(self, params, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_list = []
        for size in params["hidden_sizes"][:-1]:
            self.dense_list.append(
                layers.Dense(size, activation=params["activation"])
            )
        self.dense_mean = layers.Dense(params["hidden_sizes"][-1])
        self.dense_std  = layers.Dense(params["hidden_sizes"][-1])
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2))
        self.exp  = layers.Lambda(lambda x:K.exp(x))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mid = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            mid = dense(layers.Dropout(0.5)(mid))

        ### Carry the relevant information
        mean = self.dense_mean(mid)
        ### Preserve the inherent uncertainty
        std  = self.exp(self.clip(self.dense_std(mid)))        
        self._init_graph_network(x_in, [mean, std])
        super(Encoder, self).build(input_shapes)


class Sampler(network.Network):
    '''
        Use reparameterization trick to take samples 
        from the variational distribution to form an
        unbiased Monte Carlo gradient estimator.
    '''
    def __init__(self,
                 encodertype,
                 **kwargs):

        super(Sampler, self).__init__(**kwargs)
        self.encodertype = encodertype
        self.rep_gauss  = ReparameterizeGaussian()
        self.gauss_loss = AddGaussianLoss(encodertype)

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        sample  = self.rep_gauss([mean, std])
        ### Compute the KL loss with prior as an information bottleneck
        if self.encodertype == "user":
            kl_loss = self.gauss_loss([mean, std])
            self._init_graph_network([mean, std], [sample, kl_loss])
        else:
            uemb_in = layers.Input(input_shapes[-1][1:])
            kl_loss = self.gauss_loss([mean, std, uemb_in])
            self._init_graph_network([mean, std, uemb_in], [sample, kl_loss])

        super(Sampler, self).build(input_shapes)

    def call(self, inputs):
        inputs  = generic_utils.to_list(inputs)
        [sample, kl_loss], _ = self._run_internal_graph(inputs)

        ### Remember to add the KL loss to the encoder-decoder objective
        self.add_loss(kl_loss)
        return sample


class RegressionDecoder(network.Network):
    '''
        For tasks where popularity is represented by a 
        single numerical value, MLP with MSE loss is used 
        as the variational decoder to make the popularity
        prediction based on fused hidden representation
    '''
    def __init__(self,
                 hidden_sizes,
                 activation,
                 **kwargs):
        super(RegressionDecoder, self).__init__(**kwargs)
        self.dense_list = []
        for size in hidden_sizes[:-1]:
            self.dense_list.append(
                layers.Dense(size, activation=activation)
            )
        self.dense_list.append(layers.Dense(1, activation=None))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mid = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            mid = dense(layers.Dropout(0.5)(mid))
        self._init_graph_network(x_in, mid)
        super(RegressionDecoder, self).build(input_shapes)


class VariationalEncoderDecoder(models.Model):
    '''
        Assemble the modality-specific encoder, sampler, 
        decoder defined above into the proposed multimodal 
        variational encoder-decoder (MMVED) framework
    '''
    def __init__(self,
                 enc_args,
                 dec_args,
                 input_shapes,
                 **kwargs):
        super(VariationalEncoderDecoder, self).__init__(**kwargs)
        if "user" in enc_args.keys():
            self.encodertype = "user"
            self.uindim = enc_args['user']['uindim']
            self.uembdim = enc_args['user']['uembdim']
        else:
            self.encodertype = "video"
        self.num_mods = len(enc_args)
        self.encoders = [Encoder(enc_arg, name="Encoder_{}".format(i)) \
            for i, enc_arg in enumerate(enc_args.values())]

        ### Modality-specific embeddings are fused with PoE principle
        self.poe_gaussian = ProductOfExpertGaussian()
        self.sampler = Sampler(self.encodertype, name="Sampler")
        self.decoder = RegressionDecoder(**dec_args, name="Decoder")
        self.build(input_shapes)

    def build(self, input_shapes):
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
            input_space = [uid_in] + [mods_in]
        else:
            uemb_in = layers.Input(input_shapes[0], name="uemb")
            mods_in = [layers.Input(input_shapes[i+1], name="modality_{}".format(i)) \
                       for i in range(self.num_mods)]
            mean_stds = [encoder(mod_in) for encoder, mod_in in \
                         zip(self.encoders, mods_in)]
            mean, std = self.poe_gaussian(mean_stds)
            hid_in = self.sampler([mean, std, uemb_in])
            input_space = [uemb_in] + mods_in

        pop_pred = self.decoder(hid_in)
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