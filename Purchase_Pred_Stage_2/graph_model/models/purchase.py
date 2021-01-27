# -*- coding: utf-8 -*-

from keras import layers as L
from keras import Model
from keras import backend as K
from .base_model import BaseModel
import tensorflow as tf
#from .metrics import f1_metric


class Purchase(BaseModel):
    def __init__(self, config):
        super(Purchase, self).__init__(config)
        self.build()

    def encode_layer(self, x):
        x = L.Conv1D(self.config["filters"], self.config["kernel_size"], padding="same", activation="relu")(x)
        x = L.Bidirectional(L.CuDNNLSTM(self.config["hidden_size"]))(x)
        return x

    def build(self):
        K.clear_session()
        self.model = None
        u_in = L.Input(name="user", shape=(10, ), dtype="int32")
        s_in = L.Input(name="sku", shape=(10, ), dtype="int32")
        ds_in = L.Input(name="discount", shape=(10, 11, ), dtype="float32")

        u, s, ds = u_in, s_in, ds_in
        #mask = self.mask_layer(x)
        u = self.embedding_layer(u)#, input_length = 1)
        s = self.embedding_layer(s)#, input_length = 10)
        #x = self.dropout_layer(x)
        u_h = L.Dense(self.config["latent_dim"], name="user_map")(u)#, activation="relu")(u)
        
        

        ##### Change following 2 lines to remove or include discount info
        s_h = L.Dense(self.config["latent_dim"], name="sku_map", activation="relu")(s)

        #s_h = L.Dense(self.config["latent_dim"], name="sku_map")(s) #NO DISCOUNT
        #####



        ds = L.Dense(self.config["latent_dim"], name="disc_map", activation="relu")(ds)
        s_ds =  L.Concatenate()([s_h, ds])
        s_ds_h = L.Dense(self.config["latent_dim"], name="sku_disc_map")(s_ds)
        

        ##### Change following 2 lines to remove or include discount info
        x = L.Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-1, keep_dims=False), name='first_order')([u_h, s_ds_h])

        #x = L.Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-1, keep_dims=False), name='first_order')([u_h, s_h]) #NO DISCOUNT
        #####



        #x = L.Softmax()(x)
        #mix = L.Concatenate()([u_h, s_h, ds])
        #s_ds_h = L.Dense(self.config["latent_dim"], name="sku_disc_map", activation="relu")(s_ds)
        #print("u_h", u_h)
        #print("s_ds_h", s_ds_h)
        #mix = L.Multiply()([u_h, s_ds_h])
        #mix = L.Concatenate()([u_h, s_ds_h])
        #mix = L.Dense(self.config['latent_dim'])(mix)
        #o = L.Flatten()(mix)
        #o = L.Lambda(lambda x: K.max(x, -1))(mix)
        #o = L.Dense(self.config['output_size'], activation="sigmoid")(o)


        def focal_loss(gamma=2., alpha=.25):
            def focal_loss_fixed(y_true, y_pred):
                pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            return focal_loss_fixed


        def line_loss(y_true, y_pred):
            #print("y_true:", y_true)
            #print("y_pred:", y_pred)
            #print("output", K.sigmoid(y_true*y_pred))
            return -K.mean(K.log(K.sigmoid(y_true*y_pred) + 1e-12))


        def binary_focal_loss(gamma=2, alpha=0.25):

            alpha = tf.constant(alpha, dtype=tf.float32)
            gamma = tf.constant(gamma, dtype=tf.float32)

            def binary_focal_loss_fixed(y_true, y_pred):
                """
                y_true shape need be (None,1)
                y_pred need be compute after sigmoid
                """
                y_true = tf.cast(y_true, tf.float32)
                alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)

                p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
                focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
                return K.mean(focal_loss)
            return binary_focal_loss_fixed


        self.model = Model(inputs=[u_in, s_in, ds_in], outputs=x)

        self.model.compile(
            loss=line_loss, #binary_focal_loss(gamma=2, alpha=0.25), #line_loss,#(alpha=.25, gamma=2),
            #metrics=["accuracy", f1_metric],
            optimizer=self.config["optimizer"])
        self.model.summary()

    @staticmethod
    def default_params():
        return dict(BaseModel.default_params(), **{
            "kernel_size": 3,
            "filters": 128,
            "hidden_size": 128
        })