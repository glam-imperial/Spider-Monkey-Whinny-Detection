import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Convolution2D, Flatten, MaxPooling2D,\
    BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, Concatenate,\
    Bidirectional, LSTM, TimeDistributed, Reshape, LayerNormalization, RNN, LSTMCell
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import NASNetLarge, Xception, InceptionResNetV2, MobileNetV2


def get_model(x_tf_placeholder_dict,
              support_tf_placeholder_dict,
              model_configuration):
    bottom_model = model_configuration["bottom_model"]
    bottom_model_configuration = model_configuration["bottom_model_configuration"]
    core_model = model_configuration["core_model"]
    core_model_configuration = model_configuration["core_model_configuration"]
    input_type_list = model_configuration["input_type_list"]
    global_pooling = model_configuration["global_pooling"]
    global_pooling_configuration = model_configuration["global_pooling_configuration"]

    input_placeholder_dict = dict()
    for input_type in input_type_list:
        input_placeholder_dict[input_type] = x_tf_placeholder_dict[input_type]

    # input_layer = tf.keras.Input(shape=model_configuration["name_to_metadata"]["logmel_spectrogram"]["numpy_shape"])
    # input_features = tf.keras.Input(tensor=input_placeholder)

    input_layer_list = list()

    if bottom_model == "Identity":
        input_layer_list.append(tf.keras.Input(tensor=input_placeholder_dict["logmel_spectrogram"]))
        net_train, \
        net_test = get_Identity_block(input_placeholder_dict["logmel_spectrogram"],
                                      core_model_configuration)
    elif bottom_model == "Wavegram":
        if ("waveform" in input_placeholder_dict.keys()) and \
           ("logmel_spectrogram" in input_placeholder_dict.keys()):
            input_layer_list.append(
                tf.keras.Input(shape=model_configuration["name_to_metadata"]["waveform"]["numpy_shape"]))
            input_layer_list.append(
                tf.keras.Input(shape=model_configuration["name_to_metadata"]["logmel_spectrogram"]["numpy_shape"]))
            net_train, \
            net_test = get_Wavegram_block(input_placeholder_dict,
                                          core_model_configuration)
        else:
            raise NotImplementedError
    elif bottom_model == "1DCNN":
        if "waveform" not in input_placeholder_dict.keys():
            raise ValueError("1DCNN is built for processing raw audio waveform.")
        input_layer_list.append(
                tf.keras.Input(shape=model_configuration["name_to_metadata"]["waveform"]["numpy_shape"]))
        net_train, \
        net_test = get_1DCNN_block(input_placeholder_dict["waveform"],
                                   core_model_configuration)
    elif bottom_model == "2DCNN":
        if "logmel_spectrogram" not in input_placeholder_dict.keys():
            raise ValueError("2DCNN is built for processing logmel spectrograms.")
        input_layer_list.append(
                tf.keras.Input(shape=model_configuration["name_to_metadata"]["logmel_spectrogram"]["numpy_shape"]))
        net_train, \
        net_test = get_2DCNN_block(input_placeholder_dict["logmel_spectrogram"],
                                   core_model_configuration)
    else:
        raise ValueError("Invalid")

    if core_model == "RNN":
        net_train, \
        net_test = get_rnn_block(net_train,
                                 net_test,
                                 core_model_configuration)
    elif core_model == "ResNet38_PANN":
        net_train, \
            net_test = get_ResNet38_PANN_block(net_train,
                                               net_test,
                                                     core_model_configuration)
    elif core_model == "CNN14_PANN":
        net_train, \
        net_test = get_CNN14_PANN_block(net_train,
                                        net_test,
                                        core_model_configuration)
    elif core_model == "Hong":
        net_train, \
        net_test = get_Hong_block(net_train,
                                  net_test,
                                  core_model_configuration)
    elif core_model == "VGG16":
        net_train, \
        net_test = get_VGG16_model(net_train,
                                   net_test,
                                   model_configuration)
    else:
        raise ValueError("Invalid core_model type.")

    if global_pooling == "Average":
        prediction_train, \
        prediction_test = get_average_global_pooling(net_train,
                                                       net_test,
                                                       global_pooling_configuration)
    elif global_pooling == "Max":
        prediction_train, \
        prediction_test = get_max_global_pooling(net_train,
                                                 net_test,
                                                 global_pooling_configuration)
    elif global_pooling == "AvgMax_PANN":
        prediction_train, \
        prediction_test = get_AvgMax_PANN_pooling(net_train,
                                                  net_test,
                                                  global_pooling_configuration)
    elif global_pooling == "Prediction":
        prediction_train, \
        prediction_test = get_Prediction_pooling(net_train,
                                                 net_test,
                                                 global_pooling_configuration)
    elif global_pooling == "Attention":
        prediction_train, \
        prediction_test = get_attention_global_pooling(net_train,
                                                       net_test,
                                                       global_pooling_configuration)
    elif global_pooling == "VGG16Top":
        prediction_train, \
        prediction_test = get_VGG16Top_global_pooling(net_train,
                                                      net_test,
                                                      global_pooling_configuration)
    else:
        raise ValueError("Invalid global_pooling type.")

    keras_model_train = tf.keras.Model(inputs=input_layer_list, outputs=prediction_train["whinny_single"])
    keras_model_test = tf.keras.Model(inputs=input_layer_list, outputs=prediction_test["whinny_single"])

    return prediction_train, prediction_test, keras_model_train, keras_model_test


def get_Identity_block(input_net,
                       config_dict):
    net_train = input_net
    net_test = input_net

    return net_train, net_test


def get_Wavegram_block(input_net_dict,
                       config_dict):

    input_waveform = input_net_dict["waveform"]
    input_logmel_spectrogram = input_net_dict["logmel_spectrogram"]

    waveform_layer_list = list()
    waveform_layer_list.append(Reshape((75 * 640, 1)))
    # waveform_layer_list.append(tf.keras.layers.Conv1D(filters=64, kernel_size=11, strides=5, padding='valid',
    #                                          data_format='channels_last', dilation_rate=1,
    #                                          activation=None, use_bias=False, kernel_initializer='glorot_uniform'))
    # waveform_layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))

    waveform_layer_list.append(ConvPreWavBlock(64, kernel_size=10, pool=16))
    # waveform_layer_list.append(ConvPreWavBlock(64, kernel_size=3, pool=4))
    waveform_layer_list.append(ConvPreWavBlock(64, kernel_size=3, pool=10))
    waveform_layer_list.append(Reshape((300, 64, 1)))
    waveform_layer_list.append(ConvBlock(32, pool=(2, 1), num_layers=1))

    waveform_layer_list.append(Reshape((150, 64, 32)))

    mel_layer_list = list()
    mel_layer_list.append(Reshape((300,
                                   128,
                                   1)))
    mel_layer_list.append(ConvBlock(64, pool=(2, 2), num_layers=2))

    concat_layer = tf.keras.layers.Concatenate(axis=-1)

    waveform_net_train = waveform_layer_list[0](input_waveform, training=True)
    for l in waveform_layer_list[1:]:
        waveform_net_train = l(waveform_net_train, training=True)

    waveform_net_test = waveform_layer_list[0](input_waveform, training=False)
    for l in waveform_layer_list[1:]:
        waveform_net_test = l(waveform_net_test, training=False)

    mel_net_train = mel_layer_list[0](input_logmel_spectrogram, training=True)
    for l in mel_layer_list[1:]:
        mel_net_train = l(mel_net_train, training=True)

    mel_net_test = mel_layer_list[0](input_logmel_spectrogram, training=False)
    for l in mel_layer_list[1:]:
        mel_net_test = l(mel_net_test, training=False)

    net_train = concat_layer([waveform_net_train, mel_net_train])
    net_test = concat_layer([waveform_net_test, mel_net_test])

    return net_train, net_test


class ConvPreWavBlock():
    def __init__(self, filters, kernel_size, pool):
        self.layer_list = list()
        self.layer_list.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                                             data_format='channels_last', dilation_rate=1,
                                             activation=None, use_bias=False, kernel_initializer='glorot_uniform'))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        # self.layer_list.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
        #                                          data_format='channels_last', dilation_rate=2,
        #                                          activation=None, use_bias=False, kernel_initializer='glorot_uniform'))
        # self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.MaxPool1D(pool_size=pool, strides=pool, padding='valid'))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


class ConvBlock():
    def __init__(self, filters, pool, num_layers):
        self.layer_list = list()
        for n in range(num_layers):
            self.layer_list.append(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       data_format='channels_last', dilation_rate=(1, 1),
                                       activation=None, use_bias=False, kernel_initializer='glorot_uniform'))
            self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))

        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=pool, strides=pool, padding='valid'))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


def get_1DCNN_block(input_net,
                    config_dict):
    layer_list = list()
    layer_list.append(Reshape((75 * 640, 1)))
    layer_list.append(tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, padding='same',
                                             data_format='channels_last', dilation_rate=1,
                                             activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.MaxPool1D(pool_size=10, strides=10, padding='valid'))

    layer_list.append(tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=1, padding='same',
                                             data_format='channels_last', dilation_rate=1,
                                             activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.MaxPool1D(pool_size=8, strides=8, padding='valid'))

    layer_list.append(tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=1, padding='same',
                                             data_format='channels_last', dilation_rate=1,
                                             activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.MaxPool1D(pool_size=8, strides=8, padding='valid'))

    layer_list.append(Reshape((75,
                               128)))

    net_train = layer_list[0](input_net, training=True)
    for l in layer_list[1:]:
        net_train = l(net_train, training=True)

    net_test = layer_list[0](input_net, training=False)
    for l in layer_list[1:]:
        net_test = l(net_test, training=False)

    return net_train, net_test


def get_Hong_block(net_train,
                   net_test,
                   config_dict):
    # batch_size = config_dict["batch_size"]

    # _, _, num_features = waveform_input.get_shape().as_list()
    # net = tf.reshape(waveform_input, [batch_size, -1, num_features, 1])

    layer_list = list()
    layer_list.append(Reshape((300,
                                   128,
                                   1)))
    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))


    layer_list.append(Reshape((18,
                               1,
                               8 * 128)))

    net_train = layer_list[0](net_train, training=True)
    for l in layer_list[1:]:
        net_train = l(net_train, training=True)

    net_test = layer_list[0](net_test, training=False)
    for l in layer_list[1:]:
        net_test = l(net_test, training=False)

    return net_train, net_test


def get_ResNet38_PANN_block(net_train,
                            net_test,
                            config_dict):
    use_se = config_dict["use_se"]

    resnet38_pann = ResNet38_PANN(config_dict, senet=use_se, from_wavegram=False)

    net_train = resnet38_pann(net_train, training=True)
    net_test = resnet38_pann(net_test, training=False)

    return net_train, net_test


class ResNet38_PANN():
    def __init__(self, config_dict, senet, from_wavegram):
        self.senet = senet

        self.layer_list = list()

        if not from_wavegram:
            self.layer_list.append(Reshape((300,
                                            128,
                                            1)))
            self.layer_list.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                 dilation_rate=(1, 1), activation=None,
                                                 use_bias=False, kernel_initializer='glorot_uniform'))
            # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
            self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
            self.layer_list.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                 dilation_rate=(1, 1), activation=None,
                                                 use_bias=False, kernel_initializer='glorot_uniform'))
            # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
            self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
            self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        else:
            self.layer_list.append(Reshape((150,
                                            64,
                                            96)))

        # self.layer_list.append(ResBlock(64))
        self.layer_list.append(ResBlock(64, senet=self.senet))
        self.layer_list.append(ResBlock(64, senet=self.senet))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # self.layer_list.append(ResBlock(128))
        self.layer_list.append(ResBlock(128, senet=self.senet))
        self.layer_list.append(ResBlock(128, senet=self.senet))
        self.layer_list.append(ResBlock(128, senet=self.senet))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # self.layer_list.append(ResBlock(256))
        self.layer_list.append(ResBlock(256, senet=self.senet))
        self.layer_list.append(ResBlock(256, senet=self.senet))
        self.layer_list.append(ResBlock(256, senet=self.senet))
        self.layer_list.append(ResBlock(256, senet=self.senet))
        self.layer_list.append(ResBlock(256, senet=self.senet))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # self.layer_list.append(ResBlock(512))
        self.layer_list.append(ResBlock(512, senet=self.senet))
        self.layer_list.append(ResBlock(512, senet=self.senet))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=2048, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None,
                                                      use_bias=False, kernel_initializer='glorot_uniform'))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=2048, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None,
                                                      use_bias=False, kernel_initializer='glorot_uniform'))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))

        self.layer_list.append(Reshape((9,
                                        1,
                                        4 * 2048)))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


class ResBlock():
    def __init__(self, filters, senet, ratio=4):
        self.senet = senet

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                 dilation_rate=(1, 1), activation=None,
                                                 use_bias=False, kernel_initializer='glorot_uniform')
        # self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
        # self.dropout_1 = tf.keras.Dropout(rate=0.1)

        self.conv2d_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                          dilation_rate=(1, 1), activation=None,
                                          use_bias=False, kernel_initializer='glorot_uniform')
        # self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_2 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.conv_res = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                                 dilation_rate=(1, 1), activation=None,
                                                 use_bias=False, kernel_initializer='glorot_uniform')
        # self.bn_res = tf.keras.layers.BatchNormalization(axis=-1)

        if self.senet:
            self.se_pooling = tf.keras.layers.GlobalAveragePooling2D()
            self.se_fc_1 = tf.keras.layers.Dense(filters // ratio)
            self.se_relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.se_fc_2 = tf.keras.layers.Dense(filters)
            self.se_sigmoid_1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            self.se_reshape = tf.keras.layers.Reshape((1, 1, filters))

    def __call__(self, x, training):
        cnv2d_1 = self.conv2d_1(x, training=training)
        # bn_1 = self.bn_1(cnv2d_1, training=training)
        relu_1 = self.relu_1(cnv2d_1, training=training)
        # do_1 = self.dropout_1(relu_1, training=training)

        cnv2d_2 = self.conv2d_2(relu_1, training=training)
        # bn_2 = self.bn_1(cnv2d_2, training=training)

        res = self.conv_res(x, training=training)
        # res = self.bn_res(res, training=training)

        bn_2 = cnv2d_2 + res

        relu_2 = self.relu_1(bn_2, training=training)

        if self.senet:
            squeeze = self.se_pooling(relu_2, training=training)
            excitation = self.se_fc_1(squeeze, training=training)
            excitation = self.se_relu_1(excitation, training=training)
            excitation = self.se_fc_2(excitation, training=training)
            excitation = self.se_sigmoid_1(excitation, training=training)
            excitation = self.se_reshape(excitation, training=training)

            relu_2 = relu_2 * excitation

        return relu_2


def get_CNN14_PANN_block(net_train,
                         net_test,
                         config_dict):
    layer_list = list()
    layer_list.append(Reshape((300,
                               128,
                               1)))

    layer_list.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 dilation_rate=(1, 1), activation=None,
                                 use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=2048, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
    layer_list.append(tf.keras.layers.Conv2D(filters=2048, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))

    layer_list.append(Reshape((9,
                               1,
                               4 * 2048)))

    net_train = layer_list[0](net_train, training=True)
    for l in layer_list[1:]:
        net_train = l(net_train, training=True)

    net_test = layer_list[0](net_test, training=False)
    for l in layer_list[1:]:
        net_test = l(net_test, training=False)

    return net_train, net_test


def get_2DCNN_block(net_input,
                    config_dict):
    layer_list = list()
    layer_list.append(Reshape((300,
                               128,
                               1)))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(5, 1), strides=(5, 1), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'))

    layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=tf.nn.relu,
                                             use_bias=True, kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
    # layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
    layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))

    layer_list.append(Reshape((29,
                               128 * 128)))

    net_train = layer_list[0](net_input, training=True)
    for l in layer_list[1:]:
        net_train = l(net_train, training=True)

    net_test = layer_list[0](net_input, training=False)
    for l in layer_list[1:]:
        net_test = l(net_test, training=False)

    return net_train, net_test


def get_rnn_block(net_train,
                  net_test,
                  config_dict):
    cells = list()
    for l_no in range(2):
        cells.append(LSTMCell(512))

    rnn = RNN(cells, return_sequences=True)
    # reshape_layer = Reshape((75, 1, 512))
    # reshape_layer = Reshape((29, 1, 512))
    reshape_layer = Reshape((-1, 1, 512))

    net_train = rnn(net_train, training=True)
    net_test = rnn(net_test, training=False)

    net_train = reshape_layer(net_train, training=True)
    net_test = reshape_layer(net_test, training=False)

    return net_train, net_test


def get_InceptionResNet_model(input_features,
                              config_dict):
    layer_list = list()
    layer_list.append(tf.keras.layers.Reshape((300,
                                               128,
                                               1)))
    layer_list.append(tf.keras.layers.Concatenate(axis=-1))
    layer_list.append(InceptionResNetV2(include_top=False,
                                  # weights="imagenet",
                                  weights=None,
                                  input_shape=(300, 128, 3)))

    net_train = layer_list[0](input_features, training=True)
    net_test = layer_list[0](input_features, training=False)

    net_train = layer_list[1]([net_train,
                               net_train,
                               net_train], training=True)
    net_test = layer_list[1]([net_test,
                              net_test,
                              net_test], training=False)

    net_train = layer_list[2](net_train, training=True)
    net_test = layer_list[2](net_test, training=False)

    # net = tf.keras.layers.Reshape((300,
    #                                128,
    #                                1))(input_features)
    # net = tf.keras.layers.Concatenate(axis=-1)([net,
    #                                             net,
    #                                             net])
    #
    # net = MobileNetV2(include_top=False,
    #             weights=None,
    #             input_shape=(300, 128, 3))(net)

    return net_train, net_test


def get_MobileNet_model(input_features,
                    config_dict):
    layer_list = list()
    layer_list.append(tf.keras.layers.Reshape((300,
                                               128,
                                               1)))
    layer_list.append(tf.keras.layers.Concatenate(axis=-1))
    layer_list.append(MobileNetV2(include_top=False,
                # weights="imagenet",
                weights=None,
                input_shape=(300, 128, 3)))

    net_train = layer_list[0](input_features, training=True)
    net_test = layer_list[0](input_features, training=False)

    net_train = layer_list[1]([net_train,
                               net_train,
                               net_train], training=True)
    net_test = layer_list[1]([net_test,
                              net_test,
                              net_test], training=False)

    net_train = layer_list[2](net_train, training=True)
    net_test = layer_list[2](net_test, training=False)

    # net = tf.keras.layers.Reshape((300,
    #                                128,
    #                                1))(input_features)
    # net = tf.keras.layers.Concatenate(axis=-1)([net,
    #                                             net,
    #                                             net])
    #
    # net = MobileNetV2(include_top=False,
    #             weights=None,
    #             input_shape=(300, 128, 3))(net)

    return net_train, net_test


def get_VGG16_model(input_features,
                             config_dict,
                    model_configuration):
    vgg16 = VGG16(config_dict)

    net_train = vgg16(input_features, training=True)
    net_test = vgg16(input_features, training=False)

    return net_train, net_test


class VGG16():
    def __init__(self, config_dict):
        self.layer_list = list()

        self.layer_list.append(Reshape((300,
                               128,
                               1)))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                 dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                                      dilation_rate=(1, 1), activation=None))
        # self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.layer_list.append(Reshape((9,
                                   1, 4 * 512)))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def get_Prediction_pooling(net_train,
                           net_test,
                           global_pooling_configuration):
    pooling_type = global_pooling_configuration["pooling_type"]  # ["max", "linsoft", "softmax", "attention", "power"]
    softmax_scaling = global_pooling_configuration["softmax_scaling"]  # ["Auto", "Uniform"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]
    # gumbel_train = global_pooling_configuration["gumbel_train"]
    max_test = global_pooling_configuration["max_test"]

    constant_float = 1e-5

    frame_dense_layer = Dense(2)

    net_train = tf.reshape(net_train, (-1, number_of_features))
    net_test = tf.reshape(net_test, (-1, number_of_features))

    logits_single_train = frame_dense_layer(net_train, training=True)
    logits_single_test = frame_dense_layer(net_test, training=False)

    logits_single_train = tf.reshape(logits_single_train, (-1, sequence_length, 2))
    logits_single_test = tf.reshape(logits_single_test, (-1, sequence_length, 2))

    probs_single_train = tf.math.sigmoid(logits_single_train)
    probs_single_test = tf.math.sigmoid(logits_single_test)

    # if gumbel_train:
    #     probs_single_train = probs_single_train + sample_gumbel(tf.shape(probs_single_train))

    if pooling_type == "max":
        max_pool_layer = tf.keras.layers.GlobalMaxPooling1D()

        # if gumbel_train:
        #     auto_pool_variable = tf.constant(1.0, dtype=tf.float32)
        #     denom_train = auto_pool_variable * probs_single_train
        #     denom_train = tf.reduce_sum(denom_train,
        #                                 axis=1,
        #                                 keep_dims=True) + tf.abs(tf.reduce_min(denom_train, axis=1,
        #                                                                        keep_dims=True)) + \
        #                   tf.constant(constant_float, dtype=tf.float32)
        #
        #     prediction_single_train = tf.divide(tf.reduce_sum(auto_pool_variable * tf.math.pow(probs_single_train,
        #                                                                                        2),
        #                                                       axis=1,
        #                                                       keep_dims=True), denom_train)
        #
        #     prediction_single_train = tf.reshape(prediction_single_train, (-1, 2))
        #     k = tf.shape(probs_single_train)[-1]
        #     y_hard = tf.cast(tf.equal(prediction_single_train, tf.reduce_max(prediction_single_train, 1, keep_dims=True)),
        #                      prediction_single_train.dtype)
        #     prediction_single_train = tf.stop_gradient(y_hard - prediction_single_train) + prediction_single_train
        # else:
        #     prediction_single_train = max_pool_layer(probs_single_train, training=True)
        prediction_single_train = max_pool_layer(probs_single_train, training=True)
        prediction_single_test = max_pool_layer(probs_single_test, training=False)

    elif pooling_type in ["linsoft", "power", "softmax", "attention"]:
        if softmax_scaling == "Auto":
            auto_pool_variable = tf.Variable(1.0, dtype=tf.float32)
        elif softmax_scaling == "Uniform":
            auto_pool_variable = tf.constant(1.0, dtype=tf.float32)
        else:
            raise ValueError("Invalid softmax scaling type.")

        if pooling_type == "linsoft":
            denom_train = auto_pool_variable * probs_single_train
            denom_train = tf.reduce_sum(denom_train,
                                        axis=1,
                                        keep_dims=True) + tf.abs(tf.reduce_min(denom_train, axis=1,
                                        keep_dims=True)) +\
                          tf.constant(constant_float, dtype=tf.float32)

            prediction_single_train = tf.divide(tf.reduce_sum(auto_pool_variable * tf.math.pow(probs_single_train,
                                                                                     2),
                                                    axis=1,
                                        keep_dims=True), denom_train)

            prediction_single_train = tf.reshape(prediction_single_train, (-1, 2))

            # denom_test = auto_pool_variable * probs_single_test
            # denom_test = tf.reduce_sum(denom_test,
            #                            axis=1,
            #                            keep_dims=True) + tf.abs(tf.reduce_min(denom_test, axis=1,
            #                                                                   keep_dims=True)) + \
            #              tf.constant(constant_float, dtype=tf.float32)
            # prediction_single_test = tf.divide(tf.reduce_sum(auto_pool_variable * tf.math.pow(probs_single_test,
            #                                                                                   2),
            #                                                  axis=1,
            #                                                  keep_dims=True), denom_test)
            # prediction_single_test = tf.reshape(prediction_single_test, (-1, 2))

            if max_test:
                max_pool_layer = tf.keras.layers.GlobalMaxPooling1D()

                prediction_single_test = max_pool_layer(probs_single_test, training=False)
            else:
                denom_test = auto_pool_variable * probs_single_test
                denom_test = tf.reduce_sum(denom_test,
                                           axis=1,
                                           keep_dims=True) + tf.abs(tf.reduce_min(denom_test, axis=1,
                                                                                  keep_dims=True)) + \
                             tf.constant(constant_float, dtype=tf.float32)
                prediction_single_test = tf.divide(tf.reduce_sum(auto_pool_variable * tf.math.pow(probs_single_test,
                                                                                                  2),
                                                                 axis=1,
                                                                 keep_dims=True), denom_test)
                prediction_single_test = tf.reshape(prediction_single_test, (-1, 2))

        elif pooling_type == "power":
            if softmax_scaling != "Auto":
                raise ValueError("Need to have an adaptive parameter for power pooling.")
            denom_train = tf.math.pow(probs_single_train,
                                      auto_pool_variable)
            denom_train = tf.reduce_sum(denom_train,
                                        axis=1,
                                        keep_dims=True) + tf.abs(tf.reduce_min(denom_train, axis=1,
                                                                       keep_dims=True)) +\
                          tf.constant(constant_float, dtype=tf.float32)

            prediction_single_train = tf.divide(tf.reduce_sum(tf.multiply(probs_single_train,
                                                                tf.math.pow(probs_single_train,
                                                                auto_pool_variable)),
                                                    axis=1,
                                                              keep_dims=True), denom_train)

            prediction_single_train = tf.reshape(prediction_single_train, (-1, 2))

            # denom_test = tf.math.pow(probs_single_test,
            #                          auto_pool_variable)
            # denom_test = tf.reduce_sum(denom_test,
            #                            axis=1,
            #                            keep_dims=True) + tf.abs(tf.reduce_min(denom_test,
            #                                                                   axis=1, keep_dims=True)) + \
            #              tf.constant(constant_float, dtype=tf.float32)
            # prediction_single_test = tf.divide(tf.reduce_sum(tf.multiply(probs_single_test,
            #                                                              tf.math.pow(probs_single_test,
            #                                                                          auto_pool_variable)),
            #                                                  axis=1,
            #                                                  keep_dims=True), denom_test)
            # prediction_single_test = tf.reshape(prediction_single_test, (-1, 2))

            if max_test:
                max_pool_layer = tf.keras.layers.GlobalMaxPooling1D()

                prediction_single_test = max_pool_layer(probs_single_test, training=False)
            else:
                denom_test = tf.math.pow(probs_single_test,
                                         auto_pool_variable)
                denom_test = tf.reduce_sum(denom_test,
                                           axis=1,
                                           keep_dims=True) + tf.abs(tf.reduce_min(denom_test,
                                                                                  axis=1, keep_dims=True)) + \
                             tf.constant(constant_float, dtype=tf.float32)
                prediction_single_test = tf.divide(tf.reduce_sum(tf.multiply(probs_single_test,
                                                                             tf.math.pow(probs_single_test,
                                                                                         auto_pool_variable)),
                                                                 axis=1,
                                                                 keep_dims=True), denom_test)
                prediction_single_test = tf.reshape(prediction_single_test, (-1, 2))

        elif pooling_type == "softmax":
            raise NotImplementedError
        elif pooling_type == "attention":
            raise NotImplementedError
        else:
            raise ValueError("Invalid pooling type.")

    else:
        raise ValueError("Invalid pooling type.")

    prediction_single_train = tf.reshape(prediction_single_train[:, 1], (-1, 1))
    prediction_single_train = tf.where(prediction_single_train < constant_float,
                                        prediction_single_train + constant_float,
                                        prediction_single_train)
    prediction_single_train = tf.where(prediction_single_train > 1 - constant_float,
                                        prediction_single_train - constant_float,
                                        prediction_single_train)

    prediction_single_test = tf.reshape(prediction_single_test[:, 1], (-1, 1))
    # prediction_single_test = tf.where(prediction_single_test < constant_float,
    #                                    prediction_single_test + constant_float,
    #                                    prediction_single_test)
    # prediction_single_test = tf.where(prediction_single_test > 1 - constant_float,
    #                                    prediction_single_test - constant_float,
    #                                    prediction_single_test)

    one_constant = tf.constant(1.0, dtype=tf.float32)

    prediction_single_train = tf.concat([prediction_single_train,
                                         one_constant - prediction_single_train], axis=1)
    prediction_single_test = tf.concat([prediction_single_test,
                                        one_constant - prediction_single_test], axis=1)

    prediction_train = dict()
    prediction_train["whinny_single"] = prediction_single_train
    prediction_train["whinny_continuous"] = prediction_single_train
    prediction_train["attention_weights"] = prediction_single_train

    prediction_test = dict()
    prediction_test["whinny_single"] = prediction_single_test
    prediction_test["whinny_continuous"] = prediction_single_train
    prediction_test["attention_weights"] = prediction_single_train

    return prediction_train, prediction_test


def get_AvgMax_PANN_pooling(net_train,
                            net_test,
                            global_pooling_configuration):
    avg_pool_layer = GlobalAveragePooling2D()
    max_pool_layer = GlobalMaxPool2D()
    dense_layer = Dense(2,
                        name='prediction_single')

    net_single_train_avg = avg_pool_layer(net_train, training=True)
    net_single_test_avg = avg_pool_layer(net_test, training=False)

    net_single_train_max = max_pool_layer(net_train, training=True)
    net_single_test_max = max_pool_layer(net_test, training=False)

    net_single_train = net_single_train_avg + net_single_train_max
    net_single_test = net_single_test_avg + net_single_test_max

    prediction_single_train = dense_layer(net_single_train, training=True)
    prediction_single_test = dense_layer(net_single_test, training=False)

    # net_single = GlobalAveragePooling2D()(net)
    # prediction_single = Dense(2,
    #             name='prediction_single')(net_single)
    #
    net_continuous_train = prediction_single_train
    net_continuous_test = prediction_single_test

    prediction_train = dict()
    prediction_train["whinny_single"] = prediction_single_train
    prediction_train["whinny_continuous"] = net_continuous_train
    prediction_train["attention_weights"] = net_continuous_train

    prediction_test = dict()
    prediction_test["whinny_single"] = prediction_single_test
    prediction_test["whinny_continuous"] = net_continuous_test
    prediction_test["attention_weights"] = net_continuous_test

    return prediction_train, prediction_test


def get_max_global_pooling(net_train,
                           net_test,
                           global_pooling_configuration):
    max_pool_layer = GlobalMaxPool2D()
    dense_layer = Dense(2,
                name='prediction_single')

    net_single_train = max_pool_layer(net_train, training=True)
    net_single_test = max_pool_layer(net_test, training=False)

    prediction_single_train = dense_layer(net_single_train, training=True)
    prediction_single_test = dense_layer(net_single_test, training=False)

    net_continuous_train = prediction_single_train
    net_continuous_test = prediction_single_test

    prediction_train = dict()
    prediction_train["whinny_single"] = prediction_single_train
    prediction_train["whinny_continuous"] = net_continuous_train
    prediction_train["attention_weights"] = net_continuous_train

    prediction_test = dict()
    prediction_test["whinny_single"] = prediction_single_test
    prediction_test["whinny_continuous"] = net_continuous_test
    prediction_test["attention_weights"] = net_continuous_test

    return prediction_train, prediction_test


def get_average_global_pooling(net_train,
                               net_test,
                               global_pooling_configuration):
    use_temporal_std = global_pooling_configuration["use_temporal_std"]
    pool_heads = global_pooling_configuration["pool_heads"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]

    avg_pool_layer = GlobalAveragePooling2D()

    dense_layer = Dense(2,
                name='prediction_single')

    mean_hidden_train = avg_pool_layer(net_train, training=True)
    mean_hidden_test = avg_pool_layer(net_test, training=False)

    mean_hidden_train = tf.reshape(mean_hidden_train, (-1, 1, number_of_features, 1))
    mean_hidden_test = tf.reshape(mean_hidden_test, (-1, 1, number_of_features, 1))

    net_train = tf.reshape(net_train, (-1, sequence_length, number_of_features, 1))
    net_test = tf.reshape(net_test, (-1, sequence_length, number_of_features, 1))

    if use_temporal_std:
        a_train = tf.multiply(net_train, net_train)
        a_test = tf.multiply(net_test, net_test)

        b_train = tf.multiply(mean_hidden_train, mean_hidden_train)
        b_test = tf.multiply(mean_hidden_test, mean_hidden_test)

        std_hidden_train = tf.reduce_mean(a_train, axis=1, keep_dims=True) - b_train
        std_hidden_train = std_hidden_train + tf.abs(tf.reduce_min(std_hidden_train, axis=1, keep_dims=True)) + tf.constant(0.000001,
                                                                                                    dtype=tf.float32)
        std_hidden_train = tf.math.sqrt(std_hidden_train)

        std_hidden_test = tf.reduce_mean(a_test, axis=1, keep_dims=True) - b_test
        std_hidden_test = std_hidden_test + tf.abs(tf.reduce_min(std_hidden_test, axis=1, keep_dims=True)) + tf.constant(0.000001,
                                                                                                 dtype=tf.float32)
        std_hidden_test = tf.math.sqrt(std_hidden_test)

    if pool_heads == "gating":
        mean_glu_layer = tf.keras.layers.Dense(number_of_features)
        if use_temporal_std:
            std_glu_layer = tf.keras.layers.Dense(number_of_features)
        mean_hidden_train = tf.reshape(mean_hidden_train, (-1, number_of_features))
        mean_hidden_test = tf.reshape(mean_hidden_test, (-1, number_of_features))
        if use_temporal_std:
            std_hidden_train = tf.reshape(std_hidden_train, (-1, number_of_features))
            std_hidden_test = tf.reshape(std_hidden_test, (-1, number_of_features))

            mean_hidden_train = tf.multiply(tf.math.sigmoid(mean_glu_layer(mean_hidden_train,
                                                                                   training=True)),
                                                         mean_hidden_train)
            mean_hidden_test = tf.multiply(tf.math.sigmoid(mean_glu_layer(mean_hidden_test,
                                                                                  training=False)),
                                                        mean_hidden_test)
            if use_temporal_std:
                std_hidden_train = tf.multiply(tf.math.sigmoid(std_glu_layer(std_hidden_train,
                                                                                            training=True)),
                                                             std_hidden_train)
                std_hidden_test = tf.multiply(tf.math.sigmoid(std_glu_layer(std_hidden_test,
                                                                                           training=False)),
                                                            std_hidden_test)

                mean_hidden_train = tf.concat([mean_hidden_train,
                                                            std_hidden_train], axis=1)
                mean_hidden_test = tf.concat([mean_hidden_test,
                                                           std_hidden_test], axis=1)
    elif pool_heads == "attention":
        raise NotImplementedError
    elif pool_heads == "no_pool":
        mean_hidden_train = tf.reshape(mean_hidden_train, (-1, number_of_features))
        mean_hidden_test = tf.reshape(mean_hidden_test, (-1, number_of_features))
        if use_temporal_std:
            std_hidden_train = tf.reshape(std_hidden_train, (-1, number_of_features))
            std_hidden_test = tf.reshape(std_hidden_test, (-1, number_of_features))
            mean_hidden_train = tf.concat([mean_hidden_train,
                                                            std_hidden_train], axis=1)
            mean_hidden_test = tf.concat([mean_hidden_test,
                                                           std_hidden_test], axis=1)
    else:
        raise ValueError("Invalid head pooling method.")

    if use_temporal_std:
        mean_hidden_train = tf.reshape(mean_hidden_train, (-1, 2 * number_of_features))
        mean_hidden_test = tf.reshape(mean_hidden_test, (-1, 2 * number_of_features))
    else:
        mean_hidden_train = tf.reshape(mean_hidden_train, (-1, number_of_features))
        mean_hidden_test = tf.reshape(mean_hidden_test, (-1, number_of_features))

    prediction_single_train = dense_layer(mean_hidden_train, training=True)
    prediction_single_test = dense_layer(mean_hidden_test, training=False)

    net_continuous_train = prediction_single_train
    net_continuous_test = prediction_single_test

    prediction_train = dict()
    prediction_train["whinny_single"] = prediction_single_train
    prediction_train["whinny_continuous"] = net_continuous_train
    prediction_train["attention_weights"] = net_continuous_train

    prediction_test = dict()
    prediction_test["whinny_single"] = prediction_single_test
    prediction_test["whinny_continuous"] = net_continuous_test
    prediction_test["attention_weights"] = net_continuous_test

    return prediction_train, prediction_test


def get_attention_global_pooling(net_train,
                                 net_test,
                                 global_pooling_configuration):
    number_of_heads = global_pooling_configuration["number_of_heads"]
    use_temporal_std = global_pooling_configuration["use_temporal_std"]
    pool_heads = global_pooling_configuration["pool_heads"]
    auto_pooling = global_pooling_configuration["auto_pooling"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]
    use_auto_array = global_pooling_configuration["use_auto_array"]
    # gumbel_train = global_pooling_configuration["gumbel_train"]
    # number_of_features = 4 * 2048
    # sequence_length = 9

    net_train = tf.reshape(net_train, (-1, number_of_features))  # [bs*sequence_length, number_of_features]
    net_test = tf.reshape(net_test, (-1, number_of_features))  # [bs*sequence_length, number_of_features]

    heads_list = list()
    if use_auto_array:
        for head_i in range(number_of_heads):
            heads_list.append(Dense(number_of_features, use_bias=False))
    else:
        for head_i in range(number_of_heads):
            heads_list.append(Dense(1, use_bias=False))

    energy_list_train = list()
    energy_list_test = list()
    for head_i in range(number_of_heads):
        if use_auto_array:
            energy_train = tf.reshape(heads_list[head_i](net_train, training=True), (-1, sequence_length, number_of_features, 1))
        else:
            energy_train = tf.reshape(heads_list[head_i](net_train, training=True), (-1, sequence_length, 1, 1))
        # if gumbel_train:
        #     energy_train = energy_train + sample_gumbel(tf.shape(energy_train))
        # else:
        #     energy_train

        energy_list_train.append(energy_train)
        if use_auto_array:
            energy_test = tf.reshape(heads_list[head_i](net_test, training=False), (-1, sequence_length, number_of_features, 1))
        else:
            energy_test = tf.reshape(heads_list[head_i](net_test, training=False), (-1, sequence_length, 1, 1))
        energy_list_test.append(energy_test)

    if auto_pooling == "Auto":
        auto_pool_variable_list = list()

        attention_weights_list_train = list()
        attention_weights_list_test = list()
        for head_i in range(number_of_heads):
            auto_init = 1 / max(1., np.floor((float(head_i) - 1.) / 2.) * 5)
            auto_pool_variable_list.append(tf.Variable(auto_init, dtype=tf.float32))
            attention_weights_list_train.append(
                tf.nn.softmax(auto_pool_variable_list[head_i] * energy_list_train[head_i], axis=1))
            attention_weights_list_test.append(
                tf.nn.softmax(auto_pool_variable_list[head_i] * energy_list_test[head_i], axis=1))

        if number_of_heads > 1:
            attention_weights_train = tf.concat(attention_weights_list_train, axis=3)  # [-1, sequence_length, 1, number_of_heads]
            attention_weights_test = tf.concat(attention_weights_list_test, axis=3)  # [-1, sequence_length, 1, number_of_heads]
        elif number_of_heads == 1:
            attention_weights_train = attention_weights_list_train[0]  # [-1, sequence_length, 1, 1]
            attention_weights_test = attention_weights_list_test[0]  # [-1, sequence_length, 1, 1]
        else:
            raise ValueError("Invalid number of heads.")
    elif auto_pooling == "MultiResolution":
        if number_of_heads == 1:
            raise ValueError("One head does not make sense for multiresolution prediction pooling.")
        auto_pool_constant_list = list()

        attention_weights_list_train = list()
        attention_weights_list_test = list()
        for head_i in range(number_of_heads):
            auto_init = 1 / max(1., np.floor((float(head_i) - 1.) / 2.) * 5)
            auto_pool_constant_list.append(tf.constant(auto_init, dtype=tf.float32))

            attention_weights_list_train.append(
                tf.nn.softmax(auto_pool_constant_list[head_i] * energy_list_train[head_i], axis=1))
            attention_weights_list_test.append(
                tf.nn.softmax(auto_pool_constant_list[head_i] * energy_list_test[head_i], axis=1))

        if number_of_heads > 1:
            attention_weights_train = tf.concat(attention_weights_list_train, axis=3)  # [-1, sequence_length, 1, number_of_heads]
            attention_weights_test = tf.concat(attention_weights_list_test, axis=3)  # [-1, sequence_length, 1, number_of_heads]
        elif number_of_heads == 1:
            attention_weights_train = attention_weights_list_train[0]  # [-1, sequence_length, 1, 1]
            attention_weights_test = attention_weights_list_test[0]  # [-1, sequence_length, 1, 1]
        else:
            raise ValueError("Invalid number of heads.")
    elif auto_pooling == "no_auto":
        attention_weights_list_train = list()
        attention_weights_list_test = list()
        for head_i in range(number_of_heads):
            attention_weights_list_train.append(tf.nn.softmax(energy_list_train[head_i], axis=1))
            attention_weights_list_test.append(tf.nn.softmax(energy_list_test[head_i], axis=1))

        if number_of_heads > 1:
            attention_weights_train = tf.concat(attention_weights_list_train, axis=3)  # [-1, sequence_length, 1, number_of_heads]
            attention_weights_test = tf.concat(attention_weights_list_test, axis=3)  # [-1, sequence_length, 1, number_of_heads]
        elif number_of_heads == 1:
            attention_weights_train = attention_weights_list_train[0]  # [-1, sequence_length, 1, 1]
            attention_weights_test = attention_weights_list_test[0]  # [-1, sequence_length, 1, 1]
        else:
            raise ValueError("Invalid number of heads.")

    else:
        raise ValueError("Invalid auto pooling type.")

    net_train = tf.reshape(net_train, (-1, sequence_length, number_of_features, 1))  # [bs, sequence_length, number_of_features, 1]
    net_test = tf.reshape(net_test, (-1, sequence_length, number_of_features, 1))  # [bs, sequence_length, number_of_features, 1]

    mean_hidden_list_train = list()
    mean_hidden_list_test = list()
    for head_i in range(number_of_heads):
        mean_hidden_train = tf.reduce_sum(tf.multiply(net_train, attention_weights_list_train[head_i]), axis=1, keep_dims=True)
        mean_hidden_test = tf.reduce_sum(tf.multiply(net_test, attention_weights_list_test[head_i]), axis=1, keep_dims=True)
        mean_hidden_list_train.append(mean_hidden_train)  # [bs, 1, number_of_features, 1]
        mean_hidden_list_test.append(mean_hidden_test)  # [bs, 1, number_of_features, 1]

    if use_temporal_std:
        std_hidden_list_train = list()
        std_hidden_list_test = list()
        for head_i in range(number_of_heads):
            a_train = tf.multiply(tf.multiply(net_train, net_train),
                                  attention_weights_list_train[head_i])
            a_test = tf.multiply(tf.multiply(net_test, net_test),
                                 attention_weights_list_test[head_i])

            b_train = tf.multiply(mean_hidden_list_train[head_i],
                                  mean_hidden_list_train[head_i])
            b_test = tf.multiply(mean_hidden_list_test[head_i],
                                 mean_hidden_list_test[head_i])

            std_hidden_train = tf.reduce_sum(a_train, axis=1, keep_dims=True) - b_train
            std_hidden_train = std_hidden_train + tf.abs(tf.reduce_min(std_hidden_train)) + tf.constant(0.000001, dtype=tf.float32)
            std_hidden_train = tf.math.sqrt(std_hidden_train)
            std_hidden_test = tf.reduce_sum(a_test, axis=1, keep_dims=True) - b_test
            std_hidden_test = std_hidden_test + tf.abs(tf.reduce_min(std_hidden_test)) + tf.constant(0.000001, dtype=tf.float32)
            std_hidden_test = tf.math.sqrt(std_hidden_test)
            std_hidden_list_train.append(std_hidden_train)  # [bs, 1, number_of_features, 1]
            std_hidden_list_test.append(std_hidden_test)  # [bs, 1, number_of_features, 1]
            # mean_hidden_list_train[head_i] = tf.concat([mean_hidden_list_train[head_i],
            #                                             std_hidden_train], axis=2)
            # mean_hidden_list_test[head_i] = tf.concat([mean_hidden_list_test[head_i],
            #                                            std_hidden_test], axis=2)

    if pool_heads == "gating":
        mean_glu_layer = tf.keras.layers.Dense(number_of_features)
        if use_temporal_std:
            std_glu_layer = tf.keras.layers.Dense(number_of_features)
        for head_i in range(number_of_heads):
            mean_hidden_train = tf.reshape(mean_hidden_list_train[head_i], (-1, number_of_features))
            mean_hidden_test = tf.reshape(mean_hidden_list_test[head_i], (-1, number_of_features))
            if use_temporal_std:
                std_hidden_train = tf.reshape(std_hidden_list_train[head_i], (-1, number_of_features))
                std_hidden_test = tf.reshape(std_hidden_list_test[head_i], (-1, number_of_features))

            mean_hidden_list_train[head_i] = tf.multiply(tf.math.sigmoid(mean_glu_layer(mean_hidden_train,
                                                                                   training=True)),
                                                         mean_hidden_train)  # [bs, number_of_features]
            mean_hidden_list_test[head_i] = tf.multiply(tf.math.sigmoid(mean_glu_layer(mean_hidden_test,
                                                                                  training=False)),
                                                        mean_hidden_test)  # [bs, number_of_features]
            if use_temporal_std:
                std_hidden_list_train[head_i] = tf.multiply(tf.math.sigmoid(std_glu_layer(std_hidden_train,
                                                                                            training=True)),
                                                             std_hidden_train)
                std_hidden_list_test[head_i] = tf.multiply(tf.math.sigmoid(std_glu_layer(std_hidden_test,
                                                                                           training=False)),
                                                            std_hidden_test)

                mean_hidden_list_train[head_i] = tf.concat([mean_hidden_list_train[head_i],
                                                            std_hidden_list_train[head_i]], axis=1)  # [bs, 2*number_of_features]
                mean_hidden_list_test[head_i] = tf.concat([mean_hidden_list_test[head_i],
                                                           std_hidden_list_test[head_i]], axis=1)  # [bs, 2*number_of_features]
                # mean_hidden_train = tf.multiply(tf.math.sigmoid(glu_layer(mean_hidden_train, training=True)),
                #                                 mean_hidden_train)
                # mean_hidden_test = tf.multiply(tf.math.sigmoid(glu_layer(mean_hidden_test, training=False)),
                #                                mean_hidden_test)
    elif pool_heads == "gating_old":
        if use_temporal_std:
            glu_layer = tf.keras.layers.Dense(2 * number_of_features * number_of_heads)
        else:
            glu_layer = tf.keras.layers.Dense(number_of_features * number_of_heads)
        for head_i in range(number_of_heads):
            mean_hidden_list_train[head_i] = tf.reshape(mean_hidden_list_train[head_i], (-1, number_of_features))
            mean_hidden_list_test[head_i] = tf.reshape(mean_hidden_list_test[head_i], (-1, number_of_features))
            if use_temporal_std:
                std_hidden_list_train[head_i] = tf.reshape(std_hidden_list_train[head_i], (-1, number_of_features))
                std_hidden_list_test[head_i] = tf.reshape(std_hidden_list_test[head_i], (-1, number_of_features))

        if use_temporal_std:
            mean_hidden_list_train.extend(std_hidden_list_train)
            mean_hidden_list_test.extend(std_hidden_list_test)

        mean_hidden_train = tf.concat(mean_hidden_list_train, axis=1)
        mean_hidden_test = tf.concat(mean_hidden_list_test, axis=1)

        mean_hidden_train = tf.multiply(tf.math.sigmoid(glu_layer(mean_hidden_train,
                                                                                        training=True)),
                                                         mean_hidden_train)
        mean_hidden_test = tf.multiply(tf.math.sigmoid(glu_layer(mean_hidden_test,
                                                                  training=True)),
                                       mean_hidden_test)

        mean_hidden_list_train = list()
        mean_hidden_list_test = list()
        mean_hidden_list_train.append(mean_hidden_train)  # [bs, number_of_heads*number_of_features] -- consider STD
        mean_hidden_list_test.append(mean_hidden_test)  # [bs, number_of_heads*number_of_features] -- consider STD

    elif pool_heads in ["attention", "attention_auto"]:
        if number_of_heads == 1:
            raise NotImplementedError

        pool_heads_layer = Dense(1, use_bias=False)
        pool_heads_hidden_list_train = list()
        pool_heads_hidden_list_test = list()
        pool_heads_energy_list_train = list()
        pool_heads_energy_list_test = list()

        for head_i in range(number_of_heads):
            mean_hidden_train = tf.reshape(mean_hidden_list_train[head_i], (-1, number_of_features))
            mean_hidden_test = tf.reshape(mean_hidden_list_test[head_i], (-1, number_of_features))
            if use_temporal_std:
                std_hidden_train = tf.reshape(std_hidden_list_train[head_i], (-1, number_of_features))
                std_hidden_test = tf.reshape(std_hidden_list_test[head_i], (-1, number_of_features))
                pool_heads_hidden_list_train.append(tf.concat([mean_hidden_train,
                                                            std_hidden_train], axis=1))
                pool_heads_hidden_list_test.append(tf.concat([mean_hidden_test,
                                                           std_hidden_test], axis=1))
            else:
                pool_heads_hidden_list_train.append(mean_hidden_train)
                pool_heads_hidden_list_test.append(mean_hidden_test)
            pool_heads_energy_train = tf.reshape(pool_heads_layer(pool_heads_hidden_list_train[head_i], training=True), (-1, 1))
            pool_heads_energy_test = tf.reshape(pool_heads_layer(pool_heads_hidden_list_test[head_i], training=False), (-1, 1))
            pool_heads_energy_list_train.append(pool_heads_energy_train)
            pool_heads_energy_list_test.append(pool_heads_energy_test)
        pool_heads_energy_train = tf.concat(pool_heads_energy_list_train, axis=1)
        pool_heads_energy_test = tf.concat(pool_heads_energy_list_test, axis=1)

        if pool_heads == "attention":
            pool_heads_attention_weights_train = tf.reshape(tf.nn.softmax(pool_heads_energy_train, axis=1),
                                                            (-1, number_of_heads))
            pool_heads_attention_weights_test = tf.reshape(tf.nn.softmax(pool_heads_energy_test, axis=1),
                                                           (-1, number_of_heads))
        elif pool_heads == "attention_auto":
            pool_heads_auto_pool_variable = tf.Variable(1.0, dtype=tf.float32)

            pool_heads_attention_weights_train = tf.reshape(tf.nn.softmax(pool_heads_auto_pool_variable * pool_heads_energy_train, axis=1),
                                                            (-1, number_of_heads))
            pool_heads_attention_weights_test = tf.reshape(tf.nn.softmax(pool_heads_auto_pool_variable * pool_heads_energy_test, axis=1),
                                                           (-1, number_of_heads))
        else:
            raise NotImplementedError

        pooled_hidden_train = pool_heads_hidden_list_train[0] * tf.expand_dims(pool_heads_attention_weights_train[:, 0], axis=1)
        pooled_hidden_test = pool_heads_hidden_list_test[0] * tf.expand_dims(pool_heads_attention_weights_test[:, 0], axis=1)
        for head_i in range(1, number_of_heads):
            pooled_hidden_train = pooled_hidden_train + pool_heads_hidden_list_train[head_i] * tf.expand_dims(pool_heads_attention_weights_train[:, head_i], axis=1)

            pooled_hidden_test = pooled_hidden_test + pool_heads_hidden_list_test[head_i] * tf.expand_dims(pool_heads_attention_weights_test[:, head_i], axis=1)

        mean_hidden_list_train = list()
        mean_hidden_list_test = list()
        mean_hidden_list_train.append(pooled_hidden_train)  # [bs, number_of_heads*number_of_features] -- consider STD
        mean_hidden_list_test.append(pooled_hidden_test)  # [bs, number_of_heads*number_of_features] -- consider STD
    elif pool_heads == "no_pool":
        for head_i in range(number_of_heads):
            mean_hidden_train = tf.reshape(mean_hidden_list_train[head_i], (-1, number_of_features))
            mean_hidden_test = tf.reshape(mean_hidden_list_test[head_i], (-1, number_of_features))
            if use_temporal_std:
                std_hidden_train = tf.reshape(std_hidden_list_train[head_i], (-1, number_of_features))
                std_hidden_test = tf.reshape(std_hidden_list_test[head_i], (-1, number_of_features))
                mean_hidden_list_train[head_i] = tf.concat([mean_hidden_train,
                                                            std_hidden_train], axis=1)
                mean_hidden_list_test[head_i] = tf.concat([mean_hidden_test,
                                                           std_hidden_test], axis=1)
    else:
        raise ValueError("Invalid head pooling method.")

    if (len(mean_hidden_list_train) > 1) and (len(mean_hidden_list_test) > 1):
        mean_hidden_train = tf.concat(mean_hidden_list_train, axis=1)
        mean_hidden_test = tf.concat(mean_hidden_list_test, axis=1)
    elif (len(mean_hidden_list_train) == 1) and (len(mean_hidden_list_test) == 1):
        mean_hidden_train = mean_hidden_list_train[0]
        mean_hidden_test = mean_hidden_list_test[0]
    else:
        raise ValueError("Invalid number of heads.")

    if pool_heads == "attention":
        if use_temporal_std:
            mean_hidden_train = tf.reshape(mean_hidden_train, (-1, 2 * number_of_features))
            mean_hidden_test = tf.reshape(mean_hidden_test, (-1, 2 * number_of_features))
        else:
            mean_hidden_train = tf.reshape(mean_hidden_train, (-1, number_of_features))
            mean_hidden_test = tf.reshape(mean_hidden_test, (-1, number_of_features))
    else:
        if use_temporal_std:
            mean_hidden_train = tf.reshape(mean_hidden_train, (-1, 2 * number_of_features * number_of_heads))
            mean_hidden_test = tf.reshape(mean_hidden_test, (-1, 2 * number_of_features * number_of_heads))
        else:
            mean_hidden_train = tf.reshape(mean_hidden_train, (-1, number_of_features * number_of_heads))
            mean_hidden_test = tf.reshape(mean_hidden_test, (-1, number_of_features * number_of_heads))

    classification_layer = tf.keras.layers.Dense(2)

    prediction_single_train = classification_layer(mean_hidden_train, training=True)
    prediction_single_test = classification_layer(mean_hidden_test, training=False)

    prediction_train = dict()
    prediction_train["whinny_single"] = prediction_single_train
    prediction_train["whinny_continuous"] = prediction_single_train
    prediction_train["attention_weights"] = attention_weights_train

    prediction_test = dict()
    prediction_test["whinny_single"] = prediction_single_test
    prediction_test["whinny_continuous"] = prediction_single_test
    prediction_test["attention_weights"] = attention_weights_test

    return prediction_train, prediction_test


def get_VGG16Top_global_pooling(net_train,
                                net_test,
                                global_pooling_configuration):
    flatten_single = Flatten(name='flatten')
    dense_1_single = Dense(4096, activation='relu', name='fc1_single')
    dense_2_single = Dense(4096, activation='relu', name='fc2_single')
    dense_3_single = Dense(2, name='prediction_single')

    # reshape_1_continuous = Reshape((4*512, ))
    dense_1_continuous = Dense(4096, activation='relu', name='fc1_continuous')
    dense_2_continuous = Dense(4096, activation='relu', name='fc2_continuous')
    dense_3_continuous = Dense(2, name='prediction_continuous')
    # reshape_2_continuous = Reshape((9, 2))

    net_single_train = flatten_single(net_train, training=True)
    net_single_train = dense_1_single(net_single_train, training=True)
    net_single_train = dense_2_single(net_single_train, training=True)
    prediction_single_train = dense_3_single(net_single_train, training=True)

    net_single_test = flatten_single(net_test, training=False)
    net_single_test = dense_1_single(net_single_test, training=False)
    net_single_test = dense_2_single(net_single_test, training=False)
    prediction_single_test = dense_3_single(net_single_test, training=False)

    # net_continuous_train = reshape_1_continuous(net_train, training=True)
    net_continuous_train = tf.reshape(net_train, (-1, 4*512))
    net_continuous_train = dense_1_continuous(net_continuous_train, training=True)
    net_continuous_train = dense_2_continuous(net_continuous_train, training=True)
    net_continuous_train = dense_3_continuous(net_continuous_train, training=True)
    # prediction_continuous_train = reshape_2_continuous(net_continuous_train, training=True)
    prediction_continuous_train = tf.reshape(net_continuous_train, (-1, 9, 2))

    # net_continuous_test = reshape_1_continuous(net_test, training=False)
    net_continuous_test = tf.reshape(net_test, (-1, 4*512))
    net_continuous_test = dense_1_continuous(net_continuous_test, training=False)
    net_continuous_test = dense_2_continuous(net_continuous_test, training=False)
    net_continuous_test = dense_3_continuous(net_continuous_test, training=False)
    # prediction_continuous_test = reshape_2_continuous(net_continuous_test, training=False)
    prediction_continuous_test = tf.reshape(net_continuous_test, (-1, 9, 2))

    attention_weights_train = prediction_continuous_train
    attention_weights_test = prediction_continuous_test

    prediction_train = dict()
    prediction_train["whinny_single"] = prediction_single_train
    prediction_train["whinny_continuous"] = net_continuous_train
    prediction_train["attention_weights"] = net_continuous_train

    prediction_test = dict()
    prediction_test["whinny_single"] = prediction_single_test
    prediction_test["whinny_continuous"] = net_continuous_test
    prediction_test["attention_weights"] = net_continuous_test

    return prediction_train, prediction_test


# The below are for the breath sensing paper.

def pooling_block(net,
                  config_dict):
    batch_size = config_dict["batch_size"]
    seq_length = config_dict["sequence_length"]
    hidden_units = 256
    # hidden_units = 128
    number_of_outputs = config_dict["number_of_outputs"]

    outputs = net

    # if seq_length is None:
    #     seq_length = -1

    if config_dict["attention"] == "feature":
        prediction_continuous = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

        prediction_continuous = tf.layers.dense(prediction_continuous, number_of_outputs)

        prediction_continuous = tf.reshape(prediction_continuous, (batch_size, seq_length, number_of_outputs))

        att_batch_size = batch_size
        att_seq_length = seq_length

        outputs = tf.reshape(outputs, (att_batch_size, att_seq_length, hidden_units))

        query_value_attention_seq, attention_weights = custom_attention(q=outputs,
                                                                        v=outputs,
                                                                        k=outputs)
        attention_weights = tf.reshape(attention_weights, (batch_size, seq_length, att_seq_length))

        # outputs = tf.keras.layers.GlobalAveragePooling1D()(
        #     outputs)
        # query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        #     query_value_attention_seq)
        #
        # outputs = tf.concat([outputs, query_value_attention], axis=1)
        # net = tf.reshape(outputs, [batch_size, 2 * hidden_units])

        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
            query_value_attention_seq)

        outputs = query_value_attention
        net = tf.reshape(outputs, [batch_size, hidden_units])

        prediction_single = tf.layers.dense(net, number_of_outputs)

        prediction_single = tf.reshape(prediction_single, (batch_size, number_of_outputs))

    elif config_dict["attention"] == "prediction":

        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

        prediction = tf.layers.dense(outputs, number_of_outputs)

        prediction_continuous = tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))

        att_batch_size = batch_size
        att_seq_length = seq_length

        outputs = tf.reshape(prediction_continuous, (att_batch_size, att_seq_length, number_of_outputs))

        query_value_attention_seq, attention_weights = custom_attention(q=outputs,
                                                                        v=outputs,
                                                                        k=outputs)
        attention_weights = tf.reshape(attention_weights, (batch_size, seq_length, att_seq_length))

        # outputs = tf.keras.layers.GlobalAveragePooling1D()(
        #     outputs)
        # query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        #     query_value_attention_seq)
        #
        # outputs = tf.concat([outputs, query_value_attention], axis=1)
        # net = tf.reshape(outputs, [batch_size, 2 * hidden_units])

        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
            query_value_attention_seq)

        outputs = query_value_attention
        prediction_single = tf.reshape(outputs, [batch_size, number_of_outputs])

    elif config_dict["attention"] == "flatten":
        net = outputs
        net = tf.reshape(net, (batch_size * seq_length, hidden_units))

        prediction_continuous = tf.layers.dense(net, number_of_outputs)
        prediction_continuous = tf.reshape(prediction_continuous, (batch_size, seq_length, number_of_outputs))

        net = tf.reshape(net, (batch_size, 75 * 256))

        net = tf.layers.dense(net, hidden_units)
        attention_weights = tf.ones(shape=(batch_size, seq_length, 50), dtype=tf.float32)
        net = tf.reshape(net, (batch_size, hidden_units))

        prediction_single = tf.layers.dense(net, number_of_outputs)

        prediction_single = tf.reshape(prediction_single, (batch_size, number_of_outputs))

    elif config_dict["attention"] == "none":
        net = outputs
        net = tf.reshape(net, (batch_size * seq_length,hidden_units))

        prediction_continuous = tf.layers.dense(net, number_of_outputs)
        prediction_continuous = tf.reshape(prediction_continuous, (batch_size, seq_length, number_of_outputs))

        net = tf.reshape(net, (batch_size, seq_length, hidden_units))

        net = tf.keras.layers.GlobalMaxPooling1D()(
            net)
        attention_weights = tf.ones(shape=(batch_size, seq_length, 50), dtype=tf.float32)
        net = tf.reshape(net, (batch_size, hidden_units))

        prediction_single = tf.layers.dense(net, number_of_outputs)

        prediction_single = tf.reshape(prediction_single, (batch_size, number_of_outputs))

    else:
        raise ValueError

    return prediction_single, prediction_continuous, attention_weights


def custom_attention(q, v, k):
    scores = tf.tanh(tf.matmul(q, k, transpose_b=True))
    # print(scores.shape)
    distribution = tf.nn.softmax(scores)
    # print(distribution.shape)
    weighted_sum = tf.matmul(distribution, v)
    # print(weighted_sum.shape)
    return weighted_sum, distribution
