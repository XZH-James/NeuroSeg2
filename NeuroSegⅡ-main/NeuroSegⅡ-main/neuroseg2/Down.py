"""
NeuroSegⅡ
The Additional subsampling of NeuroSegⅡ .

changed from YOlOv4 : https://github.com/bubbliiiing/yolov4-keras/blob/master/nets/CSPdarknet53.py

Written by ZheHao Xu

"""

from keras.layers import (BatchNormalization, Conv2D, Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Add,
                          Input, Multiply, Reshape, ZeroPadding2D, MaxPooling2D)
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2

from keras.layers import GlobalMaxPool2D, Concatenate


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer': RandomNormal(stddev=0.02),
                           'kernel_regularizer': l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_SiLU(*args, **kwargs):  # Block of convolution
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum=0.97, epsilon=0.001, name=kwargs['name'] + '.bn'),
        SiLU())


def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name=""):
    y = compose(
        DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name=name + '.cv1'),
        DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name=name + '.cv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y


def Res(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)
    # ----------------------------------------------------------------#
    #   The main body loops through num_blocks, and inside the loop is a residual structure.
    # ----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name=name + '.cv1')(x)
    # --------------------------------------------------------------------#
    #   Then, a large residual edge shortconv is established. This large residual edge bypasses many residual structures
    # --------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name=name + '.cv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name=name + '.m.' + str(i))
    # ----------------------------------------------------------------#
    #   Stack the big residual edge back up
    # ----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])

    # ----------------------------------------------------------------#
    #   Finally, the number of channels is integrated
    # ----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name=name + '.cv3')(route)


def SPPBottleneck(x, out_channels, weight_decay=5e-4, name=""):
    # ---------------------------------------------------#
    #   The SPP structure is used, which is the maximum pooled stack of different scales.
    # ---------------------------------------------------#
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name=name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name=name + '.cv2')(x)
    return x


def down(inputs, base_channels=64, weight_decay=5e-4, name=""):
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    output = DarknetConv2D_BN_SiLU(int(base_channels * 4), (3, 3), strides=(2, 2), weight_decay=weight_decay,
                                   name=name)(x)
    return output


def downlast(inputs, base_channels=64, weight_decay=5e-4, name=""):
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x1 = DarknetConv2D_BN_SiLU(int(base_channels * 4), (3, 3), strides=(2, 2), weight_decay=weight_decay, name=name)(x)
    x2 = SPPBottleneck(x1, int(base_channels * 4), weight_decay=weight_decay, name=name + '.1')
    return x2
