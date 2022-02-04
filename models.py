from tensorflow.keras.layers import Input,Conv2D,Concatenate, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D,Dropout,UpSampling2D, Add
from tensorflow.keras.models import Model

############ SIMPLE UNET ############################


def simple_unet_encoder_block(inp, param, activation='relu'):
    enc = Conv2D(param, kernel_size=(3, 3), padding='same')(inp)
    enc = BatchNormalization()(enc)
    enc = Activation(activation)(enc)
    enc = Conv2D(param, kernel_size=(3, 3), padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = Activation(activation)(enc)

    return enc


def simple_unet_decoder_block(inp, conc, param, activation='relu'):
    dec = Conv2DTranspose(param, kernel_size=(2, 2), strides=(2, 2), padding='same')(inp)
    dec = Concatenate()([dec, conc])
    dec = Conv2D(param, kernel_size=(3, 3), padding='same')(dec)
    dec = BatchNormalization()(dec)
    dec = Activation(activation)(dec)
    dec = Conv2D(param, kernel_size=(3, 3), padding='same')(dec)
    dec = BatchNormalization()(dec)
    dec = Activation(activation)(dec)

    return dec


def simple_unet(inp_shape, param=[], dropout=False, dropout_rate=0, num_class=2, last_activation='sigmoid'):
    inp = Input(shape=inp_shape)
    encoder_blocks = []
    for i in range(len(param)):
        if i == 0:
            enc = simple_unet_encoder_block(inp, param[i])
        else:
            enc = simple_unet_encoder_block(mp, param[i])
        if i == len(param) - 1:
            encoder_blocks.append(enc)
        else:
            mp = MaxPooling2D((2, 2))(enc)
            if dropout == True:
                mp = Dropout(dropout_rate)(mp)

            encoder_blocks.append(enc)

    for i in range(len(param) - 1, -1, -1):
        if i == 0:
            pass
        else:
            print(i)
            if i == len(param) - 1:
                dec = simple_unet_decoder_block(enc, encoder_blocks[i - 1], param[i - 1])
            else:
                dec = simple_unet_decoder_block(dec, encoder_blocks[i - 1], param[i - 1])
            if dropout==True:
                dec = Dropout(dropout_rate)(dec)

    fi = Conv2D(filters=num_class, kernel_size=1)(dec)
    fi = Activation(last_activation)(fi)
    model = Model(inputs=inp, outputs=fi)
    return model


################ DEEP UNET ####################################


def deep_unet_encoder_block_in(inp, kernel_size_1=(3, 3), kernel_size_2=(3, 3), param_1=64, param_2=32):
    enc = Conv2D(param_1, kernel_size=kernel_size_1, padding='same')(inp)
    enc = BatchNormalization()(enc)
    enc = Activation('relu')(enc)
    enc = Conv2D(param_2, kernel_size=kernel_size_2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = Activation('relu')(enc)
    return enc


def deep_unet_encoder_block(inp, activation='relu', kernel_size_1=(3, 3), kernel_size_2=(3, 3), param_1=64, param_2=32):
    inp_1 = inp
    enc = Conv2D(param_1, kernel_size=kernel_size_1, padding='same')(inp)
    enc = BatchNormalization()(enc)
    enc = Activation('relu')(enc)
    enc = Conv2D(param_2, kernel_size=kernel_size_2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = Activation('relu')(enc)
    enc = Add()([enc, inp_1])
    return enc


def deep_unet_decoder_block_ups(inp, conc, activation='relu', kernel_size_1=(3, 3), kernel_size_2=(3, 3), param_1=64,
                                param_2=32):
    dec = UpSampling2D((2, 2))(inp)
    dec = Concatenate()([dec, conc])
    dec = Conv2D(param_1, kernel_size=kernel_size_1, padding='same')(dec)
    dec = BatchNormalization()(dec)
    dec = Activation('relu')(dec)
    dec = Conv2D(param_2, kernel_size=kernel_size_2, padding='same')(dec)
    dec = BatchNormalization()(dec)
    dec = Activation('relu')(dec)
    return dec


def deep_unet(inp_shape, layer_deep=5, dropout=False, dropout_rate=0, num_class=2, last_activation='sigmoid'):
    inp = Input(shape=inp_shape)
    encoder_blocks = []
    for i in range(layer_deep):
        if i == 0:
            enc = deep_unet_encoder_block_in(inp)
        else:
            enc = deep_unet_encoder_block(mp)
        if i == layer_deep - 1:
            encoder_blocks.append(enc)
        else:
            mp = MaxPooling2D((2, 2))(enc)
            if dropout == True:
                mp = Dropout(dropout_rate)(mp)
            encoder_blocks.append(enc)

    for i in range(layer_deep - 1, -1, -1):

        if i == layer_deep - 1:
            dec = deep_unet_encoder_block_in(enc)
        else:
            dec = deep_unet_decoder_block_ups(dec, encoder_blocks[i])
            if dropout == True:
                dec = Dropout(dropout_rate)(dec)

    fi = Conv2D(filters=num_class, kernel_size=1)(dec)
    fi = Activation('sigmoid')(fi)

    model = Model(inputs=inp, outputs=fi)

    return model


## DeepLabV3+

def backbone(name):
    pass

def ASPP(input):
    pass