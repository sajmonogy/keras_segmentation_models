from tensorflow.keras.layers import Input,Conv2D,Concatenate, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D,Dropout,UpSampling2D, Add, AveragePooling2D
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


def simple_unet(inp_shape, param=[], dropout=False, dropout_rate=0, num_class=2, last_activation='sigmoid'): # parameters have to be defined
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