from tensorflow.keras.layers import Input,Conv2D,Concatenate, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D,Dropout,UpSampling2D, Add, AveragePooling2D, DepthwiseConv2D
from tensorflow.keras.models import Model


def ASPP(inputs, param, pool_type='average', rates=[1, 6, 12, 18]):
    shape = inputs.shape

    if pool_type == 'average':
        y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    else:
        y_pool = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y_pool = Conv2D(filters=param, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    atr = []
    atr.append(y_pool)
    for i in range(len(rates)):
        if i == 0:
            y_atr = DepthwiseConv2D(kernel_size=1, dilation_rate=rates[i], padding='same', use_bias=False)(inputs)
        else:
            y_atr = DepthwiseConv2D(kernel_size=3, dilation_rate=rates[i], padding='same', use_bias=False)(inputs)
        y_atr = BatchNormalization()(y_atr)
        y_atr = Activation('relu')(y_atr)
        atr.append(y_atr)

    y = Concatenate()(atr)

    y = Conv2D(filters=param, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y
    
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

def unet_aspp(inp_shape, param=[], dropout=False, dropout_rate=0, num_class=2, last_activation='sigmoid'):
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

    aspp = ASPP(mp,param=256)
    aspp = UpSampling2D((4, 4), interpolation="bilinear")(aspp)
    

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
        if dec.shape[1] == aspp.shape[1]:
            x = Concatenate()([dec, aspp])
            x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((4, 4), interpolation="bilinear")(x)
            
    un = Concatenate()([dec,x])

    fi = Conv2D(filters=num_class, kernel_size=1)(un)
    fi = Activation(last_activation)(fi)
    model = Model(inputs=inp, outputs=fi)

    return model