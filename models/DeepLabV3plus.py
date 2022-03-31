from tensorflow.keras.layers import Input,Conv2D,Concatenate, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D,Dropout,UpSampling2D, Add, AveragePooling2D, DepthwiseConv2D, Reshape
from tensorflow.keras.models import Model

def backbone_dlv3(inputs, name,os=16):
    if os == 16:
        if name == 'resnet50':
            from tensorflow.keras.applications import ResNet50
            base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
            a = base.get_layer('conv4_block6_out').output
            b = base.get_layer('conv2_block3_out').output
        if name == 'resnet101':
            from tensorflow.keras.applications import ResNet101
            base = ResNet101(weights='imagenet', include_top=False, input_tensor=inputs)
            a = base.get_layer('conv4_block23_out').output
            b = base.get_layer('conv2_block3_out').output
        if name == 'xception':
            from tensorflow.keras.applications.xception import Xception
            base = Xception(weights='imagenet', include_top=False, input_tensor=inputs)
            a = base.get_layer('block13_sepconv2_bn').output
            b = base.get_layer('block3_sepconv2_bn').output
    if os == 8:
        if name == 'resnet50':
            from tensorflow.keras.applications import ResNet50
            base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
            a = base.get_layer('conv3_block4_out').output
            b = base.get_layer('conv2_block3_out').output
        if name == 'resnet101':
            from tensorflow.keras.applications import ResNet101
            base = ResNet101(weights='imagenet', include_top=False, input_tensor=inputs)
            a = base.get_layer('conv4_block23_out').output
            b = base.get_layer('conv2_block3_out').output
    return a, b


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


def DeepLabV3plus(input_shape, backbone, last_activation='sigmoid', num_class=2, os=16): # dostosować stride
    inputs = Input(input_shape)

    a, b = backbone_dlv3(inputs, name=backbone,os=os)


    if os==16:
        x_a = ASPP(a, param=256)
        x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
    elif os==8:
        x_a = ASPP(a, param=256,rates=[1, 12, 24, 36])
        x_a = UpSampling2D((2, 2), interpolation="bilinear")(x_a)

    x_b = b
    print(x_b.shape)
    x_b = Reshape((x_b.shape[1]+1,x_b.shape[2]+1,x_b.shape[3]))(b)
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    x = Conv2D(num_class, (1, 1))(x)
    x = Activation(last_activation)(x)

    model = Model(inputs=inputs, outputs=x)
    return model