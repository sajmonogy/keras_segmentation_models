import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical,normalize
import matplotlib.pyplot as plt
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


def read_images(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    x = normalize(x, axis=1)
    return x


def read_masks(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def read_train_flow(path_train):
    path_img = os.path.join(path_train, 'images')
    path_masks = os.path.join(path_train, 'masks')
    names = os.listdir(path_img)
    all_img = []
    all_masks = []
    for name in names:
        img = read_images(os.path.join(path_img, name))
        mask = read_masks(os.path.join(path_masks, name))
        all_img.append(img)
        all_masks.append(mask)
    all_img = np.array(all_img)
    all_masks = np.array(all_masks)
    return all_img, all_masks


def read_val_flow(path_val, num_class=2):
    path_img = os.path.join(path_val, 'images')
    path_masks = os.path.join(path_val, 'masks')
    names = os.listdir(path_img)
    all_img = []
    all_masks = []
    for name in names:
        img = read_images(os.path.join(path_img, name))
        mask = read_masks(os.path.join(path_masks, name))
        all_img.append(img)
        all_masks.append(mask)
    all_img = np.array(all_img)
    all_masks = np.array(all_masks)

    return all_img, all_masks


def preprocess_data(img, mask, num_class=2, backbone=''):
    if backbone == '':
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        mask = to_categorical(mask, num_class)
        mask = mask.reshape((mask.shape[0], mask.shape[1], num_class))
    else:
        BACKBONE = backbone
        preprocess_input = sm.get_preprocessing(BACKBONE)
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        img = preprocess_input(img)
        mask = to_categorical(mask, num_class)
        mask = mask.reshape((mask.shape[0], mask.shape[1], num_class))
    return (img, mask)


def my_image_mask_generator(image_generator, mask_generator, num_classes=2):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class=num_classes)
        yield (img, mask)


def make_generator_flow(imgs, masks, batch_size=2, seed=42, augment_dict_i={}, augment_dict_m={}):
    if augment_dict_i == {}:
        img_gen = ImageDataGenerator()
        mask_gen = ImageDataGenerator()
    else:
        img_gen = ImageDataGenerator(**augment_dict_i)
        img_gen.fit(imgs, augment=True, seed=seed)
        mask_gen = ImageDataGenerator(**augment_dict_m)
        mask_gen.fit(masks, augment=True, seed=seed)

    img_gener = img_gen.flow(imgs, batch_size=batch_size, seed=seed, shuffle=False)
    mask_gen = mask_gen.flow(masks, batch_size=batch_size, seed=seed, shuffle=False)

    my_generator = my_image_mask_generator(img_gener, mask_gen)

    return my_generator


def make_generator_flow_dir(train_img_path, train_mask_path, num_class, augment_dict_i={}, augment_dict_m={},target_size=(),batch_size=2,seed=42):
    if augment_dict_i == {}:
        img_gen = ImageDataGenerator()
        mask_gen = ImageDataGenerator()
    else:
        img_gen = ImageDataGenerator(**augment_dict_i)
        img_gen.fit(imgs, augment=True, seed=seed)
        mask_gen = ImageDataGenerator(**augment_dict_m)
        mask_gen.fit(masks, augment=True, seed=seed)

    image_generator = img_gen.flow_from_directory(
        train_img_path,
        target_size=target_size,
        class_mode=None,
        batch_size=batch_size,
        seed=seed)

    mask_generator = mask_gen.flow_from_directory(
        train_mask_path,
        target_size=target_size,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    my_generator = my_image_mask_generator(image_generator,mask_generator)

    return my_generator


def plot_gen(gen, batch_size=2):
    x, y = gen.__next__()
    for i in range(0, batch_size):
        image = x[i]
        mask = np.argmax(y[i],axis=2)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.show()
