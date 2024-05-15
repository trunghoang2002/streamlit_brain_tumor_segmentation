from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Input, BatchNormalization, Lambda
from keras.models import Model
import tensorflow as tf

IMG_DIM = (128, 128, 1)

def sharpness(image):
    # Define a sharpening filter
    sharpening_filter = tf.constant([[-1, -1, -1],
                                     [-1, 9, -1],
                                     [-1, -1, -1]], dtype=tf.float32)

    # Expand dimensions to match the shape of the image
    sharpening_filter = tf.expand_dims(tf.expand_dims(sharpening_filter, axis=-1), axis=-1)

    # Apply the convolution operation to enhance sharpness
    sharpened_image = tf.nn.conv2d(image, filters=sharpening_filter, strides=[1, 1, 1, 1], padding='SAME')

    # Clip the values to be in the valid range [0, 1]
    sharpened_image = tf.clip_by_value(sharpened_image, 0, 1)

    return sharpened_image


def conv2d_block(input_tensor, n_filters, kernel_size=(3, 3), name="contraction"):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal',
               padding='same', activation="relu", name=name + '_1')(input_tensor)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal',
               padding='same', activation="relu", name=name + '_2')(x)
    return x

def unet_plusplus(input_shape, num_classes):
    inp = Input(shape=input_shape)

    # Thêm lớp độ sắc nét vào đầu vào của mô hình
    sharpness_layer = Lambda(sharpness, name='sharpness_layer')(inp)

    # Gộp lớp độ sắc nét và đầu vào của U-Net++
    merged_input = concatenate([sharpness_layer, inp])

    d1 = conv2d_block(merged_input, 64, name="contraction_1")
    p1 = MaxPooling2D(pool_size=(2, 2))(d1)
    p1 = BatchNormalization(momentum=0.8)(p1)
    p1 = Dropout(0.1)(p1)

    d2 = conv2d_block(p1, 128, name="contraction_2_1")
    p2 = MaxPooling2D(pool_size=(2, 2))(d2)
    p2 = BatchNormalization(momentum=0.8)(p2)
    p2 = Dropout(0.1)(p2)

    d3 = conv2d_block(p2, 256, name="contraction_3_1")
    p3 = MaxPooling2D(pool_size=(2, 2))(d3)
    p3 = BatchNormalization(momentum=0.8)(p3)
    p3 = Dropout(0.1)(p3)

    d4 = conv2d_block(p3, 512, name="contraction_4_1")
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)
    p4 = BatchNormalization(momentum=0.8)(p4)
    p4 = Dropout(0.1)(p4)

    d5 = conv2d_block(p4, 512, name="contraction_5_1")

    u1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(d5)
    u1 = concatenate([u1, d4])
    u1 = Dropout(0.1)(u1)
    c1 = conv2d_block(u1, 512, name="expansion_1")

    u2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c1)
    u2 = concatenate([u2, d3])
    u2 = Dropout(0.1)(u2)
    c2 = conv2d_block(u2, 256, name="expansion_2")

    u3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c2)
    u3 = concatenate([u3, d2])
    u3 = Dropout(0.1)(u3)
    c3 = conv2d_block(u3, 128, name="expansion_3")

    u4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, d1])
    u4 = Dropout(0.1)(u4)
    c4 = conv2d_block(u4, 64, name="expansion_4")

    out = Conv2D(1, (1, 1), name="output", activation='sigmoid')(c4)

    model = Model(inputs=inp, outputs=out)
    return model

num_classes = 2
# Sử dụng hàm unet_plusplus để xây dựng mô hình U-Net++
model_unetplusplus = unet_plusplus(IMG_DIM, num_classes)
# print(model_unetplusplus.summary())