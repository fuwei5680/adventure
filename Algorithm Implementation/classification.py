import cv2
from keras.models import load_model, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from tensorflow.python.keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf


def classify_model():
    inputs = Input(shape=(32, 32, 3))

    conv2D = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                    kernel_initializer='he_normal')(inputs)
    conv2D_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(conv2D)
    max_pooling2d = MaxPooling2D(pool_size=2, strides=2, padding='valid')(conv2D_1)

    conv2d_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(max_pooling2d)
    conv2d_3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(conv2d_2)
    max_pooling2d_1 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(conv2d_3)

    conv2d_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(max_pooling2d_1)
    conv2d_5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(conv2d_4)
    max_pooling2d_2 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(conv2d_5)

    global_pooling = GlobalAveragePooling2D()(max_pooling2d_2)
    dense = (Dense(10, activation='softmax'))(global_pooling)

    model = Model(inputs=inputs, outputs=dense, name='model')
    model.summary()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    return model


def train():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model = classify_model()
    hist = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test), shuffle=True)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    model.save('./CIFAR10_model_else.h5')


def decode(index):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return labels[index]


def get_layer(model, x, index=-1):
    layer = K.function([model.input], model.layers[index].output)
    return layer([x])


def predict():
    model = load_model('./CIFAR10_model_else.h5')
    # airplane
    for i in range(10):
        count = 0
        err_list = []
        for j in range(10):
            filepath = './examples/' + decode(i) + '/' + decode(i) + str(j + 1) + '.jpg'
            img = load_img(filepath, target_size=(32, 32))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            pred = np.argmax(model.predict(img))
            if pred == i:
                count += 1
            else:
                err_list.append((j + 1, decode(pred)))
        print(decode(i), ' acc: ', count, '/10')
        print('error instance: ', err_list)


def pred_CAM(model, test_img):
    pred = model.predict(test_img)
    conv_2d_output = get_layer(model, test_img, -3)
    conv_2d_output_np = np.reshape(conv_2d_output, (4, 4, 256))
    wb_dense = np.array(model.layers[11].get_weights())
    heat_map = np.zeros((32, 32))
    for i in range(0, 256):
        tmp1 = conv_2d_output_np[:, :, i]
        tmp2 = np.reshape(tmp1, (4, 4))
        tmp3 = cv2.resize(tmp2, (32, 32), interpolation=cv2.INTER_AREA)
        tmp4 = wb_dense[0][i, np.argmax(pred)] * tmp3  # wc*f(x,y)
        heat_map[:, :] = heat_map[:, :] + tmp4[:, :]
    heat_map = np.maximum(heat_map, 0)
    return heat_map


def pred_Grad_CAM(model, test_img):
    with tf.GradientTape() as gtape:
        middle = Model(inputs=model.input, outputs=[model.output, model.get_layer('max_pooling2d_2').output])
        pred, last = middle(test_img)
        prob = pred[:, np.argmax(pred[0])]
        grads = gtape.gradient(prob, last)
    res = np.multiply(grads[0], last[0])
    res = np.sum(res, axis=2)
    heat_map = (res - np.min(res)) / (np.max(res) - np.min(res))
    return heat_map


def pred_MS_CAM(model, test_img):
    with tf.GradientTape(persistent=True) as gtape:
        middle = Model(inputs=model.input,
                       outputs=[model.output,
                                model.get_layer('max_pooling2d').output,
                                model.get_layer('max_pooling2d_1').output,
                                model.get_layer('max_pooling2d_2').output])
        pred, feature, feature_1, feature_2 = middle(test_img)
        prob = pred[:, np.argmax(pred[0])]
    grads = gtape.gradient(prob, feature)
    grads_1 = gtape.gradient(prob, feature_1)
    grads_2 = gtape.gradient(prob, feature_2)
    del gtape
    heat = np.multiply(grads[0], feature[0])
    heat = np.sum(heat, axis=2)
    heat = cv2.resize(heat, (32, 32), interpolation=cv2.INTER_AREA)

    heat_1 = np.multiply(grads_1[0], feature_1[0])
    heat_1 = np.sum(heat_1, axis=2)
    heat_1 = cv2.resize(heat_1, (32, 32), interpolation=cv2.INTER_AREA)

    heat_2 = np.multiply(grads_2[0], feature_2[0])
    heat_2 = np.sum(heat_2, axis=2)
    heat_2 = cv2.resize(heat_2, (32, 32), interpolation=cv2.INTER_AREA)

    heat_map = heat + heat_1 + heat_2
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
    return heat_map

def heat_map_demo(img_path):
    model = load_model('./CIFAR10_model_else.h5')
    ori_img = load_img(img_path, target_size=(32, 32, 3))
    test_img = np.expand_dims(ori_img, axis=0)
    test_img=test_img/255

    CAM = pred_CAM(model, test_img)
    GradCAM = pred_Grad_CAM(model, test_img)
    MSCAM = pred_MS_CAM(model, test_img)

    ax = plt.subplot(221)
    ax.imshow(ori_img)
    ax.set_title('origin img')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(222)
    ax.imshow(CAM, cmap='jet')
    ax.set_title('cam')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(223)
    ax.imshow(GradCAM, cmap='jet')
    ax.set_title('grad-cam')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(224)
    ax.imshow(MSCAM, cmap='jet')
    ax.set_title('ms-cam')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# train()
# predict()
heat_map_demo('airplane.jpg')
