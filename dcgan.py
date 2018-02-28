import argparse
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout
from keras.optimizers import SGD

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('relu'))
    model.add(Dense(128*7*7))
    model.add(Activation('relu'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(
        Conv2D(64, (5, 5),
               padding='same',
               input_shape=(28, 28, 1))
    )
    model.add(LeakyReLU(0.2))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), subsample=(2, 2))) # +
    model.add(LeakyReLU(0.2)) # +
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256)) # model.add(Dense(1024))
    # model.add(Activation('tanh'))
    model.add(LeakyReLU(0.2)) # +
    model.add(Dropout(0.5)) # +
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model # discriminatorは何を誤差として捉えているのか?

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def train(batch_size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    d = discriminator_model()
    g = generator_model()

    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True) # nesterov?
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer='SGD')
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        for index in range(X_train.shape[0]//batch_size):
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = X_train[index*batch_size:(index+1)*batch_size]
            generated_image = g.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_image)) # 元画像 + 生成した画像
            y = [1] * batch_size + [0] * batch_size # 正解画像に対するラベル => 1, 生成画像 => 0
            # discriminatorは, 元の画像を1, 生成された画像を0として学習していく
            d_loss = d.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False
            # generatorは, 本当の画像
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            print("batch {} d_loss {}".format(index, d_loss))
            d.trainable = True
            print("batch {} g_loss: {}".format(index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

def generate(batch_size, nice=False):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()

    if args.mode == "train":
        train(batch_size=args.batch_size)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
