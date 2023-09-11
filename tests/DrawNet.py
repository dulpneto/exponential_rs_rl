import sys

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils.vis_utils import plot_model

import visualkeras
from PIL import ImageFont

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

QUIET = True

def main():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu',
                                       input_shape=(100, 100, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(216, activation='relu'))
    model.add(keras.layers.Dense(4, activation=None, name="OutputLayer"))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7))


    font = ImageFont.truetype("Keyboard.ttf", 16)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model,padding=10, scale_z=0.1, scale_xy=4,legend=True, font=font, to_file='network.png', spacing=50)  # font is optional!
    #plot_model(model, to_file='network_plot.png', show_shapes=True, show_layer_names=True)





if __name__ == "__main__":
    main()



