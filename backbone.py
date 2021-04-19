import tensorflow as tf


class VGG:
    def __init__(self, initial_block_depth=32, length=2, activation='relu', kernel_size=3, io_dim=2):
        self.initial_block_depth = initial_block_depth
        self.length = length
        self.activation = activation
        self.kernel_size = kernel_size
        if io_dim == 2:
            self.conv = tf.keras.layers.Conv2D
        elif io_dim == 3:
            self.conv = tf.keras.layers.Conv3D
        else:
            raise ValueError

    def __call__(self, X, level):
        block_depth = self.initial_block_depth * 2**level

        for _ in range(self.length):
            X = self.conv(filters=block_depth, kernel_size=self.kernel_size,
                          kernel_initializer="he_normal", padding="same")(X)
            X = tf.keras.layers.Activation(self.activation)(X)

        return X, block_depth
