import tensorflow as tf


class UNet:
    def __init__(self, input_shape=(None, None, 1), depth=3, output_classes=2, output_activation='sigmoid', io_dim=2):
        self.input_shape = input_shape
        self.depth = depth
        self.output_classes = output_classes
        self.output_activation = output_activation
        if io_dim == 2:
            self.conv = tf.keras.layers.Conv2D
            self.conv_t = tf.keras.layers.Conv2DTranspose
            self.pool = tf.keras.layers.MaxPool2D
        elif io_dim == 3:
            self.conv = tf.keras.layers.Conv3D
            self.conv_t = tf.keras.layers.Conv3DTranspose
            self.pool = tf.keras.layers.MaxPool3D
        else:
            raise ValueError

    def __call__(self, backbone_encoder, backbone_decoder):
        # input
        inputs = tf.keras.Input(shape=self.input_shape)
        X = inputs

        # save blocks outputs
        encoder_out = []
        encoder_out_depth = []

        # encoder
        for i in range(self.depth - 1):
            X, depth = backbone_encoder(X, i)
            X = self.pool(2)(X)
            encoder_out.append(X)
            encoder_out_depth.append(depth)

        X, depth = backbone_encoder(X, self.depth - 1)
        encoder_out.append(X)
        encoder_out_depth.append(depth)

        # decoder
        for i in range(self.depth - 1,  0, -1):
            X = self.conv_t(encoder_out_depth[i], 2, 2, padding='valid')(X)
            X, _ = backbone_decoder(X, i-1)

        # output activation
        if self.output_classes > 2:
            outputs = self.conv(self.output_classes, 1, activation=self.output_activation, name="output")(X)
        else:
            outputs = self.conv(1, 1, activation=self.output_activation, name="output")(X)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TISM_UNET")
        return model
