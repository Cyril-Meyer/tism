import tensorflow as tf
import architecture
import backbone


def get(architecture=architecture.UNet(input_shape=(256, 256, 1)), backbone_encoder=backbone.VGG(), backbone_decoder=backbone.VGG(length=1)):
    return architecture(backbone_encoder, backbone_decoder)


model = get()
model.summary()
