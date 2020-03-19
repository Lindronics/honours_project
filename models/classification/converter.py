import tensorflow as tf

from models import ResNet

model = ResNet("fusion", 8, input_shape=(120, 160, 4)).get_model()
model.load_weights("/Users/lindronics/workspace/4th_year/out/out/ResNet/ResNet_fusion_weights.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
lite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(lite_model)
