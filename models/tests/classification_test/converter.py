import tensorflow as tf

model = tf.keras.models.load_model("test.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
lite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(lite_model)