import tensorflow.keras as K
from models import AlexNet

model = AlexNet("fusion", 8, (160, 120, 4)).get_model()


model.compile(optimizer="sgd",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

K.utils.plot_model(model, to_file="asdf.png")