import os
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from sklearn.metrics import classification_report

from dataset import Dataset

def grid_search(train_labels: str, 
                test_labels: str, 
                output:str, 
                res:tuple=(120, 160), 
                batch_size:int=16, 
                epochs:int=20):
    """
    Runs a grid search over all known models.

    Params
    ------
        train_labels: str
            Path to training labels
        test_labels: str
            Path to testing labels
        output: str
            Path to output directory
        res: tuple
            Input resolution of network
        batch_size: int
            Batch size in case of lazy loading
        epochs: int
            Training epochs
    """

    # Data
    print("=> Loading data.")
    train = Dataset(train_labels, res=res, batch_size=batch_size)
    test = Dataset(test_labels, res=res, batch_size=batch_size)

    X_train, y_train = train.get_all()
    X_test, y_test = test.get_all()
    X_train = X_train[..., 0, None]
    X_test = X_test[..., 0, None]

    def net(x, num_classes=1):
        x = K.applications.resnet_v2.ResNet50V2(include_top=False, weights=None, input_shape=x.shape[1:])(x)
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(num_classes, activation="softmax", name="new_final_layer")(x)
        return x

    print("\n=> Training model.")
    input_tensor = K.layers.Input((160, 120, 1))
    output_tensor = net(input_tensor, num_classes=train.num_classes())
    model = K.Model(input_tensor, output_tensor)
    model.load_weights(os.path.join(output, "flir_pretrained_weights.h5"), by_name=True)

    model.compile(optimizer="sgd",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])

    # Train model
    model.fit(x=X_train, 
              y=y_train, 
              epochs=epochs, 
              validation_data=(X_test, y_test), 
              batch_size=batch_size,  
              verbose=2)

    # Save weights
    model.save_weights(os.path.join(output, "transfer_trained_weights.h5"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on FLIR dataset.")
    parser.add_argument("train", help="Directory containing training labels")
    parser.add_argument("test", help="Directory containing testing labels")
    parser.add_argument("out", help="Output directory for results")
    parser.add_argument("epochs", help="Number of epochs")
    args = vars(parser.parse_args())

    grid_search(args["train"], 
                args["test"], 
                args["out"], 
                res=(120, 160), 
                epochs=int(args["epochs"]))
    
    print("\n=> Finished.")