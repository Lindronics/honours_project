import os
import argparse
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from sklearn.metrics import classification_report

from models import AlexNet, ResNet, CustomNet, ResNet152v2
from dataset import Dataset

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def grid_search(train_labels: str, 
                test_labels: str, 
                output:str, 
                res:tuple=(120, 160), 
                lazy:bool=True, 
                batch_size:int=16, 
                epochs:int=20, 
                register:bool=False):
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
        lazy: bool
            Whether to load data lazily in batches during training
        batch_size: int
            Batch size in case of lazy loading
        epochs: int
            Training epochs
        register: bool
            Whether to attempt registering the images
    """

    print("=> Starting grid search.")

    models = [CustomNet, AlexNet, ResNet]

    # Data
    print("=> Loading data.")
    train = Dataset(train_labels, res=res, batch_size=batch_size, register=register)
    test = Dataset(test_labels, res=res, batch_size=batch_size, register=register)

    if not lazy:
        X_train, y_train = train.get_all()
        X_test, y_test = test.get_all()

    # Models
    for model_type in models:
        print(f"\n=> ### Starting evaluation of {model_type.__name__}. ###")

        sub_path = os.path.join(output, model_type.__name__)
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)

        with open(os.path.join(sub_path, "report.txt"), "w") as f:
            title = f"##### Evaluation results for {model_type.__name__} #####"
            f.write("#" * len(title) + "\n")
            f.write(title + "\n")
            f.write("#" * len(title) + "\n\n")

        for mode in model_type.modes:
            print(f"\n=> Evaluating {mode} mode.")
            name_prefix = f"{model_type.__name__}_{mode}_"

            # Prepare model
            net = model_type(mode, num_classes=train.num_classes(), input_shape=train.shape(), weight_dir=output)
            model = net.get_model()

            optimizer = K.optimizers.Adam(learning_rate=0.000001, epsilon=0.005)
            model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

            # Train model
            if lazy:
                hist = model.fit(x=train, 
                                 epochs=epochs, 
                                 validation_data=train, 
                                 verbose=2)
            else:
                hist = model.fit(x=X_train, 
                                 y=y_train, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 validation_data=(X_test, y_test), 
                                 verbose=2)

            print("\n=> Saving weights and training history")
            # Save weights
            model.save_weights(os.path.join(sub_path, name_prefix + "weights.h5"))

            # Save history
            with open(os.path.join(sub_path, name_prefix + "history.pickle"), "wb") as f:
                pickle.dump(hist.history, f)

            # Evaluate
            print("\n=> Starting evaluation")
            with open(os.path.join(sub_path, "report.txt"), "a") as f:
                title = f"##### Evaluation results for {mode} mode #####"
                f.write("\n\n" + title + "\n")
                f.write("-" * len(title) + "\n")

                # Test classification report
                f.write("\n##### Test #####\n")
                if lazy:
                    y_pred = np.argmax(model.predict(test), axis=1)
                    y_test_ = test.get_labels()[:y_pred.shape[0]]
                else:
                    y_pred = np.argmax(model.predict(X_test), axis=1)
                    y_test_ = np.argmax(y_test, axis=1)
                f.write(classification_report(y_test_, y_pred, target_names=test.class_labels))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a grid search on all known models.")
    parser.add_argument("train", help="Labels file of training data")
    parser.add_argument("test", help="Labels file of validation data")
    parser.add_argument("out", help="Output directory for results")
    parser.add_argument("epochs", help="Number of epochs")
    parser.add_argument("batchsize", help="Batch size to use for training")
    parser.add_argument("-l", "--lazy", dest="lazy", help="Load data lazily", action="store_true")
    parser.add_argument("-s", "--small", dest="small", help="Use 160x120 resolution", action="store_true")
    parser.add_argument("-r", "--register", dest="register", help="Attempt to register images", action="store_true")
    args = vars(parser.parse_args())

    res = (120, 160) if args["small"] else (480, 640)

    grid_search(args["train"], 
                args["test"], 
                args["out"], 
                res=res, 
                lazy=bool(args["lazy"]), 
                epochs=int(args["epochs"]), 
                register=bool(args["register"]),
                batch_size=int(args["batchsize"]))
    
    print("\n=> Finished.")