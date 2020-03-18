import os
import argparse
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from sklearn.metrics import classification_report

from models import ResNet
from dataset import Dataset

def grid_search(train_labels: str, 
                test_labels: str, 
                output:str, 
                res:tuple=(120, 160), 
                epochs:int=20, 
                register:bool=False):
    """
    Runs a grid search over different batch sizes.

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
        epochs: int
            Training epochs
        register: bool
            Whether to attempt registering the images
    """

    print("=> Starting batch size grid search.")
    BATCH_SIZES = [2, 8, 32, 64, 128, 512]

    # Data
    print("=> Loading data.")
    train = Dataset(train_labels, res=res, batch_size=1, register=register)
    test = Dataset(test_labels, res=res, batch_size=1, register=register)

    X_train, y_train = train.get_all()
    X_test, y_test = test.get_all()

    # Models
    sub_path = os.path.join(output, "batch_size")
    if not os.path.isdir(sub_path):
        os.mkdir(sub_path)

    with open(os.path.join(sub_path, "report.txt"), "w") as f:
        title = f"##### Evaluation results for batch size grid search #####"
        f.write("#" * len(title) + "\n")
        f.write(title + "\n")
        f.write("#" * len(title) + "\n\n")

    for batch_size in BATCH_SIZES:
        print(f"\n=> Evaluating batch size {batch_size}.")
        name_prefix = f"{batch_size}_batches_"

        # Prepare model
        net = ResNet("fusion", num_classes=train.num_classes(), input_shape=train.shape(), weight_dir=output)
        model = net.get_model()

        optimizer = K.optimizers.Adam(learning_rate=0.00001, epsilon=0.005)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # Train model
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
            title = f"##### Evaluation results for batch size {batch_size} #####"
            f.write("\n\n" + title + "\n")
            f.write("-" * len(title) + "\n")

            # Test classification report
            f.write("\n##### Test #####\n")
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_ = np.argmax(y_test, axis=1)
            f.write(classification_report(y_test_, y_pred, target_names=test.class_labels))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a grid search on all known models.")
    parser.add_argument("train", help="Labels file of training data")
    parser.add_argument("test", help="Labels file of validation data")
    parser.add_argument("out", help="Output directory for results")
    parser.add_argument("epochs", help="Number of epochs")
    parser.add_argument("-l", "--lazy", dest="lazy", help="Load data lazily", action="store_true")
    parser.add_argument("-r", "--register", dest="register", help="Attempt to register images", action="store_true")
    args = vars(parser.parse_args())

    grid_search(args["train"], 
                args["test"], 
                args["out"], 
                res=(120, 160), 
                lazy=bool(args["lazy"]), 
                epochs=int(args["epochs"]), 
                register=bool(args["register"]))
    
    print("\n=> Finished.")