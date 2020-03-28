import os
import argparse
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from models import ResNet
from dataset import Dataset

def grid_search(train_labels: str, 
                test_labels: str, 
                output:str, 
                res:tuple=(120, 160), 
                epochs:int=20,
                n_splits:int=5,
                register:bool=False):
    """
    Runs a k-fold CV.

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
        n_splits: int
            Number of CV splits
        register: bool
            Whether to attempt registering the images
    """

    print("=> Starting k-fold.")

    # Data
    print("=> Loading data.")
    train = Dataset(train_labels, res=res, batch_size=1, register=register)
    test = Dataset(test_labels, res=res, batch_size=1, register=register)

    X_train, y_train = train.get_all()
    X_test, y_test = test.get_all()

    y_train = np.argmax(y_train, axis=1)[:, None]
    label_encoder = OneHotEncoder(sparse=False)
    label_encoder.fit(y_train)

    sub_path = os.path.join(output, "kfold")
    if not os.path.isdir(sub_path):
        os.mkdir(sub_path)

    with open(os.path.join(sub_path, "report.txt"), "w") as f:
        title = f"##### Evaluation results for k-fold CV #####"
        f.write("#" * len(title) + "\n")
        f.write(title + "\n")
        f.write("#" * len(title) + "\n\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print(f"=> Running Fold {i+1}/{n_splits}")
        name_prefix = f"fold_{i}_"

        X_train_ = X_train[train_index]
        y_train_ = y_train[train_index]
        y_train_ = label_encoder.transform(y_train_)

        # Prepare model
        net = ResNet("fusion", num_classes=train.num_classes(), input_shape=train.shape(), weight_dir=output)
        model = net.get_model()

        optimizer = K.optimizers.Adam(learning_rate=0.000001, epsilon=0.005)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # Train model
        hist = model.fit(x=X_train_, 
                         y=y_train_, 
                         epochs=epochs, 
                         batch_size=32, 
                         validation_data=(X_test, y_test), 
                         verbose=2)

        # Save history
        print("\n=> Saving training history")
        with open(os.path.join(sub_path, name_prefix + "history.pickle"), "wb") as f:
            pickle.dump(hist.history, f)

        # Evaluate
        print("\n=> Starting evaluation")
        with open(os.path.join(sub_path, "report.txt"), "a") as f:
            title = f"##### Evaluation results for fold {i} #####"
            f.write("\n\n" + title + "\n")
            f.write("-" * len(title) + "\n")

            # Test classification report
            f.write("\n##### Test #####\n")
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_ = np.argmax(y_test, axis=1)
            f.write(classification_report(y_test_, y_pred, target_names=test.class_labels))

        # Save classification report
        with open(os.path.join(sub_path, name_prefix + f"report_{i}.pickle")) as f:
            pickle.dump(classification_report(y_test_, y_pred, target_names=test.class_labels, output_dict=True), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a grid search on all known models.")
    parser.add_argument("train", help="Labels file of training data")
    parser.add_argument("test", help="Labels file of validation data")
    parser.add_argument("out", help="Output directory for results")
    parser.add_argument("epochs", help="Number of epochs")
    parser.add_argument("n_splits", help="Number CV splits")
    parser.add_argument("-l", "--lazy", dest="lazy", help="Load data lazily", action="store_true")
    parser.add_argument("-r", "--register", dest="register", help="Attempt to register images", action="store_true")
    args = vars(parser.parse_args())

    grid_search(args["train"], 
                args["test"], 
                args["out"], 
                res=(120, 160), 
                epochs=int(args["epochs"]), 
                n_splits=int(args["n_splits"]), 
                register=bool(args["register"]))
    
    print("\n=> Finished.")