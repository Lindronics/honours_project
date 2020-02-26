#!/bin/bash

python3 nfs/honours_project/preprocessing/augment.py -d nfs/honours_project_data/animals_train_cleaned/images nfs/honours_project_data/animals_train_cleaned/images_augmented nfs/honours_project_data/animals_train_cleaned/labels.txt 5

python3 nfs/honours_project/preprocessing/augment.py -d nfs/honours_project_data/animals_test_cleaned/images nfs/honours_project_data/animals_test_cleaned/images_augmented nfs/honours_project_data/animals_test_cleaned/labels.txt 2