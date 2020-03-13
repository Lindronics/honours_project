#!/bin/bash

python3 nfs/honours_project/preprocessing/augment.py -d nfs/honours_project_data/animals_train_final/images nfs/honours_project_data/animals_train_final/images_augmented nfs/honours_project_data/animals_train_final/labels.txt 5

python3 nfs/honours_project/preprocessing/augment.py -d nfs/honours_project_data/animals_test_final/images nfs/honours_project_data/animals_test_final/images_augmented nfs/honours_project_data/animals_test_final/labels.txt 2