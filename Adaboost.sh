#!/usr/bin/env bash
python main_for_Adaboost.py --name A1 --pca_dim 320 --n_estimators 100 --max_depth 4
python main_for_Adaboost.py --name A2 --pca_dim 320 --n_estimators 200 --max_depth 4
python main_for_Adaboost.py --name A3 --pca_dim 320 --n_estimators 300 --max_depth 4
python main_for_Adaboost.py --name A4 --pca_dim 320 --n_estimators 400 --max_depth 4
python main_for_Adaboost.py --name A5 --pca_dim 320 --n_estimators 500 --max_depth 4
python main_for_Adaboost.py --name A6 --pca_dim 320 --n_estimators 600 --max_depth 4