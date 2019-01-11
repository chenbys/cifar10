#!/usr/bin/env bash
python main.py --model_name vgg16_1 --wd 0e0 --lr 1e-1 --optim sgd --num_epoch 200
python main.py --model_name vgg16_2 --wd 5e-4 --lr 1e-1 --optim sgd --num_epoch 200

python main.py --model_name vgg16_3 --wd 5e-4 --lr 1e-3 --optim adam --num_epoch 200
python main.py --model_name vgg16_4 --wd 5e-4 --lr 1e-4 --optim adam --num_epoch 200
python main.py --model_name vgg16_5 --wd 0e0 --lr 1e-3 --optim adam --num_epoch 200
python main.py --model_name vgg16_6 --wd 5e-4 --lr 1e-3 --optim adam --num_epoch 200