#!/usr/bin/env bash
python main.py --model_name res18_1 --wd 0e0 --lr 1e-1 --optim sgd --num_epoch 200
python main.py --model_name res18_2 --wd 5e-4 --lr 1e-1 --optim sgd --num_epoch 200

python main.py --model_name res18_3 --wd 5e-4 --lr 1e-3 --optim adam --num_epoch 200
python main.py --model_name res18_4 --wd 5e-4 --lr 1e-4 --optim adam --num_epoch 200
python main.py --model_name res18_5 --wd 0e0 --lr 1e-3 --optim adam --num_epoch 200
python main.py --model_name res18_6 --wd 5e-4 --lr 1e-3 --optim adam --num_epoch 200