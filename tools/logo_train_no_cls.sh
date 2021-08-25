#/bin/bash

MASTER_ADDR='127.0.0.1' MASTER_PORT='29500' WORLD_SIZE=1 RANK=0 PYTHONPATH=Robust_Logo_Detection/ python3 Robust_Logo_Detection/tools/train.py Robust_Logo_Detection/configs/robustlogodet/robust_logo_r50_rfp_1x__no_cls.py --launcher pytorch