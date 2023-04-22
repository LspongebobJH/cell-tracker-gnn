#!/bin/bashexport 
CUDA_VISIBLE_DEVICES=0

python run_feat_extract.py \
params.input_images="./data/CTC/Training/Fluo-N2DH-SIM+" \
params.input_masks="./data/CTC/Training/Fluo-N2DH-SIM+" \
params.input_seg="./data/CTC/Training/Fluo-N2DH-SIM+" \
params.output_csv="./data/basic_features/" \
params.sequences=['01','02'] \
params.seg_dir='_GT/TRA' \
params.basic=True