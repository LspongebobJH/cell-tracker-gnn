#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# export HYDRA_FULL_ERROR=1

# python run_feat_extract.py \
# params.input_images="./data/CTC/Training/Fluo-N2DH-SIM+" \
# params.input_masks="./data/CTC/Training/Fluo-N2DH-SIM+" \
# params.input_seg="./data/CTC/Training/Fluo-N2DH-SIM+" \
# params.output_csv="./data/basic_features/" \
# params.sequences=['01','02'] \
# params.seg_dir='_GT/TRA' \
# params.basic=True

# python run_train_metric_learning.py \
# dataset.kwargs.data_dir_img="./data/CTC/Training/Fluo-N2DH-SIM+" \
# dataset.kwargs.data_dir_mask="./data/CTC/Training/Fluo-N2DH-SIM+" \
# dataset.kwargs.dir_csv="./data/basic_features/Fluo-N2DH-SIM+" \
# dataset.kwargs.subdir_mask='GT/TRA'

# python run_feat_extract.py \
# params.input_images="./data/CTC/Training/Fluo-N2DH-SIM+" \
# params.input_masks="./data/CTC/Training/Fluo-N2DH-SIM+" \
# params.input_seg="./data/CTC/Training/Fluo-N2DH-SIM+" \
# params.output_csv="./data/ct_features/" \
# params.sequences=['01','02']  \
# params.seg_dir='_GT/TRA' \
# params.basic=False \
# params.input_model="/home/ubuntu/cell-tracker-gnn/outputs/2023-04-22/13-26-13/all_params.pth"

python run.py \
datamodule.dataset_params.main_path="./data/ct_features/Fluo-N2DH-SIM+" \
datamodule.dataset_params.exp_name="2D_SIM" \
datamodule.dataset_params.drop_feat=[]