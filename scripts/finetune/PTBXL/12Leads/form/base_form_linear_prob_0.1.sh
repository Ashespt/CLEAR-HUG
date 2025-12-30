#!/bin/bash

# Define the split_ratios array to use
split_ratios=(0.1)


sampling_methods=("random")


export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1


generate_random_port() {
    while :; do
        PORT=$(( ( RANDOM % 63000 )  + 2000 ))
        ss -lpn | grep -q ":$PORT " || break
    done
    echo $PORT
}
# # Train
# 
rm -r checkpoints/finetune_ours_test_0.1/ptbxl/form/
for ratio in "${split_ratios[@]}"
do
    for method in "${sampling_methods[@]}"
    do
        PORT=$(generate_random_port)
        echo "Running training with split_ratio: $ratio and sampling_method: $method on port: $PORT"
        torchrun --nnodes=1 --master_port=$PORT --nproc_per_node=1 run_class_finetuning.py \
            --dataset_dir datasets/ecg_datasets/PTBXL_QRS_12Leads_ours_mask_missuniform/form \
            --output_dir checkpoints/finetune_ours_test_0.1/ptbxl/form/finetune_form_base_linear_${ratio}_${method}/ \
            --log_dir checkpoints/finetune_ours_test_0.1/ptbxlfinetune_form_base_linear_${ratio}_${method} \
            --model CLEAR_finetune_base \
            --trainable linear \
            --split_ratio $ratio \
            --finetune ./released_ckpt.pth \
            --sampling_method $method \
            --weight_decay 0.05 \
            --batch_size 256 \
            --lr 5e-3 \
            --update_freq 1 \
            --warmup_epochs 10 \
            --epochs 100 \
            --layer_decay 0.9 \
            --save_ckpt_freq 100 \
            --seed 0 \
            --is_binary \
            --nb_classes 19 \
            --world_size 1 \
            --atten_mask \
            --cls_token_num 12 \
            --mask_ratio 0
    done
done

# Test

# 
for ratio in "${split_ratios[@]}"
do
    
    for method in "${sampling_methods[@]}"
    do
        
        PORT=$(generate_random_port)
        echo "Running testing with split_ratio: $ratio and sampling_method: $method on port: $PORT"
        torchrun --nnodes=1 --master_port=$PORT --nproc_per_node=1 run_class_finetuning.py \
            --dataset_dir datasets/ecg_datasets/PTBXL_QRS_12Leads_ours_mask_missuniform/form \
            --output_dir checkpoints/finetune_ours_test_0.1/ptbxl/form/finetune_form_base_linear_${ratio}_${method}/ \
            --log_dir log/finetune_test_ours_0.1/finetune_form_base_linear_${ratio}_${method} \
            --model CLEAR_finetune_base \
            --eval \
            --trainable linear \
            --split_ratio $ratio \
            --sampling_method $method \
            --batch_size 256 \
            --seed 0 \
            --is_binary \
            --nb_classes 19 \
            --world_size 1 \
            --atten_mask \
            --cls_token_num 12 \
            --mask_ratio 0
    done
done
