OMP_NUM_THREADS=32 
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
torchrun --nnodes=1 --master_port 49209 --nproc_per_node=$NUM_GPU  run_clear_pretraining.py\
        --output_dir  checkpoint/MIMIC-IV \
        --log_dir  checkpoint/MIMIC-IV \
        --model CLEAR \
        --batch_size 256 \
        --lr 5e-4 \
        --warmup_epochs 5 \
        --clip_grad 3.0 \
        --layer_scale_init_value 0.1 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 100 \
        --save_ckpt_freq 50 \
        --gradient_accumulation_steps 1 \
        --mask_ratio 0.8 \
        --world_size $NUM_GPU \
        --cls_token_num 12 \
        --dataset_name 'MIMIC-IV' \
        --atten_mask