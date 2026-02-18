model_name_or_path="unsloth/Meta-Llama-3.1-8B-Instruct"

export WANDB_PROJECT="insur-raft-FFT-top3"
export WANDB_ENTITY="experiment_team"
export OMP_NUM_THREADS="16"
export TOKENIZERS_PARALLELISM="false"
# export NCCL_P2P_DISABLE=1

CUDA_ARCH_LIST=`CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"`
arch="`echo $CUDA_ARCH_LIST | cut -c2`.`echo $CUDA_ARCH_LIST | cut -c5`"

exp_name="profit-llama"
TORCH_CUDA_ARCH_LIST=$arch \
deepspeed --include localhost:0,1 --master_port 61000 train_fft.py \
    --output_dir=/data/shyu/outputs/${exp_name} \
    --dataset_name=none \
    --model_name_or_path=${model_name_or_path} \
    --learning_rate=3e-6 \
    --lr_scheduler_type=cosine \
    --weight_decay=0.01 \
    --warmup_ratio=0.1 \
    --max_length=131072 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --eval_strategy=no \
    --num_train_epochs=3 \
    --save_strategy=no \
    --logging_strategy=steps \
    --logging_steps=1 \
    --save_total_limit=1 \
    --remove_unused_columns=False \
    --dataloader_num_workers=32 \
    --gradient_checkpointing=True \
    --torch_compile=True \
    --neftune_noise_alpha=5 \
    --optim=adamw_torch \
    --report_to=wandb \
    --run_name=${exp_name} \
    --dtype=bfloat16 \
    --use_liger_kernel=False \
    --padding_free=True \
    --attn_implementation=flash_attention_2 \
    --use_profit_loss=True \
    --prob_threshold=0.3 \
    --threshold_direction=higher \
    --deepspeed=./config/zero2_llama_v0.6.json