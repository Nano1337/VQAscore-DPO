MODEL_VERSION=clip_flant5_xl_loraft_dpo_hardneg

TEXT_DPO_DATA=/home/haoli/Documents/t2v_metrics/datasets/GenAI-Image-527/genaibench_dpo_dataset.json
# MODEL_CKPT=/home/haoli/Documents/t2v_metrics/hf_cache/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965
# MODEL_CKPT=liuhaotian/llava-v1.5-7b
MODEL_CKPT=zhiqiulin/clip-flant5-xl
# MODEL_CKPT=/home/haoli/Documents/t2v_metrics/hf_cache/models--zhiqiulin--clip-flant5-xl/snapshots/2cee5afb3386792d9b21066a89232687dfce2c62

# textvqa is the placeholder to notate the GenAI Bench data
deepspeed seva/train_dpo_ours.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --deepspeed seva/scripts/zero3.json \
    --model_name_or_path ${MODEL_CKPT} \
    --version t5_chat \
    --textvqa_data_path ${TEXT_DPO_DATA} \
    --textvqa_image_path /home/haoli/Documents/t2v_metrics/datasets/GenAI-Image-527/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/${MODEL_VERSION} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bits 4 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${MODEL_VERSION} \
    --beta 0.1

# 4 bit training enables QLoRA training. Search script_args.bits in seva/train_dpo_ours.py for more details.

