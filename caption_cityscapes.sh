#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

python image_captioning.py \
    --model_name "blip2_t5" \
    --model_type "pretrain_flant5xxl" \
    --data_dir "./Cityscapes" \
    --dataset_format "cityscapes" \
    --sets "train-val-test" \
    --semantics_config "./semantic_configs/cityscapes.yaml" \
    --question_prompts "./question_prompts/cityscapes_questions.yaml" \
    --results_dir "./results/cityscapes"    
