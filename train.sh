#!/bin/bash

# Ensure the script stops on errors
set -e

# Define default values for environment variables (you can override these when running the script)
MODEL_NAME=${MODEL_NAME:-"sangmin6600/t5-v1_1-xl-ko"}
TOKENIZER_NAME=${TOKENIZER_NAME:-"sangmin6600/t5-v1_1-xl-ko"}  # Leave blank to use the model's tokenizer
DROPOUT_RATIO=${DROPOUT_RATIO:-0.2}
DATA_DIR=${DATA_DIR:-"./dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
TRAIN_RATIO=${TRAIN_RATIO:-0.98}
VALID_RATIO=${VALID_RATIO:-0.01}
TEST_RATIO=${TEST_RATIO:-0.01}
PROJECT_NAME=${PROJECT_NAME:-"braille-translator"}
RUN_NAME=${RUN_NAME:-"t5-xlarge-5epochs-from-scratch"}
BATCH_SIZE=${BATCH_SIZE:-12}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
EPOCHS=${EPOCHS:-8}
LOGGING_STEPS=${LOGGING_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-10000}
SEED=${SEED:-42}
OPTIM=${OPTIM:-"paged_adamw_8bit"}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}

# Run the Python script with the defined arguments
python main.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name_or_path $TOKENIZER_NAME \
    --dropout_rate $DROPOUT_RATIO \
    --data_dir $DATA_DIR \
    --train_ratio $TRAIN_RATIO \
    --valid_ratio $VALID_RATIO \
    --test_ratio $TEST_RATIO \
    --output_dir $OUTPUT_DIR \
    --project_name $PROJECT_NAME \
    --run_name $RUN_NAME \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --seed $SEED \
    --optim $OPTIM \
    --bf16 True \
    --greater_is_better False \
    --report_to wandb \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --warmup_ratio 0.05
