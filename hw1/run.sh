CONTEXT_FILE=$1
TEST_FILE=$2
PREDICTION=$3

CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_type bert \
    --tokenizer_name luhua/chinese_pretrain_mrc_macbert_large \
    --model_name_or_path luhua/chinese_pretrain_mrc_macbert_large \
    --context_file $CONTEXT_FILE \
    --test_file $TEST_FILE \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --output_dir output \
    --output_prediction_file $PREDICTION \
    --with_tracking \