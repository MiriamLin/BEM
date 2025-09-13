INPUT_FILE=$1
OUTPUT_FILE=$2

CUDA_VISIBLE_DEVICES=0 python test.py \
  --test_file $INPUT_FILE \
  --output_file $OUTPUT_FILE \
  --model_dir "output" \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_test_batch_size 4 \
  --num_beams 8 \
  --seed 1 \
