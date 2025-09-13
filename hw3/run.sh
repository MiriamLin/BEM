BASE_MODEL_PATH=$1
ADAPTER_CHECKPOINT=$2
INPUT_FILE=$3
OUTPUT_FILE=$4

python test.py \
    --base_model_path $BASE_MODEL_PATH \
    --adapter_checkpoint $ADAPTER_CHECKPOINT \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \