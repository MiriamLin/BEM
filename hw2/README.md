# README

## B12902007 林映辰

1. 資料集存放在 data 資料夾中，分為 public.jsonl 和 train.jsonl，並開了空資料夾 output 存放模型、tokenizer 以及 learning curve data，以及 test_results 存放各種 generation strategies 生成的結果。

2. 在終端機進入資料夾，輸入以下指令，會執行 train.py，將 model 和 tokenizers 存放在 output 中。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path "google/mt5-small" \
  --tokenizer_name "google/mt5-small" \
  --train_file "data/train.jsonl" \
  --output_dir "./output" \
  --num_train_epochs 10 \
  --learning_rate 1e-3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_source_length 1024 \
  --max_target_length 128 \
  --num_beams 4 \
  --with_tracking \
  --seed 1 \
```

3. 若要 training 完直接在本地 testing，可輸入以下指令，會將 output.jsonl 存放在目前的資料夾中。

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --test_file "data/public.jsonl" \
  --output_file ./output.jsonl \
  --model_dir "output" \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_test_batch_size 4 \
  --num_beams 8 \
  --seed 1 \
```

4. 或者，可用以下指令下載放在 Google Drive 的 model 和 tokenizer。

<https://drive.google.com/drive/folders/1_UgK3zUIhcPSrXAxRQHJk_Xs928qZVuw?usp=sharing>

```bash
bash ./download.sh
```

5. 完成下載後，會把 model 和 tokenizer 下載在 output 資料夾中。

6. 使用以下指令進行 testing，會把 output.jsonl 存放在想存放的位置。

```bash
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```