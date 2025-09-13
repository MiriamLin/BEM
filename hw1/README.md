# README

b12902007 林映辰
1. 資料集存放在 dataset 資料夾中，分為 train.json, valid.json, context.json, text.json，並開了一個空資料夾 output 存放訓練結果。

2. 訓練 Multiple choice 的模型
在終端機進入資料夾，輸入以下指令，會將 model 和 tokenizers 存放在 output/train_mc 中。
```
CUDA_VISIBLE_DEVICES=0 python train_mc.py \
    --model_type bert \
    --tokenizer_name luhua/chinese_pretrain_mrc_macbert_large \
    --model_name_or_path luhua/chinese_pretrain_mrc_macbert_large \
    --train_file dataset/train.json \
    --validation_file dataset/valid.json \
    --context_file dataset/context.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --output_dir output/train_mc \
    --with_tracking \
```

3. 訓練 Extractive QA 的模型
在終端機進入資料夾，輸入以下指令，會將 model 和 tokenizers 存放到 output/train_qa 中。
```
CUDA_VISIBLE_DEVICES=0 python train_qa.py \
    --model_type bert \
    --tokenizer_name luhua/chinese_pretrain_mrc_macbert_large \
    --model_name_or_path luhua/chinese_pretrain_mrc_macbert_large \
    --train_file dataset/train.json \
    --validation_file dataset/valid.json \
    --context_file dataset/context.json \
    --test_file dataset/test.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --output_dir output/train_qa \
    --with_tracking \
```

4. 若在本地直接 test，可輸入以下指令，會將 prediction.csv 存在目前的資料夾中。(在test.py已先設定好 mc_model_dir 和 qa_model_dir 的路徑，分別為 output/train_mc 和 output/train_qa)
```
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_type bert \
    --tokenizer_name luhua/chinese_pretrain_mrc_macbert_large \
    --model_name_or_path luhua/chinese_pretrain_mrc_macbert_large \
    --context_file dataset/context.json \
    --test_file dataset/test.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --output_dir output \
    --output_prediction_file ./prediction.csv \
    --with_tracking \
```

5. 或者，可用以下指令下載放在 Google Drive 的 models, tokenizers 和 data。
https://drive.google.com/drive/folders/1s2R8jK0-SLaa27lwfyGR7o9ssPWdoNDT?usp=sharing
```
bash ./download.sh
```

6. 完成以上操作後，會下載下來兩個資料夾。dataset 內包含 context.json, test.json, train.json, valid.json ; output 內包含 train_mc 和 train_qa 兩個資料夾，train_mc 內有 Multiple choice 的 model 和 tokenizers，train_qa 內有 Extractive QA 的 model 和 tokenizers。

7. 使用以下指令，用run.sh進行 testing，會將 prediction.csv 存在目前資料夾中。
```
bash ./run.sh dataset/context.json dataset/test.json ./prediction.csv
```
