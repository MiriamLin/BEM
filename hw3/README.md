# README

## B12902007 林映辰

1. 資料集存放在 data 資料夾中，分為 train.json, public_test.json, private_test.json。並開了空資料夾 adapter_checkpoint 存放每 100 步保存的模型。

2. 在終端機進入資料夾，輸入以下指令，會執行 train.py 開始進行 QLoRA fine tuning。（我是在工作站進行訓練，需要指定 gpu 並把 HF_HOME 設到 /tmp2 下面）

```bash
HF_HOME=/tmp2/b12902007/.cache/huggingface CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" \
    --dataset data/train.json \
    --output_dir ./adapter_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --do_train \
    --bf16 \
    --bits 4 \
    --seed 1 \
    --max_steps 1200 \
    --save_steps 100 \
```

3. training 結束後，開始用我自己修改的 myppl.py，分別對於 checkpoint-100 到 checkpoint-1200 算 mean perplexity，並將資料寫入 json 檔案中進行繪圖。

我使用以下指令進行 ppl 測試，目前每個 checkpoint 的資料都存在 adapter_checkpoint 中，會輸出 learning_curve.json 紀錄每一個 checkpoint 的 perplexity。

```bash
HF_HOME=/tmp2/b12902007/.cache/huggingface CUDA_VISIBLE_DEVICES=0 python myppl.py --base_model_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" --checkpoints_dir "./adapter_checkpoint" --test_data_path "./data/public_test.json"
```

4. 在 learning_curve.json 中我發現表現最好的是 checkpoint-500 所以我就將其他都刪除，並將 checkpoint-500 放在 [google drive](https://drive.google.com/drive/folders/14QoQviwdCe-rhYSu6Vt702k_VhfEVQ5D)。可用以下指令下載：

```bash
bash ./download.sh
```

5. 完成下載後，使用以下指令進行 testing。

```bash
bash run.sh \
    /path/to/`zake7749/gemma-2-2b-it-chinese-kyara-dpo` \
    /path/to/adapter_checkpoint/under/your/folder \
    /path/to/input \
    /path/to/output
```