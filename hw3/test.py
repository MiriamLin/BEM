import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, get_bnb_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        required=True
    )
    parser.add_argument(
        "--adapter_checkpoint",
        type=str,
        required=True
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        required=True
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        required=True
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = PeftModel.from_pretrained(model, args.adapter_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.input_file, "r") as f:
        data = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    results = []
    
    for sample in data:
        instruction = sample["instruction"]
        id = sample["id"]
        prompt = get_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        output_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        if "ASSISTANT:" in output_text:
            output_text = output_text.split("ASSISTANT:")[-1].strip()
        results.append({'id': id, 'output': output_text})
        
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
