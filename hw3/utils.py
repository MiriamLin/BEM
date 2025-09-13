from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是一位精通古代文言文與現代中文的專業翻譯。你的任務是根據用戶的問題，在文言文和白話文之間進行精確、流暢的翻譯。以下為你要翻譯的句子，請直接給出正確翻譯。USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit = True,
        load_in_8bit = False,
        llm_int8_threshold = 6.0,
        llm_int8_has_fp16_weight = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
    )
