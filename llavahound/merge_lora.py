from peft import PeftModel
import sys
import argparse
import os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava_hound_model.model import LlavaLlamaForCausalLM
import torch
from transformers import AutoTokenizer
def merge_lora(args):
    base_model= LlavaLlamaForCausalLM.from_pretrained(args.model_base,device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = PeftModel.from_pretrained(base_model, args.model_path,device_map="auto")
    model = model.merge_and_unload()
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False,default="/home/sxjiang/myproject/Hall/Video-mDPO/checkpoint_lora/llava-hound-mdpo-lora-mdpoloss-per-10k-shuffle-frames")
    parser.add_argument("--model-base", type=str, required=False,default="/home/sxjiang/model/LLaVA-Hound-SFT")
    parser.add_argument("--save-model-path", type=str, required=False,default="/home/sxjiang/myproject/Hall/Video-mDPO/checkpoint_merge/llava-hound-mdpo-mdpoloss-per-10k-shuffle-frames")

    args = parser.parse_args()

    merge_lora(args)