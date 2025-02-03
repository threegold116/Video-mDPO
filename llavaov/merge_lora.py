from peft import PeftModel
import sys
import argparse
import os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.model import LlavaQwenForCausalLM
import torch
from transformers import AutoTokenizer
def merge_lora(args):
    base_model= LlavaQwenForCausalLM.from_pretrained(args.model_base,device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.merge_and_unload()
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False,default="/share/home/jfliang/Project/Hall/Video-mDPO/checkpoint/llava-onevision-qwen2-mdpo-lora-mdpoloss-10k/")
    parser.add_argument("--model-base", type=str, required=False,default="/share/home/jfliang/Weights/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--save-model-path", type=str, required=False,default="/share/home/jfliang/Project/Hall/Video-mDPO/checkpoint_merge/llava-onevision-qwen2-mdpo-10k")
    args = parser.parse_args()
    print("开始合并模型...")
    merge_lora(args)