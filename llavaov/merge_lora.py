from peft import PeftModel
import sys
import argparse
sys.path.insert(0,"/share/home/jfliang/Project/Hall/Video-mDPO/")
from llava.model import LlavaQwenForCausalLM
from llava.model.builder import load_pretrained_model
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
    parser.add_argument("--model-path", type=str, required=False,default="/home/sxjiang/myproject/Hall/Video-mDPO/checkpoint_lora/llava-onevision-qwen2-mdpo-lora-mdpoloss-per-10k-replace-frames/")
    parser.add_argument("--model-base", type=str, required=False,default="/home/sxjiang/model/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--save-model-path", type=str, required=False,default="/home/sxjiang/myproject/Hall/Video-mDPO/checkpoint_merge/llava-onevision-qwen2-mdpo-per-10k-replace-frames")

    args = parser.parse_args()

    merge_lora(args)