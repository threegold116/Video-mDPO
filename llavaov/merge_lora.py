from peft import PeftModel
import sys
import argparse
sys.path.insert(0,"/share/home/jfliang/Project/Hall/Video-mDPO/")
from llava.model import LlavaQwenForCausalLM
from llava.model.builder import load_pretrained_model

def merge_lora(args):
    base_model= LlavaQwenForCausalLM.from_pretrained(args.model_base,device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.merge_and_unload()
    model.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False,default="/share/home/jfliang/Project/Hall/Video-mDPO/checkpoint/llava-onevision-qwen2-mdpo-lora-mdpoloss-10k/")
    parser.add_argument("--model-base", type=str, required=False,default="/share/home/jfliang/Weights/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--save-model-path", type=str, required=False,default="/share/home/jfliang/Project/Hall/Video-mDPO/checkpoint_merge/llava-onevision-qwen2-mdpo-10k")

    args = parser.parse_args()

    merge_lora(args)