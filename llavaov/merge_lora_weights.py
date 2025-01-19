import argparse
import sys
sys.path.insert(0,"/share/home/jfliang/Project/Hall/Video-mDPO/")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map="cpu")

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False,default="/share/home/jfliang/Project/Hall/Video-mDPO/checkpoint/llava-onevision-qwen2-mdpo-lora-debug/")
    parser.add_argument("--model-base", type=str, required=False,default="/share/home/jfliang/Weights/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--save-model-path", type=str, required=False,default="/share/home/jfliang/Project/Hall/Video-mDPO/checkpoint_merge/mdpo_llavaov2")

    args = parser.parse_args()

    merge_lora(args)
