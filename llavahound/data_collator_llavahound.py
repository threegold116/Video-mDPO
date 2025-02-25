from dataclasses import dataclass, field
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image
import transformers
from llava_hound_model.mm_utils import tokenizer_image_token,process_images
from decord import VideoReader
from llava_hound_model.constants import IGNORE_INDEX,X_TOKEN_INDEX
import numpy as np
from llava_hound_model import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
from llava_hound_model.mm_utils import tokenizer_X_token
import torch
from decord import cpu
from torch.nn.utils.rnn import pad_sequence
import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    # frame_idx = [i for i in range(0, len(vr), fps)]
    # frame_idx = list(range(0,len(vr),len(vr)//max_frames_num))
    # if len(frame_idx) > max_frames_num or force_sample:
    sample_fps = max_frames_num
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    X: str = None
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if X is not None:
        input_ids = torch.stack([tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) #设置为eos容易出现错位，在统计时

        rounds = conversation.split(conv.sep2)
        cur_len = 1 # 1 for bos
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if X is not None:
                round_len = len(tokenizer_X_token(rou, tokenizer, X_TOKEN_INDEX[X]))
                instruction_len = len(tokenizer_X_token(parts[0], tokenizer, X_TOKEN_INDEX[X])) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )




@dataclass
class mDPODataCollatorLlaVAHound(DPODataCollatorWithPadding):
    #THREEGOLD CHANGE
    max_frames_num: int = 16
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        video_path: str,
        perturbation: str = None,
    ) -> Dict:
        batch = {}
        #THREEGOLD CHANGE: 将prompt中的<image>替换为空字符串，并添加<image>到prompt的开头 参考llavanext中dpo的实现
        prompt = prompt.replace("<video>", "").strip()
        prompt = "<video>\n" + prompt
        
        chosen_sources = make_conv(prompt, chosen)
        rejected_sources = make_conv(prompt, rejected)
        if perturbation!=None:
            perturbation_prompt = perturbation+"\n"+prompt
            perturbation_prompt = perturbation_prompt.replace("<video>", "").strip()
            perturbation_prompt = "<video>\n" + perturbation_prompt
            perturbation_chosen_sources = make_conv(perturbation_prompt, chosen)
        else:
            perturbation_chosen_sources = {}
        
        perturbation_data_dict = preprocess_v1([perturbation_chosen_sources], self.tokenizer, X="VIDEO")
            
        chosen_data_dict = preprocess_v1([chosen_sources], self.tokenizer, X="VIDEO")
        # chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = preprocess_v1([rejected_sources], self.tokenizer, X="VIDEO")
        # rejected_data_dict['attention_mask'] = rejected_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        chosen_sequence_tokens = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_sequence_tokens = {k: v[0] for k, v in rejected_data_dict.items()}
        perturbation_sequence_tokens = {k: v[0] for k, v in perturbation_data_dict.items()}
        
        prompt_tokens = {}
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]
        

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
            "perturbation": perturbation_sequence_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        #这里可能得根据视频修改\
        video_tensor = self.model.process_video(video_path,num_frames=self.max_frames_num).to(dtype=self.model.dtype)
        # batch["video"] = video_tensor
        # if self.data_args.add_time_instruction:
        #     time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        #     sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
        # image = Image.open(img_path)#打开图片
        # image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)#将图片转换为tensor
        assert video_tensor.shape[1] == self.max_frames_num
        batch["image"] = video_tensor
        # batch["image_size"] =video_frames[0].size
        batch["has_X"] = "video"
        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            Xs, keys = [], []
            for feature in features:
                prompt = feature["prompt"]#读取prompt
                chosen = feature["chosen"]#读取chosen
                rejected = feature["rejected"]#读取rejected
                video_path = feature["video_path"]#读取img_path
                if "perturbation" in feature:
                    perturbation = feature["perturbation"]#读取Perturbation
                else:
                    perturbation = None
                
                batch_element = self.tokenize_batch_element(prompt, chosen, rejected, video_path,perturbation)#调用tokenize_batch_element函数
                tokenized_batch.append(batch_element)
            
            collated_batch = self.collate(tokenized_batch)
            #THREEGOLD NEED CHANGE：增加
            keys = [instance["has_X"] for instance in tokenized_batch]
            Xs = [instance["image"] for instance in tokenized_batch]
            #THREEGOLD NEED CHANGE：增加
            collated_batch["image"] = [Xs, keys]  # we do not change the key's name.
            return collated_batch
    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ["chosen_input_ids", "rejected_input_ids","perturbation_input_ids"]:
            attn_k = k.replace("input_ids", "attention_mask")
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)
        return padded_batch