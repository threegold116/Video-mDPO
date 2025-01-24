from dataclasses import dataclass, field
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image
import transformers
from llava.mm_utils import tokenizer_image_token,process_images
from llava.utils import process_video_with_decord
from decord import VideoReader
from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX,IGNORE_INDEX
import numpy as np
import torch
from decord import cpu
from torch.nn.utils.rnn import pad_sequence

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
def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and "<image>" in sentence["value"]:
                assert sentence["value"].startswith("<image>"), print(sentence["value"])

                _input_id = tokenizer(role).input_ids + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<image>") :]).input_ids + [im_end] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
    )




@dataclass
class mDPODataCollatorLlaVAOV(DPODataCollatorWithPadding):
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
        prompt = prompt.replace("<image>", "").strip()
        prompt = "<image>\n" + prompt
        
        chosen_sources = make_conv(prompt, chosen)
        rejected_sources = make_conv(prompt, rejected)
        if perturbation!=None:
            perturbation_prompt = perturbation+"\n"+prompt
            perturbation_prompt = perturbation_prompt.replace("<image>", "").strip()
            perturbation_prompt = "<image>\n" + perturbation_prompt
            perturbation_chosen_sources = make_conv(perturbation_prompt, chosen)
        else:
            perturbation_chosen_sources = {}
        
        perturbation_data_dict = preprocess_qwen([perturbation_chosen_sources], self.tokenizer, has_image=True)
            
        chosen_data_dict = preprocess_qwen([chosen_sources], self.tokenizer, has_image=True)
        # chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = preprocess_qwen([rejected_sources], self.tokenizer, has_image=True)
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
        #这里可能得根据视频修改
        video_frames,frame_time,video_time = load_video(video_path, self.max_frames_num,fps=1,force_sample=False)
        video_tensor = self.model.process_video(video_frames).to(dtype=self.model.dtype)
        # batch["video"] = video_tensor
        # if self.data_args.add_time_instruction:
        #     time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        #     sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
        # image = Image.open(img_path)#打开图片
        # image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)#将图片转换为tensor
        assert video_tensor.shape[0] == self.max_frames_num
        batch["image"] = video_tensor
        batch["image_size"] =video_frames[0].size
        batch["modality"] = "video"
        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

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
            collated_batch["image_sizes"] = [instance["image_size"] for instance in tokenized_batch]
            collated_batch["modalities"] = [instance["modality"] for instance in tokenized_batch]
            return collated_batch
    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
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