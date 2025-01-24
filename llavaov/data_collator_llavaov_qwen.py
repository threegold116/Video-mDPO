from dataclasses import dataclass, field
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image
import transformers
from llava.mm_utils import tokenizer_image_token,process_images
from llava.utils import process_video_with_decord
from decord import VideoReader
import numpy as np
from decord import cpu

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
    ) -> Dict:
        batch = {}
        #THREEGOLD CHANGE: 将prompt中的<image>替换为空字符串，并添加<image>到prompt的开头 参考llavanext中dpo的实现
        prompt = prompt.replace("<image>", "").strip()
        prompt = "<image>\n" + prompt
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
        prompt_tokens = {}
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        eos_token_id = self.tokenizer.eos_token_id
        # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0) 
        # 获取prompt_tokens["input_ids"]中等于EOS token的索引（通常为0）
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id#将这些索引的attention mask设置为0
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [#
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of prompt
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
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

                batch_element = self.tokenize_batch_element(prompt, chosen, rejected, video_path)#调用tokenize_batch_element函数
                tokenized_batch.append(batch_element)
            
            collated_batch = self.collate(tokenized_batch)
            #THREEGOLD NEED CHANGE：增加
            collated_batch["image_sizes"] = [instance["image_size"] for instance in tokenized_batch]
            collated_batch["modalities"] = [instance["modality"] for instance in tokenized_batch]
            return collated_batch
        
        
        