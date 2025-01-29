from llava_hound_model.model import LlavaLlamaForCausalLM
from typing import List, Optional, Tuple, Union, Dict
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchvision.transforms import v2
import random
class mDPOLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    # THREEGOLD CHANGE:增加对图像操作的设置函数
    crop_mode = "crop_images_only"
    noisy_frames_radio = 0.2
    def set_crop_mode(self, crop_mode,noisy_frames_radio=0.2):
        self.crop_mode = crop_mode
        self.noisy_frames_radio = noisy_frames_radio
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: Optional[bool] = None,
        mask_visual_tokens: Optional[bool] = False,#THREEGOLD NEED CHANGE
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if mask_visual_tokens:
            images = self.crop_images(images)
        if inputs_embeds is None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        if dpo_forward:
            outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ), labels
            #THREEGOLD NEED CHANGE
    def process_video(self, video_path,num_frames=8):
        video_tower = self.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_processor = video_tower.video_processor
        video = video_processor(video_path, return_tensors="pt",video_decode_backend="decord",num_frames=num_frames)["pixel_values"]
        # video = video_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        # video = [video]#TODO:这里需要根据视频修改
        return video[0]
    def crop_images(self, images):
        images_tensor = images[0]
        if self.crop_mode=="crop_images_only":
            new_images_tensor = []
            for image in images_tensor:
                resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
                if image.shape[0] == 1:
                    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
                else:
                    image = resize_cropper(image)
                new_images_tensor.append(image)
            return [new_images_tensor,images[1]]
        elif self.crop_mode == "replace_frames":
            new_images_tensor = []
            for image in images_tensor:
                replace_frames_num = int(image.shape[0] * self.noisy_frames_radio)  
                replace_frames_idx = random.sample(range(image.shape[0]), replace_frames_num)    
                for idx in replace_frames_idx:
                    # 替换为全白图像
                    image[idx] = torch.zeros_like(image[idx])
                new_images_tensor.append(image)
            return [new_images_tensor,images[1]]
        elif self.crop_mode == "shuffle_frames":
            new_images_tensor = []
            for image in images_tensor:
                new_image = []
                random_frames_idx = list(range(image.shape[0]))
                random.shuffle(random_frames_idx)
                for idx in random_frames_idx:
                    new_image.append(image[idx])
                new_image = torch.stack(new_image,dim=0)
                new_images_tensor.append(new_image)
            return [new_images_tensor,images[1]]
        elif self.crop_mode == "replace_frames_and_crop_images":
            new_images_tensor = []
            for image in images_tensor:
                resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
                replace_frames_num = int(image.shape[0] * self.noisy_frames_radio)  
                replace_frames_idx = random.sample(range(image.shape[0]), replace_frames_num)    
                for idx in replace_frames_idx:
                    # 替换为全白图像
                    image[idx] = torch.zeros_like(image[idx])
                crop_images_idx = range(image.shape[0])-replace_frames_idx
                for idx in crop_images_idx:
                    image[idx] = resize_cropper(image[idx].squeeze(0)).unsqueeze(0)                
                new_images_tensor.append(image)
            return [new_images_tensor,images[1]]
        else:
            return images