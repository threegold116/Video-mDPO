from transformers import AutoConfig, AutoModelForCausalLM
from abc import ABC, abstractmethod
from llava.model import LlavaQwenForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, expand2square
'''
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
'''



class mDPOLlavaQwenForCausalLM(LlavaQwenForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        mask_visual_tokens: Optional[bool] = False,#THREEGOLD NEED CHANGE
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #THREEGOLD NEED CHANGE
        if mask_visual_tokens:
            images = self.crop_images(images)
        
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ), labels
            #THREEGOLD NEED CHANGE
    def crop_images(self, images):
        new_images = []
        for image in images:
            resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
            if image.shape[0] == 1:
                image = resize_cropper(image.squeeze(0)).unsqueeze(0)
            else:
                image = resize_cropper(image)
            new_images.append(image)
        return new_images
    def process_video(self, video):
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        # video = [video]#TODO:这里需要根据视频修改
        return video    
    def process_images(self, images, model_cfg):
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "highres":
            for image in images:
                image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
                new_images.append(image)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            for image in images:
                image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
                new_images.append(image)
        elif image_aspect_ratio == "crop_split":
            for image in images:
                image = process_highres_image_crop_split(image, model_cfg, image_processor)
                new_images.append(image)
        elif image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images