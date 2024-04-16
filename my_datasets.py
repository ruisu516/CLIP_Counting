import torch, os, json, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation
from clip_count_utils import *

class TextDataset(Dataset):
    def __init__(self, data_path, ref, processor, clip_model, device, add_object_cf=False,ref_obj_file=None):
    
        self.true_texts, self.cf_texts, self.image_embeds = self.create_dataset_two2five(
            data_path, ref, processor, clip_model, device
        )

        assert len(self.true_texts) == len(self.cf_texts) == self.image_embeds.shape[0]
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.true_texts)

    def __getitem__(self, idx):
        inputs = self.processor(text=[self.true_texts[idx]], images=None, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        cf_inputs = self.processor(text=[self.cf_texts[idx]], images=None, return_tensors="pt", padding=True)
        cf_inputs = {k:v.to(self.device) for k,v in cf_inputs.items()}
        return inputs, cf_inputs, self.image_embeds[idx]
    
    def create_dataset_two2five(self,data_path, ref, processor, clip_model, device):
        with torch.no_grad():
            augmented_data = data_augmentation({ref:torch.load(data_path)})[ref]

            true_texts, cf_texts, image_embeds = [], [], []
            number_words = ["two", "three", "four", "five"]
            for idx, number_word in enumerate(number_words):
                for sample in augmented_data[idx+2]:
                    pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device) # torch.Size([1, 3, 224, 224])
                    image_embeds+=[get_image_embeds(clip_model,pixel_values.to(device),device=device).detach()]*3
                    true_texts += [f"{number_word} {ref}"]*3 # torch.Size([1, 512])
                    number_cf = number_words.copy()
                    number_cf.pop(idx)
                    cf_texts += [f"{ele} {ref}" for ele in number_cf]# torch.Size([3, 512])

            return true_texts, cf_texts, torch.cat(image_embeds, dim=0).detach().clone()


class TextEmbeddingDataset(Dataset):
    def __init__(self, data_path, ref, processor, clip_model, device, add_object_cf=False,ref_obj_file=None):
    
        self.true_text_embeds, self.cf_text_embeds, self.image_embeds = self.create_dataset_two2five(
            data_path, ref, processor, clip_model, device, add_object_cf,ref_obj_file
        )
        self.add_object_cf = add_object_cf

        assert self.true_text_embeds.shape[0] == self.cf_text_embeds.shape[0] == self.image_embeds.shape[0]

    def __len__(self):
        return len(self.true_text_embeds)

    def __getitem__(self, idx):
        return self.true_text_embeds[idx], self.cf_text_embeds[idx], self.image_embeds[idx]
    
    def create_dataset_two2five(self,data_path, ref, processor, clip_model, device, add_object_cf,ref_obj_file):
        if add_object_cf:
            with open(ref_obj_file, 'r') as file:
                other_ref_objs = file.readlines()
            other_ref_objs = [line.strip() for line in other_ref_objs]
            if ref in other_ref_objs:
                other_ref_objs.remove(ref)
            random.seed(42)
            true_number = 6
        else:
            true_number = 3
        
        with torch.no_grad():
            # augmented_data = data_augmentation({ref:torch.load(data_path)})[ref]
            # print("Using augmented data")
            
            augmented_data = torch.load(data_path)
            print("Not using augmented data")
            true_text_embeds, cf_text_embeds, image_embeds = [], [], []
            number_words = ["two", "three", "four", "five"]
            for idx, number_word in enumerate(number_words):
                for sample in augmented_data[idx+2]:
                    pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device) # torch.Size([1, 3, 224, 224])
                    image_embeds+=[get_image_embeds(clip_model,pixel_values.to(device),device=device).detach()]*true_number
                    inputs = processor(text=[f"{number_word} {ref}"], images=None, return_tensors="pt", padding=True)
                    true_text_embeds +=[clip_model.text_model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        position_ids=None,
                        output_attentions=clip_model.config.output_attentions,
                        output_hidden_states=clip_model.config.output_hidden_states,
                        return_dict=clip_model.config.use_return_dict,
                    )[1].detach()]*true_number # torch.Size([1, 512])
                    number_cf = number_words.copy()
                    number_cf.pop(idx)
                    inputs = processor(text=[f"{ele} {ref}" for ele in number_cf], images=None, return_tensors="pt", padding=True)
                    
                    cf_text_embeds.append(
                        clip_model.text_model(
                            input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            position_ids=None,
                            output_attentions=clip_model.config.output_attentions,
                            output_hidden_states=clip_model.config.output_hidden_states,
                            return_dict=clip_model.config.use_return_dict,
                        )[1].detach()
                    )                                       
                    if add_object_cf:
                        inputs = processor(text=[f"{number_word} {obj}" for obj in random.sample(other_ref_objs, 3)], images=None, return_tensors="pt", padding=True)
                        cf_text_embeds.append(
                            clip_model.text_model(
                                input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device),
                                position_ids=None,
                                output_attentions=clip_model.config.output_attentions,
                                output_hidden_states=clip_model.config.output_hidden_states,
                                return_dict=clip_model.config.use_return_dict,
                            )[1].detach()
                        )                            

            return torch.cat(true_text_embeds, dim=0).detach().clone(), torch.cat(cf_text_embeds, dim=0).detach().clone(), torch.cat(image_embeds, dim=0).detach().clone()

class ProcessedCountBenchDataset(Dataset):
    def __init__(
            self, 
            data,
            device="cuda",
            num_classes=4
    ):
        self.data = data
        self.device = device
        self.num_classes=num_classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample["image_embeds"],sample['target_obj_text'], sample["target_obj_aug_text"][:self.num_classes], sample["target_obj_with_context_text"], sample["target_obj_aug_with_context_text"][:self.num_classes], sample["gt_count"]
