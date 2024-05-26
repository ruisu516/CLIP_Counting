import torch, os, json, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler, Subset
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation
from my_datasets import TextDatasetCustomDogs,TextDatasetOnlineTrain

import torch
import torch.nn as nn
from transformers import CLIPTextModel
import wandb
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger


def count_loss(logits_per_true_text, logits_per_cf_text, device="cuda", add_object_cf=False):
        # Extract the diagonal elements
    true_diag = torch.diag(logits_per_true_text)
    cf_diag = torch.diag(logits_per_cf_text)
    count_loss = torch.mean(torch.log(1 + torch.exp(cf_diag - true_diag)))

    if add_object_cf:
        eye = torch.eye(logits_per_true_text.size(0)).bool().to(device)
        negative_mask = ~eye  # Invert the identity matrix to get the negative mask
        row_negative_logits = logits_per_true_text.masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
        row_positive_logits = true_diag.unsqueeze(1) #(bz,1)
        row_obj_loss = torch.mean(torch.log(1 + torch.exp(row_negative_logits - row_positive_logits)))
        
        col_negative_logits = logits_per_true_text.t().masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
        col_positive_logits = true_diag.unsqueeze(1) #(bz,1)
        col_obj_loss = torch.mean(torch.log(1 + torch.exp(col_negative_logits - col_positive_logits)))
        
        return count_loss + 0.5 * row_obj_loss + 0.5 * col_obj_loss
    else:
        return count_loss

class CustomCLIPModel(L.LightningModule):
    def __init__(self, pretrained_checkpoint,args,device="cuda"):
        super(CustomCLIPModel, self).__init__()
        # TODO: load local pre-trained model
        clip_model = CLIPModel.from_pretrained(pretrained_checkpoint).to(device)
        self.clip_text_model = clip_model.text_model
        self.clip_text_projection = clip_model.text_projection
        if args.train_number_shift_vectors:
            self.number_shift_vectors = number_shift_vectors
        else:
            self.number_shift_vectors = None
        self.clip_config = clip_model.configs
        self.logit_scale = None # TODO:
        self.device = device
        self.args = args
    
    def forward(
            self, 
            gt_input_ids, 
            true_attention_mask, 
            gt_label_idx, #(bz,num_classes-1)
            cf_input_ids, 
            cf_attention_mask, 
            cf_label_idx, #(bz,num_classes-1)
            target_input_ids=None, 
            target_attention_mask=None,
    ):

        assert cf_label_idx.shape == gt_label_idx.shape
        true_text_embeds = self.clip_text_projection(
            self.clip_text_model(
                input_ids=gt_input_ids,
                attention_mask=true_attention_mask,
                position_ids=None,
                output_attentions=self.clip_config.output_attentions,
                output_hidden_states=self.clip_config.output_hidden_states,
                return_dict=self.clip_config.use_return_dict,
            )[1]
        )#(bz,dim)
        cf_text_embeds = self.clip_text_projection(
            self.clip_text_model(
                input_ids=cf_input_ids,
                attention_mask=cf_attention_mask,
                position_ids=None,
                output_attentions=self.clip_config.output_attentions,
                output_hidden_states=self.clip_config.output_hidden_states,
                return_dict=self.clip_config.use_return_dict,
            )[1]
        ) #(bz,dim)

        if self.number_shift_vectors is not None:
            
            if self.args.orthogonalize and target_input_ids is not None:
                target_embeds = self.clip_text_projection(
                    self.clip_text_model(
                        input_ids=target_input_ids,
                        attention_mask=target_attention_mask,
                        position_ids=None,
                        output_attentions=self.clip_config.output_attentions,
                        output_hidden_states=self.clip_config.output_hidden_states,
                        return_dict=self.clip_config.use_return_dict,
                    )[1]
                ) # (bz, dim)
                #projection (num_classes,bz, dim)
                projection = torch.matmul(target_embeds,self.number_shift_vectors.permute(1,0)).unsqueeze(-1) / torch.sum(target_embeds*target_embeds,dim=1).unsqueeze(-1).unsqueeze(-1) * target_embeds.unsqueeze(1).repeat(1,self.number_shift_vectors.shape[0],1) 
                number_shift_vectors = self.number_shift_vectors.unsqueeze(0) - projection #(bz,num_classes,dim)

            else:
                number_shift_vectors = self.number_shift_vectors.unsqueeze(0).repeat(true_text_embeds.shape[0],1,1) #(bz,num_classes,dim)
            
            
            cf_label_idx,gt_label_idx = cf_label_idx.unsqueeze(-1),gt_label_idx.unsqueeze(-1)
            cf_batch_indices = torch.arange(true_text_embeds.shape[0]).unsqueeze(1).expand_as(cf_label_idx)
            true_text_embeds = true_text_embeds + number_shift_vectors[cf_batch_indices,gt_label_idx].squeeze()
            cf_text_embeds = cf_text_embeds + number_shift_vectors[cf_batch_indices,cf_label_idx].squeeze()
     
        text_embeds = torch.concat([true_text_embeds, cf_text_embeds], dim=0) # (2*bz,dim)
        return text_embeds / torch.norm(text_embeds, p=2, dim=-1, keepdim=True)

    def forward_step(self,batch,add_object_cf):
        gt_inputs, cf_inputs, image_embeds, target_obj_inputs, gt_label_idx, cf_label_idx = batch
        bs = image_embeds.shape[0]

        text_embeds = self.forward(
            gt_input_ids=gt_inputs["input_ids"].squeeze().to(device), 
            true_attention_mask=gt_inputs["attention_mask"].squeeze().to(device), 
            gt_label_idx=gt_label_idx.to(device),
            cf_input_ids=cf_inputs["input_ids"].squeeze().to(device), 
            cf_attention_mask=cf_inputs["attention_mask"].squeeze().to(device), 
            cf_label_idx=cf_label_idx.to(device),
            target_input_ids=target_obj_inputs["input_ids"].squeeze().to(device), 
            target_attention_mask=target_obj_inputs["attention_mask"].squeeze().to(device),
        )

        logits_per_text = torch.matmul(text_embeds, image_embeds.detach().t().to(device)) * self.logit_scale
        logits_per_true_text, logits_per_cf_text = logits_per_text[:bs], logits_per_text[bs:]

        loss = count_loss(logits_per_true_text, logits_per_cf_text,device=self.device, add_object_cf=add_object_cf)
        
        return loss
    
    def training_step(self,batch,add_object_cf):
        loss = self.forward_step(batch,add_object_cf)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self,batch,add_object_cf):
        loss = self.forward_step(batch,add_object_cf)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        trainable_parameters = []
        if "text_model" in self.args.trainable_parameters:
            for name,param in self.clip_text_model.named_parameters():
                param.requires_grad = True
                trainable_parameters.append(param)
        else:
            param.requires_grad = False
        
        if "text_projection" in self.args.trainable_parameters:
            for name,param in self.clip_text_projection.named_parameters():
                param.requires_grad = True
                trainable_parameters.append(param)
        else:
            param.requires_grad = False
        
        if self.train_number_shift_vectors:
            if args.number_shift_vectors_init_weight_path is not None:
                print("loading number_shift_vectors from",args.number_shift_vectors_init_weight_path)
                loaded_tensor = torch.load(args.number_shift_vectors_init_weight_path).to(device)
                loaded_tensor.requires_grad_(True)
                number_shift_vectors = torch.nn.Parameter(loaded_tensor)
            else:
                torch.manual_seed(42)
                number_shift_vectors = torch.nn.Parameter(torch.randn(args.num_classes, self.clip_config.projection_dim).to(device))

            trainable_parameters.append(number_shift_vectors)
        else:
            number_shift_vectors = None
        print("len(trainable_parameters)",len(trainable_parameters))

        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.args.lr)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.args.lr)
        return optimizer


def save_metadata(args, filename='metadata.json'):
    """Save the parsed arguments to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(args, f, indent=4)

def get_image_embeds(model,pixel_values,device="cpu"):
    vision_outputs = model.vision_model(
            pixel_values=pixel_values.to(device),
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
            return_dict=model.config.use_return_dict,
        )
    image_embeds = vision_outputs[1]
    image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds

# class Trainer:
#     def __init__(self, dataloader, val_dataloader, clip_model, logit_scale, model_save_path, args,device="cuda"):
#         self.dataloader = dataloader
#         self.val_dataloader = val_dataloader
#         self.args = args
#         # self.logit_scale = logit_scale
#         self.train_number_shift_vectors = args.train_number_shift_vectors
#         self.clip_config = clip_model.config
#         self.device=device
               
#         
        
#         self.training_losses = {}
#         self.validation_losses = {}
#         self.trained_epochs = 0
#         self.model_save_path = model_save_path



#     def val(self,eval_data_mode="val"):
#         self.text_model.eval()

#         if eval_data_mode=="val":
#             loader = self.val_dataloader
#         elif eval_data_mode=="train":
#             loader = self.dataloader

#         with torch.no_grad():  # No need to track gradients
#             total_loss = 0
#             total_samples = 0
#             # for gt_inputs, cf_inputs, image_embeds, target_obj_inputs, gt_label_idx, cf_label_idx in loader:
#             #     bs = image_embeds.shape[0]

#             #     text_embeds = self.text_model(
#             #         gt_input_ids=gt_inputs["input_ids"].squeeze().to(device), 
#             #         true_attention_mask=gt_inputs["attention_mask"].squeeze().to(device), 
#             #         gt_label_idx=gt_label_idx.to(device),
#             #         cf_input_ids=cf_inputs["input_ids"].squeeze().to(device), 
#             #         cf_attention_mask=cf_inputs["attention_mask"].squeeze().to(device), 
#             #         cf_label_idx=cf_label_idx.to(device),
#             #         target_input_ids=target_obj_inputs["input_ids"].squeeze().to(device), 
#             #         target_attention_mask=target_obj_inputs["attention_mask"].squeeze().to(device),
#             #     )

#             #     logits_per_text = torch.matmul(text_embeds, image_embeds.detach().t().to(device)) * self.logit_scale
#             #     logits_per_true_text, logits_per_cf_text = logits_per_text[:bs], logits_per_text[bs:]

#             #     loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
#             #     total_loss += loss.item() * bs
#             #     total_samples += bs

#             avg_val_loss = total_loss / total_samples
#             # print(f'Validation Loss: {avg_val_loss}')

#         self.text_model.train()  # Set model back to training mode
#         return avg_val_loss
    
#     def save_loss_logs(self):
#         loss_log = {
#             'training_losses': self.training_losses,
#             'validation_losses': self.validation_losses
#         }
#         with open(os.path.join(self.model_save_path, f'loss_log_lr{self.args.lr}.json'), 'w') as f:
#             json.dump(loss_log, f, indent=4)

#     def train(self, max_num_epochs):
#         self.text_model.train()

#         avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
#         pbar = tqdm(total=max_num_epochs, desc=f'Epoch 0/{max_num_epochs}, Training Loss: N/A, Val Loss: {avg_val_loss:.4f}')
        
#         if not self.args.not_log_wandb:
#             wandb.log({
#                 f"train_loss":None,
#                 f"val_loss":avg_val_loss,
#                 f"trained_epochs":0, 
#             })
            
#         best_val_loss,best_ep = avg_val_loss,0
#         early_stopped = False
#         for epoch in range(max_num_epochs):
#             cumulative_train_loss = 0
#             counter = 0
#             # for gt_inputs, cf_inputs, image_embeds, target_obj_inputs, gt_label_idx, cf_label_idx in self.dataloader:
#                 # bs = image_embeds.shape[0]
#                 # self.text_model.zero_grad()

#                 # text_embeds = self.text_model(
#                 #     gt_input_ids=gt_inputs["input_ids"].squeeze().to(device), 
#                 #     true_attention_mask=gt_inputs["attention_mask"].squeeze().to(device), 
#                 #     gt_label_idx=gt_label_idx.to(device),
#                 #     cf_input_ids=cf_inputs["input_ids"].squeeze().to(device), 
#                 #     cf_attention_mask=cf_inputs["attention_mask"].squeeze().to(device), 
#                 #     cf_label_idx=cf_label_idx.to(device),
#                 #     target_input_ids=target_obj_inputs["input_ids"].squeeze().to(device), 
#                 #     target_attention_mask=target_obj_inputs["attention_mask"].squeeze().to(device),
#                 # )

#                 # logits_per_text = torch.matmul(text_embeds, image_embeds.detach().t().to(device)) * self.logit_scale
#                 # logits_per_true_text, logits_per_cf_text = logits_per_text[:bs], logits_per_text[bs:]

#                 # loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
#                 # cumulative_train_loss += (loss.detach().item() * bs)
#                 # counter += bs

#                 # loss.backward()
#                 # self.optimizer.step()
            
#             self.trained_epochs += 1
#             avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
#             self.training_losses[self.trained_epochs] = cumulative_train_loss / counter
#             self.validation_losses[self.trained_epochs] = avg_val_loss

#             if avg_val_loss < best_val_loss:
#                 model_path = os.path.join(self.model_save_path, f'best_model.pth')
#                 best_val_loss = avg_val_loss
#                 best_ep = self.trained_epochs
#                 torch.save(self.text_model.state_dict(), model_path)

#             if not self.args.not_log_wandb:
#                 wandb.log({
#                     f"train_loss":cumulative_train_loss / counter,
#                     f"val_loss":avg_val_loss,
#                     f"trained_epochs":self.trained_epochs,
#                 })

#             # Saving the model after each epoch
#             pbar.set_description(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter}, Val Loss: {avg_val_loss}')
#             pbar.update(1)

#             if self.trained_epochs - best_ep > 10 and best_ep > 0:
#                 early_stopped = True
#                 print("Early stopping")
#                 break

#         self.save_loss_logs()
#         if not self.args.not_log_wandb:
#             wandb.config.best_ep = best_ep
#             wandb.config.best_val_loss = best_val_loss
#             wandb.config.early_stopping = early_stopped
#             wandb.finish()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on customized dataset")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--ref_obj",type=str,default=None,help="name of the object being used as an reference")   
    parser.add_argument("--optimizer",type=str,default="SGD",choices=["SGD","Adam"])   
    parser.add_argument("--train_data_path",type=str,help="path to custom data")
    parser.add_argument("--eval_data_path",type=str,help="path to custom data")
    parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    parser.add_argument('--trainable_parameters', nargs='+', help='List of trainable parameters',default=[],choices=["text_model","text_projection"])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=1)
    parser.add_argument("--random_seed", type=int, default=None)



    parser.add_argument("--number_shift_vectors_init_weight_path",type=str,default=None)
    parser.add_argument("--save_root_folder",type=str)
    parser.add_argument('--add_object_cf', action='store_true')
    parser.add_argument('--CLIP_loss', type=str,default="",choices=["both",""])
    parser.add_argument('--train_number_shift_vectors', action='store_true')
    parser.add_argument('--orthogonalize', action='store_true',help="orthogonalize the number shift vectors w.r.t. the target object") 
    parser.add_argument("--ref_obj_file",type=str,default=None,help="path to the ref objects")   
    parser.add_argument('--not_log_wandb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    

    args = parser.parse_args()
    args.num_epochs = int(args.num_epochs/args.train_ratio)
    print("args.num_epochs",args.num_epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(args.model)
    clip_model = CLIPModel.from_pretrained(args.model).to(device)

    for name,param in clip_model.named_parameters():
        param.requires_grad = False
    if args.ref_obj == "dogs":
        dataset = TextDatasetCustomDogs(args.train_data_path, args.ref_obj, processor, clip_model, device, add_object_cf=args.add_object_cf,ref_obj_file=args.ref_obj_file)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    elif args.ref_obj is None:
        dataset = TextDatasetOnlineTrain(args.train_data_path,processor,num_classes=args.num_classes,ratio=args.train_ratio)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TextDatasetOnlineTrain(args.eval_data_path, processor,num_classes=args.num_classes)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    print("len(dataset),len(val_dataset)",len(dataset),len(val_dataset))
    logit_scale = clip_model.logit_scale.exp()
    
    print("trainable_parameters",args.trainable_parameters)
    trained_param_save_name = "_".join(args.trainable_parameters)
    model_save_path = f"{args.save_root_folder}/{args.model.split('/')[1]}_{trained_param_save_name}{args.train_number_shift_vectors}{args.orthogonalize}_{args.lr}_{args.optimizer}{args.ref_obj}_{args.add_object_cf}{args.CLIP_loss}"
    os.makedirs(model_save_path, exist_ok=True)


    # if not args.not_log_wandb:
    #     wandb.init(project="train_clip_count", config=args.__dict__, entity="ruisu")

    from lightning.pytorch import Trainer, seed_everything

    seed_everything(42, workers=True)
    wandb_logger = WandbLogger(project="train_clip_count", entity="ruisu")

    # sets seeds for numpy, torch and python.random.
    model = CustomCLIPModel(
        pretrained_checkpoint=args.model,
        args=args,
        device=device
    )
    callbacks = None
    trainer = Trainer(
        deterministic=True,
        accelerator=device,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        logger=wandb_logger,
        max_epochs=500,
        min_epochs=1,
        profiler="simple", # to profile standard training events, equivalent to `profiler=SimpleProfiler()`

    )
    
    trainer.fit(model, dataloader, val_dataloader)

    # trainer.train(args.num_epochs)
    # args_dict = vars(args)
    
    # # Save the dictionary as a JSON file
    # save_metadata(args_dict,filename=os.path.join(model_save_path, 'metadata.json'))

