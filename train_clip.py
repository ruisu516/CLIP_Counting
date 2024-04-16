import torch, os, json, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation
from my_datasets import TextEmbeddingDataset, TextDataset

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

class Trainer:
    def __init__(self, dataloader, val_dataloader, clip_model, logit_scale, model_save_path, args):
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.logit_scale = logit_scale
        self.clip_model = clip_model
       
        trainable_parameters = []
        for name,param in clip_model.named_parameters():
            if any([ele in name for ele in self.args.trainable_parameters]):
                param.requires_grad = True
                trainable_parameters.append(param)
            else:
                param.requires_grad = False
        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(trainable_parameters, lr=self.args.lr)
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.args.lr)
        
        self.training_losses = {}
        self.validation_losses = {}
        self.trained_epochs = 0
        self.model_save_path = model_save_path

    # def count_loss(self, logits_per_true_text, logits_per_cf_text):
    #     # return -torch.mean(torch.log(torch.exp(logits_per_true_text) / (torch.exp(logits_per_true_text) + torch.exp(logits_per_cf_text))))
    #     return torch.mean(torch.log(1+torch.exp(logits_per_cf_text-logits_per_true_text)))
        
    def count_loss(self, logits_per_true_text, logits_per_cf_text):
        # Extract the diagonal elements
        true_diag = torch.diag(logits_per_true_text)
        cf_diag = torch.diag(logits_per_cf_text)
        
        return torch.mean(torch.log(1 + torch.exp(cf_diag - true_diag)))

    def val(self,eval_data_mode="val"):
        self.clip_model.eval()

        if eval_data_mode=="val":
            loader = self.val_dataloader
        elif eval_data_mode=="train":
            loader = self.dataloader

        with torch.no_grad():  # No need to track gradients
            total_loss = 0
            total_samples = 0
            for true_texts, cf_texts, image_embeds in loader:
                bs = image_embeds.shape[0]
                true_text_embeds = clip_model.text_model(
                    input_ids=true_texts["input_ids"].squeeze().to(device),
                    attention_mask=true_texts["attention_mask"].squeeze().to(device),
                    position_ids=None,
                    output_attentions=clip_model.config.output_attentions,
                    output_hidden_states=clip_model.config.output_hidden_states,
                    return_dict=clip_model.config.use_return_dict,
                )[1]
                cf_text_embeds = clip_model.text_model(
                    input_ids=cf_texts["input_ids"].squeeze().to(device),
                    attention_mask=cf_texts["attention_mask"].squeeze().to(device),
                    position_ids=None,
                    output_attentions=clip_model.config.output_attentions,
                    output_hidden_states=clip_model.config.output_hidden_states,
                    return_dict=clip_model.config.use_return_dict,
                )[1]

                text_embeds = self.clip_model.text_projection(torch.concat([true_text_embeds, cf_text_embeds], dim=0))
                text_embeds = text_embeds / torch.norm(text_embeds, p=2, dim=1, keepdim=True)

                logits_true_text = torch.matmul(text_embeds, image_embeds.detach().t()) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_true_text[:bs], logits_true_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                total_loss += loss.item() * bs
                total_samples += bs

            avg_val_loss = total_loss / total_samples
            # print(f'Validation Loss: {avg_val_loss}')

        self.clip_model.train()  # Set model back to training mode
        return avg_val_loss
    
    def save_loss_logs(self):
        loss_log = {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        with open(os.path.join(self.model_save_path, f'loss_log_lr{self.args.lr}.json'), 'w') as f:
            json.dump(loss_log, f, indent=4)

    def train(self, max_num_epochs):
        self.clip_model.train()

        avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
        pbar = tqdm(total=max_num_epochs, desc=f'Epoch 0/{max_num_epochs}, Training Loss: N/A, Val Loss: {avg_val_loss:.4f}')
        
        for epoch in range(max_num_epochs):
            cumulative_train_loss = 0
            counter = 0
            for true_texts, cf_texts, image_embeds in self.dataloader:
                bs = image_embeds.shape[0]
                self.clip_model.zero_grad()

                true_text_embeds = clip_model.text_model(
                    input_ids=true_texts["input_ids"].squeeze().to(device),
                    attention_mask=true_texts["attention_mask"].squeeze().to(device),
                    position_ids=None,
                    output_attentions=clip_model.config.output_attentions,
                    output_hidden_states=clip_model.config.output_hidden_states,
                    return_dict=clip_model.config.use_return_dict,
                )[1]
                cf_text_embeds = clip_model.text_model(
                    input_ids=cf_texts["input_ids"].squeeze().to(device),
                    attention_mask=cf_texts["attention_mask"].squeeze().to(device),
                    position_ids=None,
                    output_attentions=clip_model.config.output_attentions,
                    output_hidden_states=clip_model.config.output_hidden_states,
                    return_dict=clip_model.config.use_return_dict,
                )[1]

                text_embeds = self.clip_model.text_projection(torch.concat([true_text_embeds, cf_text_embeds], dim=0))
                text_embeds = text_embeds / torch.norm(text_embeds, p=2, dim=1, keepdim=True)

                logits_per_text = torch.matmul(text_embeds, image_embeds.detach().t()) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_per_text[:bs], logits_per_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                cumulative_train_loss += (loss.detach().item() * bs)
                counter += bs

                loss.backward()
                self.optimizer.step()
            
            self.trained_epochs += 1
            avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
            self.training_losses[self.trained_epochs] = cumulative_train_loss / counter
            self.validation_losses[self.trained_epochs] = avg_val_loss

            # Saving the model after each epoch
            torch.save(self.clip_model.state_dict(), os.path.join(self.model_save_path, f'model_epoch_{self.trained_epochs}.pth'))
            print(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter}, Val Loss: {avg_val_loss}')
            # pbar.set_description(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter:.4f}, Val Loss: {avg_val_loss:.4f}')
            pbar.update(1)

        self.save_loss_logs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on customized dataset")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--ref_obj",type=str,default="dogs",help="name of the object being used as an reference")   
    parser.add_argument("--optimizer",type=str,default="SGD",choices=["SGD","Adam"])   
    parser.add_argument("--train_data_path",type=str,help="path to custom data")
    parser.add_argument("--eval_data_path",type=str,help="path to custom data")
    parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    parser.add_argument('--trainable_parameters', nargs='+', help='List of trainable parameters',default=["text_model","text_projection"])

    parser.add_argument('--add_object_cf', action='store_true')
    parser.add_argument("--ref_obj_file",type=str,default=None,help="path to the ref objects")   

    

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(args.model)
    clip_model = CLIPModel.from_pretrained(args.model).to(device)

    for name,param in clip_model.named_parameters():
        param.requires_grad = False
    dataset = TextDataset(args.train_data_path, args.ref_obj, processor, clip_model, device, add_object_cf=args.add_object_cf,ref_obj_file=args.ref_obj_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TextDataset(args.eval_data_path, args.ref_obj, processor, clip_model, device, add_object_cf=args.add_object_cf,ref_obj_file=args.ref_obj_file)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("len(dataset),len(val_dataset)",len(dataset),len(val_dataset))
    logit_scale = clip_model.logit_scale.exp()
    
    print("trainable_parameters",args.trainable_parameters)
    trained_param_save_name = "_".join(args.trainable_parameters)
    model_save_path = f"trained_models/{args.model.split('/')[1]}_{trained_param_save_name}_{args.lr}_{args.optimizer}{args.ref_obj}_{args.add_object_cf}"
    print("model_save_path",model_save_path)
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    trainer = Trainer(
        dataloader, 
        val_dataloader, 
        clip_model, 
        logit_scale, 
        model_save_path,
        args,
    )

    trainer.train(args.num_epochs)


    args_dict = vars(args)
    
    # Save the dictionary as a JSON file
    save_metadata(args_dict,filename=os.path.join(model_save_path, 'metadata.json'))

