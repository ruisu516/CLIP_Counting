import torch, os, json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation

class MyDataset(Dataset):
    def __init__(self, data_path, ref, processor, clip_model, device):
    
        self.true_text_embeds, self.cf_text_embeds, self.image_embeds = self.create_dataset_two2five(
            data_path, ref, processor, clip_model, device
        )

        assert self.true_text_embeds.shape[0] == self.cf_text_embeds.shape[0] == self.image_embeds.shape[0]

    def __len__(self):
        return len(self.true_text_embeds)

    def __getitem__(self, idx):
        return self.true_text_embeds[idx], self.cf_text_embeds[idx], self.image_embeds[idx]
    
    def create_dataset_two2five(self,data_path, ref, processor, clip_model, device):
        augmented_data = data_augmentation({ref:torch.load(data_path)})[ref]
        true_text_embeds, cf_text_embeds, image_embeds = [], [], []
        number_words = ["two", "three", "four", "five"]
        for idx, number_word in enumerate(number_words):
            for sample in augmented_data[idx+2]:
                pixel_values = processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device)
                image_embeds += [get_image_embeds(clip_model, pixel_values, device).detach()] * 3
                inputs = processor(text=[f"{number_word} {ref}"], images=None, return_tensors="pt", padding=True)
                true_text_embeds += [clip_model.text_model(**inputs.to(device))[1].detach()] * 3
                number_cf = ["two", "three", "four", "five"]
                number_cf.pop(idx)
                inputs = processor(text=[f"{ele} {ref}" for ele in number_cf], images=None, return_tensors="pt", padding=True)
                cf_text_embeds += [clip_model.text_model(**inputs.to(device))[1].detach()]

        return torch.cat(true_text_embeds, dim=0), torch.cat(cf_text_embeds, dim=0), torch.cat(image_embeds, dim=0)

def get_image_embeds(model, pixel_values, device):
    return model.get_image_features(pixel_values)

class Trainer:
    def __init__(self, dataloader, val_dataloader, text_projection_module, lr, logit_scale, model_save_path):
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.text_projection_module = text_projection_module
        self.lr = lr
        self.logit_scale = logit_scale
        self.optimizer = torch.optim.SGD(self.text_projection_module.parameters(), lr=self.lr)
        self.training_losses = {}
        self.validation_losses = {}
        self.trained_epochs = 0
        self.model_save_path = model_save_path

    def count_loss(self, logits_per_true_text, logits_per_cf_text):
        return -torch.mean(torch.log(torch.exp(logits_per_true_text) / (torch.exp(logits_per_true_text) + torch.exp(logits_per_cf_text))))

    def val(self):
        self.text_projection_module.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients
            total_loss = 0
            total_samples = 0
            for true_text_embeds, cf_text_embeds, image_embeds in self.val_dataloader:
                bs = true_text_embeds.shape[0]
                text_embeds = self.text_projection_module(torch.concat([true_text_embeds.detach(), cf_text_embeds.detach()], dim=0))
                text_embeds = text_embeds / torch.norm(text_embeds, p=2, dim=1, keepdim=True)

                logits_true_text = torch.matmul(text_embeds, image_embeds.detach().t()) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_true_text[:bs], logits_true_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                total_loss += loss.item() * bs
                total_samples += bs

            avg_val_loss = total_loss / total_samples
            print(f'Validation Loss: {avg_val_loss}')

        self.text_projection_module.train()  # Set model back to training mode
        return avg_val_loss
    
    def save_loss_logs(self):
        loss_log = {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        with open(os.path.join(self.model_save_path, f'loss_log_lr{self.lr}.json'), 'w') as f:
            json.dump(loss_log, f, indent=4)

    def train(self, max_num_epochs):
        self.text_projection_module.requires_grad_(True)
        self.text_projection_module.train()
        for epoch in tqdm(range(max_num_epochs), desc='Training Epochs'):
            cumulative_train_loss = 0
            counter = 0
            for true_text_embeds, cf_text_embeds, image_embeds in self.dataloader:
                bs = true_text_embeds.shape[0]
                self.text_projection_module.zero_grad()

                text_embeds = self.text_projection_module(torch.concat([true_text_embeds.detach(), cf_text_embeds.detach()], dim=0))
                text_embeds = text_embeds / torch.norm(text_embeds, p=2, dim=1, keepdim=True)

                logits_true_text = torch.matmul(text_embeds, image_embeds.detach().t()) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_true_text[:bs], logits_true_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                cumulative_train_loss += (loss.detach().item() * bs)
                counter += bs

                loss.backward()
                self.optimizer.step()
            
            self.trained_epochs += 1
            avg_val_loss = self.val()  # Capture validation loss for the current epoch
            self.training_losses[self.trained_epochs] = cumulative_train_loss
            self.validation_losses[self.trained_epochs] = avg_val_loss

            # Saving the model after each epoch
            torch.save(self.text_projection_module.state_dict(), os.path.join(self.model_save_path, f'model_epoch_{self.trained_epochs}.pth'))
            print(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter}, Val Loss: {avg_val_loss}')
        
        self.save_loss_logs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on customized dataset")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--ref_obj",type=str,default="dogs",help="name of the object being used as an reference")   
    parser.add_argument("--train_data_path",type=str,help="path to custom data")
    parser.add_argument("--eval_data_path",type=str,help="path to custom data")
    parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    parser.add_argument("--model_save_path",type=str,help="path to custom data")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(args.model)
    clip_model = CLIPModel.from_pretrained(args.model).to(device)

    dataset = MyDataset(args.train_data_path, args.ref_obj, processor, clip_model, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = MyDataset(args.eval_data_path, args.ref_obj, processor, clip_model, device)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("len(dataset),len(val_dataset)",len(dataset),len(val_dataset))
    logit_scale = clip_model.logit_scale.exp()

    my_clip_text_projection = torch.nn.Linear(clip_model.text_projection.in_features,clip_model.text_projection.out_features,bias=False).to(device)
    my_clip_text_projection.load_state_dict(clip_model.text_projection.state_dict())

    if not os.path.isdir(args.model_save_path):
        os.mkdir(args.model_save_path)
    trainer = Trainer(
        dataloader, 
        val_dataloader, 
        my_clip_text_projection, 
        args.lr, 
        logit_scale, 
        args.model_save_path
    )
    trainer.train()

