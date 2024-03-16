import torch

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, true_text_embeds,cf_text_embeds,image_embeds):
        assert true_text_embeds.shape[0] == cf_text_embeds.shape[0]
        assert image_embeds.shape[0] == cf_text_embeds.shape[0]
        self.true_text_embeds=true_text_embeds
        self.cf_text_embeds=cf_text_embeds
        self.image_embeds=image_embeds

    def __len__(self):
        return len(self.true_text_embeds)

    def __getitem__(self, idx):
        return self.true_text_embeds[idx], self.cf_text_embeds[idx], self.image_embeds[idx]




from transformers import CLIPProcessor, CLIPModel
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name="openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
# torch.save(clip_model.text_projection.state_dict(),"clip_text_projection.pth")
for name,param in clip_model.named_parameters():
        if "text_projection" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

NUMBER_WORDS_SUB = [
        "two", "three", "four", "five",
    ]
number_range=[2,3,4,5]
# for ref in
ref='dogs'
MY_DOGS_DATA = torch.load('my_data/my_new_data.pth')[ref]
true_text_embeds,cf_text_embeds,image_embeds=[],[],[]
for number in number_range:
    predictions=[]
    for sample in MY_DOGS_DATA[number]:
        pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device) # torch.Size([1, 3, 224, 224])
        image_embeds+=[clip_count_utils.get_image_embeds(clip_model,pixel_values.to(device),device=device).detach()]*(len(number_range)-1)
        inputs = processor(text=[f"{NUMBER_WORDS_SUB[number-2]} {ref}"], images=None, return_tensors="pt", padding=True)
        true_text_embeds+=[clip_model.text_model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                position_ids=None,
                output_attentions=clip_model.config.output_attentions,
                output_hidden_states=clip_model.config.output_hidden_states,
                return_dict=clip_model.config.use_return_dict,
            )[1].detach()]*(len(number_range)-1) # torch.Size([1, 512])
        number_cf = NUMBER_WORDS_SUB.copy()
        number_cf.pop(number-2)
        inputs = processor(text=[f"{ele} {ref}" for ele in number_cf], images=None, return_tensors="pt", padding=True)
        cf_text_embeds.append(clip_model.text_model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                position_ids=None,
                output_attentions=clip_model.config.output_attentions,
                output_hidden_states=clip_model.config.output_hidden_states,
                return_dict=clip_model.config.use_return_dict,
            )[1].detach()) # torch.Size([3, 512])
true_text_embeds=torch.concat(true_text_embeds,dim=0)
cf_text_embeds=torch.concat(cf_text_embeds,dim=0)
image_embeds=torch.concat(image_embeds,dim=0)
# torch.save(true_text_embeds,"dogs_true_text_embeds.pth")
# torch.save(cf_text_embeds,"dogs_cf_text_embeds.pth")
# torch.save(image_embeds,"dogs_image_embeds.pth")
print(true_text_embeds.size(),cf_text_embeds.size(),true_text_embeds.size())
        
batch_size = 129
my_dog_dataset = MyDataset(true_text_embeds,cf_text_embeds,image_embeds)
dataloader = DataLoader(my_dog_dataset, batch_size=batch_size, shuffle=True)
my_clip_text_projection = torch.nn.Linear(clip_model.text_projection.in_features,clip_model.text_projection.out_features,bias=False).to(device)
my_clip_text_projection.load_state_dict(clip_model.text_projection.state_dict())
my_clip_text_projection = train(100,dataloader,my_clip_text_projection,1e-4)