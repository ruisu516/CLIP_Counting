from clip_count_utils import *
import os
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import pandas as pd
import torch
from clip_count_utils import *
import os
from tqdm import tqdm
from my_datasets import *
import pickle


DESIRED_COLUMNS = ["average", "dogs", "cats", "lions", "chairs", "goats", "cows", "cherries", "roses", "boats", "ref"]

# model_name= "openai/clip-vit-base-patch32" # "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16"
# model_name="openai/clip-vit-large-patch14"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cuda"
# model = CLIPModel.from_pretrained(model_name).to(device)
# model.requires_grad=False
# processor = CLIPProcessor.from_pretrained(model_name)


def get_file_name(task,model_name,ref,data_name="",num_classes="",factor=1,extension="csv"):
    if ref is None:
        ref_str = "None"
    elif isinstance(ref,str):
        ref_str = ref
    elif isinstance(ref,list):
        ref_str = f"{len(ref)}_refs"
    return f"{task}_{model_name.split('/')[1]}_{ref_str}_{data_name}_{num_classes}{factor}.{extension}"


def image_retrievel(model_name,ref_obj,sample_size,augmented_data,eval_dir,device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(model_name)

    normalize=False
    # sample_size = len(augmented_data['dogs'][2])
    num_classes = 4
    linear_shift=True
    task = 'image_retrievel'
    start_with_target_with_num = True


    all_probs_by_factors = []
    all_mean_probs_by_factors = []
    all_probs_by_target = []
    for factor in [0,1]: # output original results and results after applying our method
        # all_probs_by_target = []
        all_mean_probs_by_target = []
        for target in augmented_data.keys():
            all_probs_per_class=run_on_my_data_img_retrievel(
                model=model,
                processor=processor,
                target_data= augmented_data[target],
                target=target,
                ref=ref_obj,
                normalize=normalize,
                device=device,
                factor=factor,
                sample_size=sample_size,
                num_classes=num_classes,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num)
            mean_prob = np.mean([all_probs_per_class[i][i] for i in range(len(all_probs_per_class))])
            all_mean_probs_by_target.append(mean_prob)
            # all_probs_by_target.append(all_probs_per_class)
        # all_probs_by_factors.append(all_probs_by_target)
        all_mean_probs_by_factors.append(all_mean_probs_by_target)

    # pb_pd = pd.DataFrame(all_probs_by_factors,columns=list(augmented_data.keys()))
    # pb_pd.index = factors_list[1:]
    # pb_pd.to_csv(f"csv/final/{fn}")

    mean_pb_pd = pd.DataFrame(all_mean_probs_by_factors,columns=list(augmented_data.keys()))
    mean_pb_pd.index = [[ele]*len(all_mean_probs_by_target) for ele in [0,1]]
    mean_pb_pd["average"] = np.array(all_mean_probs_by_factors).mean(axis=1)
    mean_pb_pd.to_csv(os.path.join(eval_dir,get_file_name(task,model_name,ref_obj,data_name="custom_data",num_classes=num_classes)))

def generate_random_vectors(shape, seed,N=10):
    torch.manual_seed(seed)
    vectors = []
    
    for _ in range(N):
        vec = torch.randn(*shape)
        vectors.append(vec)
        
    concatenated_vectors = torch.stack(vectors)
    return concatenated_vectors
def img_clf_custom(
        model_name,
        model,
        processor,
        ref_objs,
        target_data,
        eval_dir,
        task_name="img_clf_custom",
        factor=1,
        device="gpu",
        normalize=False,
        linear_shift=True,
        start_with_target_with_num = True,
        use_only_number_word=False,
        normalize_number_word=False,
        use_random_vector=False,
        random_seed=None,
        use_multi_objs=False,
):
    """
        run image classification on custom dataset
    """
    

    print(f"Running task {task_name}, using model {model_name}")
    # Initialize an empty list to hold accuracy by ref to build up the DataFrame incrementally
    

    # Check if the evaluation directory exists, if not, create it
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    # Initialize or clear the file
    file_name = get_file_name(task_name, model_name, ref_objs, data_name="", num_classes="", factor=factor)
    file_path = os.path.join(eval_dir, file_name)
    # Create an empty DataFrame or clear the existing file to start fresh
    pd.DataFrame().to_csv(file_path)

    if not use_multi_objs:
        iteration = ref_objs 
    else:
        iteration = [ref_objs]
    acc_by_ref = []
    for ref in tqdm(iteration, desc=f"Processing refs for task {task_name}"):
        acc_by_target = []
        # Adding a progress bar for iterating over target_data keys
        print(f"Processing targets for ref: {ref}")
        for target in tqdm(target_data.keys()):
            _, _, acc = run_on_my_data_clf(
                model=model,
                processor=processor,
                target_data=target_data[target],
                target=target,
                ref=ref,
                factor=factor,
                normalize=normalize,
                device=device,
                num_classes=4,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num,
                use_only_number_word=use_only_number_word,
                normalize_number_word=normalize_number_word,
                use_random_vector=use_random_vector,
                random_seed=random_seed,
                use_multi_objs=use_multi_objs,
            )
            acc_by_target.append(acc)
        acc_by_ref.append(acc_by_target)

        # Update the DataFrame and save after processing each `ref`
        acc_pd = pd.DataFrame(np.array(acc_by_ref), columns=list(target_data.keys()))
        acc_pd["average"] = np.array(acc_by_ref).mean(axis=1)
        acc_pd["ref"] = iteration[:len(acc_by_ref)]  # Match the length of acc_by_ref to avoid index out of bounds
        # Save/update the CSV file
        print(f"update {file_path}...")
        try:
            acc_pd = acc_pd[DESIRED_COLUMNS]
        except:
            pass
        acc_pd.to_csv(file_path, index=False)  # Use index=False to avoid writing row indices

def img_clf_countbench(
        model_name,
        eval_dir,
        data_path,
        model,
        processor,
        num_classes,
        task_name="img_clf_countbench",
        ref_obj=None,
        normalize=False,
        factor=1,
        linear_shift=True,
        test_bz=32,
        use_target_obj_with_context=True,
        use_target_aug_sent_with_context=True,
        use_ref_with_context=False, # TODO: check this part
        start_with_target_with_num=True,
        device="cuda",
        use_only_number_word=False,
        normalize_number_word=False,
        use_multi_objs=False,
        use_self_as_ref=False,
):
    def get_ref_embed_helper(ref_obj):
        # print("target_obj_aug_with_context_text",target_obj_aug_with_context_text)
        # print("target_obj_with_context_text",target_obj_with_context_text)
        # print("target_obj_text",target_obj_text)
        if use_ref_with_context:
            ref_object = [context.replace(org,ref_obj) for org,context in zip(target_obj_text,target_obj_with_context_text)]
            ref_aug_sentences = [item.replace(target,ref_obj) for tuple_ in target_obj_aug_with_context_text for item,target in zip(tuple_,target_obj_text)]

        else:
            ref_object = [ref_obj]*len(target_obj_text)
            ref_aug_sentences=[]
            for number in NUMBER_WORDS[:num_classes]:
                ref_aug_sentences += [f"{number} {ref_obj}"]*batch_size
        
        # print(ref_aug_sentences)
        # print(ref_object)
        return get_ref_difference(
            ref_aug_sentences=ref_aug_sentences,
            ref_object=ref_object,
            model=model,
            processor=processor,
            device=device,
            normalize=normalize,
            batch_first=True
        )
        
    print("use_only_number_word",use_only_number_word,"normalize_number_word",normalize_number_word)
    print("use_ref_with_context",use_ref_with_context)
    print("use_multi_objs",use_multi_objs,len(ref_obj) if use_multi_objs else None)
    print("ref_obj",ref_obj,factor)
    print("use_target_obj_with_context",use_target_obj_with_context)
    print("use_target_aug_sent_with_context",use_target_aug_sent_with_context)
    print("use_self_as_ref",use_self_as_ref)
    countbench_dataset = ProcessedCountBenchDataset(
        data=torch.load(data_path,map_location=device),
        device=device,
        num_classes=num_classes
    )
    print("len(countbench_dataset)",len(countbench_dataset))
    countbench_dataloader = DataLoader(countbench_dataset, batch_size=test_bz, shuffle=False)

    predictions = []
    gt_labels = []
    
    for image_embeds,target_obj_text,target_obj_aug_text,target_obj_with_context_text,target_obj_aug_with_context_text, gt_count in countbench_dataloader:
        batch_size = len(image_embeds)
        
        if use_only_number_word:
            ref_aug_sentences=[f"{word}" for word in NUMBER_WORDS[:num_classes]]
            ref_diff = text2embedding(ref_aug_sentences,model,processor,device,True).unsqueeze(0).repeat(batch_size, 1, 1)
        if ref_obj is not None:
            if use_multi_objs:
                ref_diff_list,ref_prompt_single_list=[],[]
                for r in ref_obj:
                    ref_diff,ref_prompt_single = get_ref_embed_helper(r)
                    ref_diff_list.append(ref_diff)
                    ref_prompt_single_list.append(ref_prompt_single)
                    del ref_diff,ref_prompt_single
            else:
                ref_diff,ref_prompt_single = get_ref_embed_helper(ref_obj)
                
            
        target_obj_aug_with_context_text = [item for tuple_ in target_obj_aug_with_context_text for item in tuple_]
        target_obj_aug_text = [item for tuple_ in target_obj_aug_text for item in tuple_]

        if use_target_obj_with_context:
            target_embeds = text2embedding(target_obj_with_context_text,model,processor,device,normalize)
        else:
            target_embeds = text2embedding(target_obj_text,model,processor,device,normalize)

        if use_target_aug_sent_with_context:
            target_aug_embeds = text2embedding(target_obj_aug_with_context_text,model,processor,device,normalize)
        else:
            target_aug_embeds = text2embedding(target_obj_aug_text,model,processor,device,normalize)
        # target_aug_embeds = target_aug_embeds.reshape(-1,batch_size,target_aug_embeds.shape[-1]).permute(1,0,2)
        target_aug_embeds = target_aug_embeds[np.arange(batch_size * num_classes).reshape(num_classes,batch_size).T.flatten()].reshape(batch_size, num_classes, -1)
        
        """
            target_embeds: (bz, embed_dim)
            target_aug_embeds: (bz, num_class, embed_dim)
            ref_diff: (bz, num_class, embed_dim)
            ref_prompt_single: (bz, 1, embed_dim)
        """

        if use_self_as_ref:
            ref_embeds = text2embedding(target_obj_text,model,processor,device,normalize)[:,None,:]
            ref_aug_embeds = text2embedding(target_obj_aug_text,model,processor,device,normalize)
            ref_aug_embeds = ref_aug_embeds[np.arange(batch_size * num_classes).reshape(num_classes,batch_size).T.flatten()].reshape(batch_size, num_classes, -1)

            ref_diff = ref_embeds - ref_aug_embeds
            # print("ref_diff.shape",ref_diff.shape)

        if use_only_number_word or use_self_as_ref:
            # print("ref_diff.shape",ref_diff.shape)
            # ref_diff_projection_2 = torch.bmm(ref_diff, target_embeds[...,None])/torch.sum(target_embeds * target_embeds, dim=-1, keepdim=True)[...,None]*target_embeds[:,None,:]
            ref_diff_projection_2 = (torch.sum(ref_diff*target_aug_embeds, dim=-1, keepdim=True)/torch.sum(target_aug_embeds * target_aug_embeds, dim=-1, keepdim=True))*target_aug_embeds
            ref_diff = ref_diff - ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
            if use_only_number_word and normalize_number_word and ref_obj is not None:
                obj_ref_diff,obj_ref_prompt_single = get_ref_embed_helper(ref_obj)
                obj_ref_diff_projection = (torch.bmm(obj_ref_diff, obj_ref_prompt_single.permute(0,2,1)) / torch.bmm(obj_ref_prompt_single, obj_ref_prompt_single.permute(0,2,1))) * obj_ref_prompt_single
                # obj_ref_diff_projection_2 = torch.bmm(obj_ref_diff-obj_ref_diff_projection, target_embeds[...,None])/torch.sum(target_embeds * target_embeds, dim=-1, keepdim=True)[...,None]*target_embeds[:,None,:]
                obj_ref_diff_projection_2 = (torch.sum((obj_ref_diff-obj_ref_diff_projection)*target_aug_embeds, dim=-1, keepdim=True)/torch.sum(target_aug_embeds * target_aug_embeds, dim=-1, keepdim=True))*target_aug_embeds
                obj_ref_diff = obj_ref_diff - obj_ref_diff_projection - obj_ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)

                ref_diff = ref_diff * obj_ref_diff.norm(p=2,dim=-1,keepdim=True) / ref_diff.norm(p=2,dim=-1,keepdim=True)
            # print("ref_diff.shape",ref_diff.shape)
            merged_text_embeds = apply_reff_diff(target_embeds,target_aug_embeds,ref_diff,factor,linear_shift,start_with_target_with_num)
        elif ref_obj is None:
            # print("ref_obj is None")
            merged_text_embeds = target_aug_embeds
        elif use_multi_objs:
            orth_ref_diff = 0
            for ref_diff, ref_prompt_single in zip(ref_diff_list,ref_prompt_single_list):
                ref_diff_projection = (torch.bmm(ref_diff, ref_prompt_single.permute(0,2,1)) / torch.bmm(ref_prompt_single, ref_prompt_single.permute(0,2,1))) * ref_prompt_single
                # ref_diff_projection_2 = torch.bmm(ref_diff-ref_diff_projection, target_embeds[...,None])/torch.sum(target_embeds * target_embeds, dim=-1, keepdim=True)[...,None]*target_embeds[:,None,:]
                ref_diff_projection_2 = (torch.sum((ref_diff-ref_diff_projection)*target_aug_embeds, dim=-1, keepdim=True)/torch.sum(target_aug_embeds * target_aug_embeds, dim=-1, keepdim=True))*target_aug_embeds
                orth_ref_diff += ref_diff - ref_diff_projection - ref_diff_projection_2#+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
                del ref_diff,ref_prompt_single,ref_diff_projection,ref_diff_projection_2
            merged_text_embeds = apply_reff_diff(target_embeds,target_aug_embeds,orth_ref_diff/len(ref_obj),factor,linear_shift,start_with_target_with_num)
        else:
            ref_diff_projection = (torch.bmm(ref_diff, ref_prompt_single.permute(0,2,1)) / torch.bmm(ref_prompt_single, ref_prompt_single.permute(0,2,1))) * ref_prompt_single
            # ref_diff_projection_2 = torch.bmm(ref_diff-ref_diff_projection, target_embeds[...,None])/torch.sum(target_embeds * target_embeds, dim=-1, keepdim=True)[...,None]*target_embeds[:,None,:]
            ref_diff_projection_2 = (torch.sum((ref_diff-ref_diff_projection)*target_aug_embeds, dim=-1, keepdim=True)/torch.sum(target_aug_embeds * target_aug_embeds, dim=-1, keepdim=True))*target_aug_embeds

            ref_diff = ref_diff - ref_diff_projection - ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
            merged_text_embeds = apply_reff_diff(target_embeds,target_aug_embeds,ref_diff,factor,linear_shift,start_with_target_with_num)

        merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)
        _,logits_per_image= get_logits(model,merged_text_embeds,image_embeds.to(device)) # (bz,1,num_classes)
        # print(merged_text_embeds.shape,image_embeds.shape)

        # print(logits_per_image.shape,torch.argmax(logits_per_image,dim=-1).shape)
        # predictions.extend((torch.argmax(logits_per_image,dim=-1).squeeze().detach().cpu().numpy()+2).tolist())
        result = torch.argmax(logits_per_image,dim=-1).squeeze().detach().cpu().numpy()+2
        if isinstance(result, np.ndarray):
            predictions.extend(result.tolist())
        else:
            predictions.append(result)
        gt_labels.extend(gt_count.tolist())
    
    file_name = get_file_name(task_name, model_name, ref_obj, data_name="countbench", num_classes=num_classes, factor=factor, extension="pth")
    print(f"Saving to {os.path.join(eval_dir,file_name)}")
    acc = np.round((np.array(predictions)==np.array(gt_labels)).mean()*100,2)
    print("acc",acc)
    # with open(os.path.join(eval_dir,file_name), 'w') as f:
    #     json.dump({
    #         "predictions":predictions,
    #         "gt_labels":gt_labels,
    #         "acc":float(acc)
    #     }, f)
    with open(os.path.join(eval_dir,file_name), 'wb') as f:
        pickle.dump({
            "predictions":predictions,
            "gt_labels":gt_labels,
            "acc":float(acc)
        }, f)

    

