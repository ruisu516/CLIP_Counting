from clip_count_utils import *
import os
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import pandas as pd
import torch
from clip_count_utils import *
import os
from tqdm import tqdm

DESIRED_COLUMNS = ["average", "dogs", "cats", "lions", "chairs", "goats", "cows", "cherries", "roses", "boats", "ref"]

# model_name= "openai/clip-vit-base-patch32" # "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16"
# model_name="openai/clip-vit-large-patch14"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cuda"
# model = CLIPModel.from_pretrained(model_name).to(device)
# model.requires_grad=False
# processor = CLIPProcessor.from_pretrained(model_name)


def get_file_name(task,model_name,ref,data_name="",num_classes="",factor=1):
    return f"{task}_{model_name.split('/')[1]}_{ref}_{data_name}_{num_classes}{factor}.csv"


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
        normalize_number_word=None,
        use_random_vector=False,
        random_seed=None,
        use_muti_objs=False,
):
    """
        run image classification on custom dataset
    """
    

    print(f"Running task {task_name}")
    # Initialize an empty list to hold accuracy by ref to build up the DataFrame incrementally
    

    # Check if the evaluation directory exists, if not, create it
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    # Initialize or clear the file
    file_name = get_file_name(task_name, model_name, len(ref_objs), data_name="", num_classes="", factor=factor)
    file_path = os.path.join(eval_dir, file_name)
    # Create an empty DataFrame or clear the existing file to start fresh
    pd.DataFrame().to_csv(file_path)

    if not use_muti_objs:
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
                use_muti_objs=use_muti_objs,
            )
            acc_by_target.append(acc)
        acc_by_ref.append(acc_by_target)

        # Update the DataFrame and save after processing each `ref`
        acc_pd = pd.DataFrame(np.array(acc_by_ref), columns=list(target_data.keys()))
        acc_pd["average"] = np.array(acc_by_ref).mean(axis=1)
        acc_pd["ref"] = iteration[:len(acc_by_ref)]  # Match the length of acc_by_ref to avoid index out of bounds
        # Save/update the CSV file
        print(f"update {file_path}...")
        acc_pd = acc_pd[DESIRED_COLUMNS]
        acc_pd.to_csv(file_path, index=False)  # Use index=False to avoid writing row indices


