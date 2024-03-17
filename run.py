
import argparse,torch
from sd import reproduce_stable_diffusion_results
from clip import *
from data_aug import data_augmentation


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="An example script to parse command-line arguments.")

    # Add arguments
    parser.add_argument("--dataset",type=str,choices=["custom","countbench"],help="choose from custom dataset or countbench")
    parser.add_argument("--data_path",type=str,help="path to custom data")
    parser.add_argument("--eval_dir",type=str,help="directory to save evaluation results")
    parser.add_argument("--task",type=str,choices=["classification","image_retrievel","image_gen"],help="choose the task")
    
    parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    parser.add_argument('--load_trained_text_projection', action='store_true')
    parser.add_argument("--trained_text_projection_path",type=str)   

    parser.add_argument("--ref_obj",type=str,default="dogs",help="name of the object being used as an reference")   
    parser.add_argument("--ref_obj_file",type=str,default=None,help="path to the ref objects")   
    parser.add_argument("--task_name",type=str,default="")   

    parser.add_argument('--use_only_number_word', action='store_true')
    parser.add_argument("--normalize_number_word",action='store_true')   
    parser.add_argument('--use_random_vector', action='store_true')
    parser.add_argument("--random_seed",type=int,default=None)   
    parser.add_argument('--use_muti_objs', action='store_true')

    args = parser.parse_args()

    sample_size = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ref_obj_file is not None:
        with open(args.ref_obj_file, 'r') as file:
            ref_objs = file.readlines()
        ref_objs = [line.strip() for line in ref_objs]
    else:
        ref_objs = [args.ref_obj]

    # run image generation with stable diffucsion
    if args.task == "image_gen" and args.model == "stable_diffusion":
        pretrained_model_name="CompVis/stable-diffusion-v1-4"
        reproduce_stable_diffusion_results(args.eval_dir,pretrained_model_name,device)
    elif "clip" in args.model:
        model = CLIPModel.from_pretrained(args.model)
        processor = CLIPProcessor.from_pretrained(args.model)
        if args.load_trained_text_projection:
            print(f"Loading trained text projection from {args.trained_text_projection_path}")
            model.text_projection.load_state_dict(torch.load(args.trained_text_projection_path))
        for name,param in model.named_parameters():
            param.requires_grad = False
        model = model.to(device)

        if args.dataset=="custom":
            augmented_data = data_augmentation(torch.load(args.data_path))
            
            if args.task == "image_retrievel" :
                image_retrievel(
                    model_name=args.model,
                    ref_obj=args.ref_obj,
                    target_data=augmented_data,
                    eval_dir=args.eval_dir,
                    device=device
                )
            elif args.task == "classification":
                img_clf_custom(
                    model_name=args.model,
                    model=model,
                    processor=processor,
                    ref_objs=ref_objs,
                    target_data=augmented_data,
                    eval_dir=args.eval_dir,
                    task_name=args.task_name,
                    device=device,
                    use_only_number_word=args.use_only_number_word,
                    normalize_number_word=args.normalize_number_word,
                    use_random_vector=args.use_random_vector,
                    random_seed=args.random_seed,
                    use_muti_objs=args.use_muti_objs,
                )
        elif (args.dataset=="countbench") and (args.task == "classification"):
            # TODO: load dataset
            # TODO: run on countbench
            countbench_data = None


