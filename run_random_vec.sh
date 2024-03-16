for model in "openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16" "openai/clip-vit-large-patch14"; do
    for obj in "dogs" "cats"; do
        eval_dir="${obj}_use_random_vector"
        for random_seed in 20213 334143 854241 290940 811370 492774 155208 875054 251101 830373; do
            task_name="${obj}_use_random_vector_${random_seed}"
            python run.py \
                --dataset "custom" \
                --data_path "../../testadapt/my_data/my_eval_data_all.pth" \
                --eval_dir $eval_dir \
                --task "classification" \
                --model $model \
                --task_name $task_name \
                --ref_obj $obj \
                --use_random_vector \
                --random_seed $random_seed
        done
    done
done