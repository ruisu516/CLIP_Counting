for model in "openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16" "openai/clip-vit-large-patch14"; do
    echo "model: $model"
    for obj in "dogs" "cats"; do
        eval_dir="${obj}_normalize_number_word_revised_code"
        echo "eval_dir: $eval_dir"
        python run.py \
            --dataset "custom" \
            --data_path "../../testadapt/my_data/my_eval_data_all.pth" \
            --eval_dir $eval_dir \
            --task "classification" \
            --task_name "classification" \
            --model $model \
            --ref_obj $obj \
            --use_only_number_word \
            --normalize_number_word
    done
done