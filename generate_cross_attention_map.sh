CUDA_VISIBLE_DEVICES=1 python generate_cross_attention_map.py \
                       --device cuda \
                       --lora_block_weight "[1, 0, 0, 0, 0, 1,1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]" \
                       --pretrained_model_name_or_path  pretrained/AnythingV5Ink_v5PrtRE.safetensors  \
                       --lora_file_dir
                       --trg_img_folder result/in/epoch_1 \
                       --thredshold 0.05