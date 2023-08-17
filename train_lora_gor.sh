CUDA_VISIBLE_DEVICES=0 python -u train_lora_gor.py \
                              --device cuda \
                              --unet_lr 0.00003 \
                              --text_encoder_lr 0.00001 \
                              --max_train_epochs 10 \
                              --train_data_dir data/sailormoon_5 \
                              --sample_prompts test/test_prompt.txt \
                              --resolution "512,512" \
                              --output_dir result/4_sailormoon_textual_inversion_version_data_low_lr \
                              --pretrained_model_name_or_path pretrained/AnythingV5Ink_v5PrtRE.safetensors \
                              --network_alpha 4 --network_dim 32 \
                              --save_last_n_epochs 20 \
                              --target_character sailormoon \
                              --compare_character girl \
                              --output_name test_sailormoon \
                              --target_block_weight "[1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]"

