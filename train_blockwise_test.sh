CUDA_VISIBLE_DEVICES=6 python -u train_blockwise_test.py \
                              --device cuda \
                              --unet_lr 0.00005 \
                              --text_encoder_lr 0.000025 \
                              --max_train_epochs 40 \
                              --train_data_dir data/sailormoon \
                              --sample_prompts test/test_prompt.txt \
                              --resolution "512,512" \
                              --output_dir result/4_sailormoon_orthogonal_test \
                              --pretrained_model_name_or_path pretrained/AnythingV5Ink_v5PrtRE.safetensors \
                              --network_alpha 4 --network_dim 32 \
                              --target_character sailormoon \
                              --compare_character girl \
                              --output_name test_sailormoon \
                              --target_block_weight "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
nohup sh train_blockwise_test.sh > nohup/4_sailormoon_orthogonal_test
