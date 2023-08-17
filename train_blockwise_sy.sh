CUDA_VISIBLE_DEVICES=0 python -u train_blockwise_sy.py \
                              --device cuda \
                              --unet_lr 0.00003 \
                              --text_encoder_lr 0.00001 \
                              --max_train_epochs 10 \
                              --train_data_dir data/haibara \
                              --sample_prompts test/test_prompt.txt \
                              --resolution "512,512" \
                              --output_dir result/5_haibara_gradient_clipping \
                              --pretrained_model_name_or_path pretrained/AnythingV5Ink_v5PrtRE.safetensors \
                              --network_alpha 4 --network_dim 32 \
                              --save_last_n_epochs 20 \
                              --target_character 'haibara ai' \
                              --compare_character girl \
                              --output_name test_sailormoon \
                              --target_block_weight "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
# /data3/space/space-cowork-sdwebui/sdwebui/stable-diffusion-webui/models/Stable-diffusion/novelai-all.ckpt
