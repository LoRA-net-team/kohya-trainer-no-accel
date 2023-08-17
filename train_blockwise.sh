CUDA_VISIBLE_DEVICES=0 python -u train_blockwise.py --device cuda \
                              --trg_num_1 17 \
                              --unet_lr 0.000025 \
                              --text_encoder_lr 0.00005 \
                              --max_train_epochs 20 \
                              --output_dir lora_block_test