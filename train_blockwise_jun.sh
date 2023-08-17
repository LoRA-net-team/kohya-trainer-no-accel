CUDA_VISIBLE_DEVICES=4 python -u train_blockwise_jun2.py --device cuda \
                              --unet_lr 0.000035 \
                              --text_encoder_lr 0.00007 \
                              --max_train_epochs 10 \
                              --output_dir lora_block_test