accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 \
                  inference_block_weight.py \
                  --lora_folder result/4_sailormoon_4_training_image/block_wise_11111111111111111/model \
                  --prompt "a picture of sailormoon" \
                  --target_block_weight "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 \
                  inference_block_weight.py \
                  --lora_folder result/4_sailormoon_4_training_image/block_wise_10000111111000000/model \
                  --prompt "a picture of sailormoon" \
                  --target_block_weight "[1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]"

accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 \
                  inference_block_weight.py \
                  --lora_folder result/4_sailormoon_4_training_image/block_wise_11111000000111111/model \
                  --prompt "a picture of sailormoon" \
                  --target_block_weight "[1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1]"
# --------------------------------------------------------------------------------------------------------------------#
accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 \
                  inference_block_weight.py \
                  --lora_folder result/4_sailormoon_4_training_image/block_wise_11111111111111111/model \
                  --prompt "a picture of sailormoon, white background, good quality" \
                  --target_block_weight "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"

accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 \
                  inference_block_weight.py \
                  --lora_folder result/4_sailormoon_4_training_image/block_wise_10000111111000000/model \
                  --prompt "a picture of sailormoon, white background, good quality" \
                  --target_block_weight "[1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]"

accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 \
                  inference_block_weight.py \
                  --lora_folder result/4_sailormoon_4_training_image/block_wise_11111000000111111/model \
                  --prompt "a picture of sailormoon, white background, good quality" \
                  --target_block_weight "[1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1]"