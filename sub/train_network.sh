accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 train_network.py \
                  --dataset_config=data/mina_jason_dataset_config.toml \
                  --config_file=data/mina_jason_training_config.toml
