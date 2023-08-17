accelerate launch --config_file=acc_config --num_cpu_threads_per_process=1 text_experiment.py \
                  --dataset_config=data/conan_dataset_config.toml \
                  --config_file=data/conan_training_config_2.toml          
	
