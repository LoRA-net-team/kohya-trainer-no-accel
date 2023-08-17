import argparse
import gc
import math
import os
from multiprocessing import Value
from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from BlockwiseLora import library as train_util, library as huggingface_util, library as config_util, \
    library as custom_train_functions
from BlockwiseLora.library import (ConfigSanitizer, BlueprintGenerator, )
from BlockwiseLora.library import (apply_snr_weight, prepare_scheduler_for_custom_training,
                                   pyramid_noise_like, apply_noise_offset, scale_v_prediction_loss_like_noise_prediction, )
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

def save_weights(file, updated_embs, save_dtype):
    state_dict = {"emb_params": updated_embs}

    if save_dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, file)
    else:
        torch.save(state_dict, file)  # can be loaded in Web UI
def load_weights(self, file):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        data = load_file(file)
    else:
        # compatible to Web UI's file format
        data = torch.load(file, map_location="cpu")
        if type(data) != dict:
            raise ValueError(f"weight file is not dict / 重みファイルがdict形式ではありません: {file}")
        if "string_to_param" in data:  # textual inversion embeddings
            data = data["string_to_param"]
            if hasattr(data, "_parameters"):  # support old PyTorch?
                data = getattr(data, "_parameters")
    emb = next(iter(data.values()))
    if type(emb) != torch.Tensor:
        raise ValueError(f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {file}")
    if len(emb.size()) == 1:
        emb = emb.unsqueeze(0)
    return [emb]

def train(args):

    print(f'\n step 1. output name')
    if args.output_name is None:
        args.output_name = args.token_string

    print(f'\n step 2. template')
    use_template = args.use_object_template or args.use_style_template
    train_util.verify_training_args(args)

    print(f'\n step 3. training data argument')
    args.caption_extension = args.caption_extention
    args.caption_extention = None
    args.resolution = tuple([int(r) for r in args.resolution.split(",")])
    args.face_crop_aug_range = None

    print(f'\n step 4. cache latents and seed')
    cache_latents = args.cache_latents
    if args.seed is not None:
        set_seed(args.seed)

    print(f'\n step 5. tokenizer')
    tokenizer = train_util.load_tokenizer(args)

    print(f'\n step 6. prepare accelerator and model')
    accelerator, unwrap_model = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

    print(f'\n step 7. init token id')
    if args.init_word is not None:
        init_token_ids = tokenizer.encode(args.init_word, add_special_tokens=False)
        print(f'init_token_ids : {init_token_ids}')
        if len(init_token_ids) > 1 and len(init_token_ids) != args.num_vectors_per_token:
            print(
                f"token length for init words is not same to num_vectors_per_token, init words is repeated or truncated")
    else:
        init_token_ids = None

    print(f'\n step 8. add one trainable token')
    token_strings = [args.token_string] + [f"{args.token_string}{i + 1}" for i in range(args.num_vectors_per_token - 1)]
    num_added_tokens = tokenizer.add_tokens(token_strings)
    print(f'num_added_tokens : {num_added_tokens}')
    assert (num_added_tokens == args.num_vectors_per_token), f"tokenizer has same word to token string. please use another one : {args.token_string}"
    token_ids = tokenizer.convert_tokens_to_ids(token_strings)
    print(f"tokens are added: {token_ids}")
    assert min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(
        token_ids) - 1, f"token ids is not ordered"
    assert len(tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(tokenizer)}"
    text_encoder.resize_token_embeddings(len(tokenizer))

    print(f'\n step 9. change text encoder embeddings')
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if init_token_ids is not None:
        for i, token_id in enumerate(token_ids): # trainable token id
            token_embeds[token_id] = token_embeds[init_token_ids[i % len(init_token_ids)]]

    print(f'\n step 10. loading pretrained weight')
    if args.weights is not None:
        embeddings = load_weights(args.weights)
        assert len(token_ids) == len(
            embeddings), f"num_vectors_per_token is mismatch for weights / 指定した重みとnum_vectors_per_tokenの値が異なります: {len(embeddings)}"
        # print(token_ids, embeddings.size())
        for token_id, embedding in zip(token_ids, embeddings):
            token_embeds[token_id] = embedding
            # print(token_id, token_embeds[token_id].mean(), token_embeds[token_id].min())
        print(f"weighs loaded")
    print(f"create embeddings for {args.num_vectors_per_token} tokens, for {args.token_string}")

    print(f'\n step 11. dataset')
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False))
        if args.dataset_config is not None:
            print(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                print(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)))
        else:
            use_dreambooth_method = args.in_json is None
            if use_dreambooth_method:
                print("Use DreamBooth method.")
                user_config = {"datasets": [{"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,
                                                                                              args.reg_data_dir)}]}
            else:
                print("Train with captions.")
                user_config = {"datasets": [{"subsets": [{"image_dir": args.train_data_dir,"metadata_file": args.in_json, }]}]}
        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

    print(f'\n step 12. dataloader')
    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)
    if use_template:
        print("use template for training captions. is object: {args.use_object_template}")
        templates = imagenet_templates_small if args.use_object_template else imagenet_style_templates_small
        replace_to = " ".join(token_strings)
        captions = []
        for tmpl in templates:
            print(f'tmpl.format(replace_to) : {tmpl.format(replace_to)}')
            captions.append(tmpl.format(replace_to))
        train_dataset_group.add_replacement("", captions)
        if args.num_vectors_per_token > 1:
            prompt_replacement = (args.token_string, replace_to)
        else:
            prompt_replacement = None
    else:
        if args.num_vectors_per_token > 1:
            replace_to = " ".join(token_strings)
            train_dataset_group.add_replacement(args.token_string, replace_to)
            prompt_replacement = (args.token_string, replace_to)
        else:
            prompt_replacement = None
    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, show_input_ids=True)
        return
    if len(train_dataset_group) == 0:
        print("No data found. Please verify arguments / 画像がありません。引数指定を確認してください")
        return

    if cache_latents:
        assert (train_dataset_group.is_latent_cacheable()), "when caching latents, either color_aug or random_crop cannot be used "
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)
    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk,
                                              accelerator.is_main_process)
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        accelerator.wait_for_everyone()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
    print("prepare optimizer, data loader etc.")

    print(f'\n step 13. what to optimize')
    trainable_params = text_encoder.get_input_embeddings().parameters()
    _, _, optimizer = train_util.get_optimizer(args, trainable_params)

    print(f'\n step 14. dataset loader')
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset_group,batch_size=1,shuffle=True,
                                                   collate_fn=collater,num_workers=n_workers,
                                                   persistent_workers=args.persistent_data_loader_workers,)

    print(f'\n step 15. train epochs')
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
        print(f"override steps. steps for {args.max_train_epochs} epochs is / : {args.max_train_steps}")
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    print(f'\n step 16. lr')
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 17. model to accelerator')
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(text_encoder,
                                                                                  optimizer,
                                                                                  train_dataloader,
                                                                                  lr_scheduler)
    text_encoder, unet = train_util.transform_if_model_is_DDP(text_encoder, unet)

    print(f'\n step 18. index no updates')
    index_no_updates = torch.arange(len(tokenizer)) < token_ids[0]
    orig_embeds_params = unwrap_model(text_encoder).get_input_embeddings().weight.data.detach().clone()

    print(f'\n step 19. freeze all parameters except for the token embeddings in text encoder')
    text_encoder.requires_grad_(True)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    if args.gradient_checkpointing:  # according to TI example in Diffusers, train is required
        unet.train()
    else:
        unet.eval()
    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)
        text_encoder.to(weight_dtype)

    print(f'\n step 20. resume ..?')
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print(f'\n step 21. running training')
    print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    print(f"  num epochs / epoch数: {num_train_epochs}")
    print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    print(f"  total train batch size (with parallel & distributed & accumulation) : {total_batch_size}")
    print(f"  gradient ccumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,
                        desc="steps")
    global_step = 0
    print(f' (21.1) scheduler')
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    num_train_timesteps=1000, clip_sample=False)
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion" if args.log_tracker_name is None else args.log_tracker_name)

    print(f' (21.2) saving and remove function')
    def save_model(ckpt_name, embs, steps, epoch_no, force_sync_upload=False):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        save_weights(ckpt_file, embs, save_dtype)
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    print(f' (21.3) ')
    # training loop
    for epoch in range(num_train_epochs):
        print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1
        text_encoder.train()
        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(text_encoder):
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device)
                    else:
                        # latentに変換
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                b_size = latents.shape[0]
                input_ids = batch["input_ids"].to(accelerator.device)
                encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizer, text_encoder, torch.float)
                noise = torch.randn_like(latents, device=latents.device)
                if args.noise_offset:
                    noise = apply_noise_offset(latents, noise, args.noise_offset, args.adaptive_noise_scale)
                elif args.multires_noise_iterations:
                    noise = pyramid_noise_like(noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])
                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights
                if args.min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                if args.scale_v_pred_loss_like_noise_pred:
                    loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = text_encoder.get_input_embeddings().parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    unwrap_model(text_encoder).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                # save zaradress
                train_util.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, prompt_replacement)
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                # save model
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        updated_embs = unwrap_model(text_encoder).get_input_embeddings().weight[token_ids].data.detach().clone()
                        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                        save_model(ckpt_name, updated_embs, global_step, epoch)
                        if args.save_state:
                            train_util.save_and_remove_state_stepwise(args, accelerator, global_step)
                        remove_step_no = train_util.get_remove_step_no(args, global_step)
                        if remove_step_no is not None:
                            remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                            remove_model(remove_ckpt_name)
            current_loss = loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():  # tracking d*lr value
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"])
                accelerator.log(logs, step=global_step)
            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs, step=epoch + 1)
        accelerator.wait_for_everyone()
        updated_embs = unwrap_model(text_encoder).get_input_embeddings().weight[token_ids].data.detach().clone()



        # ----------------------------------------------------------------------------------------------------------------------------------------
        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
            if accelerator.is_main_process and saving:
                ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                save_model(ckpt_name, updated_embs, epoch + 1, global_step)
                remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                    remove_model(remove_ckpt_name)
                if args.save_state:
                    train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
            # -----------------------------------------------------------------------------------------------------------------------------

        train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, prompt_replacement)
        # end of epoch
    is_main_process = accelerator.is_main_process
    if is_main_process:
        text_encoder = unwrap_model(text_encoder)

    accelerator.end_training()

    if args.save_state and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    updated_embs = text_encoder.get_input_embeddings().weight[token_ids].data.detach().clone()

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        save_model(ckpt_name, updated_embs, global_step, num_train_epochs, force_sync_upload=True)
        print("model saved.")

def save_weights(file, updated_embs, save_dtype):
    state_dict = {"emb_params": updated_embs}
    if save_dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v
    print(f'saving weight dir : {file}')
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        save_file(state_dict, file)
    else:
        torch.save(state_dict, file)  # can be loaded in Web UI


def load_weights(file):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        data = load_file(file)
    else:
        data = torch.load(file, map_location="cpu")
        if type(data) != dict:
            raise ValueError(f"weight file is not dict / 重みファイルがdict形式ではありません: {file}")
        if "string_to_param" in data:  # textual inversion embeddings
            data = data["string_to_param"]
            if hasattr(data, "_parameters"):  # support old PyTorch?
                data = getattr(data, "_parameters")
    emb = next(iter(data.values()))
    if type(emb) != torch.Tensor:
        raise ValueError(f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {file}")
    if len(emb.size()) == 1:
        emb = emb.unsqueeze(0)
    return emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true",
                        help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む")
    parser.add_argument("--v_parameterization", action="store_true",
                        help="enable v-parameterization training / v-parameterization学習を有効にする")
    parser.add_argument("--pretrained_model_name_or_path",type=str,
                        default='pretrained/animefull-final-pruned-fp16.safetensors')
    parser.add_argument("--tokenizer_cache_dir",type=str,default=None,)
    parser.add_argument("--train_data_dir",type=str, default="data/datas/150_boy")
    parser.add_argument("--shuffle_caption", action="store_false",)
    parser.add_argument("--caption_extension", type=str, default=".txt",)
    parser.add_argument("--caption_extention", type=str, default=".txt",)
    parser.add_argument("--keep_tokens",type=int,default=1,)
    parser.add_argument("--color_aug", action="store_true",
                        help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
    parser.add_argument("--flip_aug", action="store_true",
                        help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
    parser.add_argument("--face_crop_aug_range",type=str,default=None,)
    parser.add_argument("--random_crop",action="store_true",)
    parser.add_argument("--debug_dataset", action="store_true",)
    parser.add_argument("--resolution",type=str,default='512,512')
    parser.add_argument("--cache_latents",action="store_true",)
    parser.add_argument("--vae_batch_size", type=int, default=1,
                        help="batch size for caching latents / latentのcache時のバッチサイズ")
    parser.add_argument("--cache_latents_to_disk", action="store_true",
                        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM",)
    parser.add_argument("--enable_bucket", action="store_true",
                        help="enable buckets for multi aspect ratio training")
    parser.add_argument("--min_bucket_reso", type=int, default=256,
                        help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024,
                        help="maximum resolution for buckets / bucketの最大解像度")
    parser.add_argument("--bucket_reso_steps",  type=int,default=64,)
    parser.add_argument("--bucket_no_upscale", action="store_true",)
    parser.add_argument("--token_warmup_min",type=int,default=1,)
    parser.add_argument("--token_warmup_step",type=float,default=0,)
    parser.add_argument("--dataset_class",type=str,
                        default=None,)
    parser.add_argument("--caption_dropout_rate", type=float, default=0.0,)
    parser.add_argument("--caption_dropout_every_n_epochs",type=int,default=0,)
    parser.add_argument("--caption_tag_dropout_rate",type=float,default=0.0,)
    parser.add_argument("--reg_data_dir", type=str, default=None,
                        help="directory for regularization images / 正則化画像データのディレクトリ")
    parser.add_argument("--in_json", type=str, default=None,
                        help="json metadata for dataset / データセットのmetadataのjsonファイル")
    parser.add_argument("--dataset_repeats", type=int, default=1,)
    parser.add_argument("--output_dir", type=str, default='20230715_textual_inversion')
    parser.add_argument("--output_name", type=str,
                        default='textual_inversion_0715')
    parser.add_argument("--huggingface_repo_id", type=str, default=None,
                        help="huggingface repo name to upload / huggingface")
    parser.add_argument("--huggingface_repo_type", type=str, default=None,
                        help="huggingface repo type to upload / huggingface")
    parser.add_argument("--huggingface_path_in_repo",type=str,default=None,)
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument("--huggingface_repo_visibility",type=str,default=None,)
    parser.add_argument("--save_state_to_huggingface", action="store_true", help="save state to huggingface")
    parser.add_argument("--resume_from_huggingface",action="store_true",
                        help="resume from huggingface",)
    parser.add_argument("--async_upload",action="store_true",
                        help="upload to huggingface asynchronously / huggingface",)
    parser.add_argument("--save_precision",type=str,default=None,
                        choices=[None, "float", "fp16", "bf16"],
                        help="precision in saving / 保存時に精度を変更して保存する",)
    parser.add_argument("--save_every_n_epochs", type=int, default=None, help="save checkpoint every N epochs ")
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="save checkpoint every N steps")
    parser.add_argument("--save_n_epoch_ratio",type=int,default=None,)
    parser.add_argument("--save_last_n_epochs",type=int,default=None,)
    parser.add_argument("--save_last_n_epochs_state",type=int,default=None,)
    parser.add_argument("--save_last_n_steps",type=int,default=None,)
    parser.add_argument("--save_last_n_steps_state",type=int,default=None,)
    parser.add_argument("--save_state",action="store_true",
                        help="save training state additionally (including optimizer states etc.) / optimizer",)
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")
    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    parser.add_argument("--max_token_length",type=int,default=None,
                        choices=[None, 150, 225],)
    parser.add_argument("--mem_eff_attn",action="store_true",
                        help="use memory efficient attention for CrossAttention / CrossAttention")
    parser.add_argument("--xformers", action="store_true",
                        help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    parser.add_argument("--vae", type=str, default=None,)
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="enable gradient checkpointing / grandient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度"
    )
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is scripts-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / Multires noiseを有効にしてこのイテレーション数を設定する（有効にする場合は6-10程度を推奨）",
    )

    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations) / Multires noiseのdiscount値を設定する（--multires_noise_iterations指定時のみ有効）",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default) / latentの平均値の絶対値 * この値をnoise_offsetに加算する（Noneの場合は無効、デフォルト）",
    )
    parser.add_argument(
        "--lowram",
        action="store_true",
        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込むなど（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）",
    )

    parser.add_argument(
        "--sample_every_n_steps", type=int, default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する"
    )
    parser.add_argument("--sample_every_n_epochs",type=int,default=1,)
    parser.add_argument("--sample_prompts", type=str, default='test_text.txt')
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--output_config", action="store_true", help="output command line args to given .toml file / 引数を.tomlファイルに出力する"
    )
    parser.add_argument(
        "--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み"
    )

    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser, False)
    parser.add_argument("--save_model_as",type=str,
                        default="pt",choices=[None, "ckpt", "pt", "safetensors"],)
    parser.add_argument("--weights", type=str, default=None,
                        help="embedding weights to initialize / 学習するネットワークの初期重み")
    parser.add_argument("--num_vectors_per_token", type=int, default=1,)
    parser.add_argument("--token_string",type=str,default='150_boy')
    parser.add_argument("--init_word", type=str, default='boy',
                        help="words to initialize vector / ベクトルを初期化に使用する単語、複数可")
    parser.add_argument("--use_object_template", action="store_false",)
    parser.add_argument("--use_style_template",action="store_true",)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train(args)