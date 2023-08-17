# reina token = 43847
import copy
import importlib
import argparse
import torch, os, sys
from library import train_util
from datetime import datetime
import safetensors.torch
from safetensors.torch import load_file
from collections import OrderedDict
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import time
from library.model_util import (load_checkpoint_with_text_encoder_conversion, create_unet_diffusers_config,
                                convert_ldm_unet_checkpoint, create_vae_diffusers_config, convert_ldm_vae_checkpoint,
                                convert_ldm_clip_checkpoint_v1)
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, logging
import torch
import json
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import torch, math
import argparse
from transformers import CLIPTokenizer
from networks.lora_block_weighing import PretrainedLoRANetwork
from attention_store import AttentionStore


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    torch_img = 2.0 * image - 1.0
    return torch_img


def register_attention_control(unet, controller):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, context=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            batch_size, sequence_length, _ = hidden_states.shape
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            dim = query.shape[-1]
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            # ----------------------------------------------------------------------------------------------------------------------------------
            # 1) attention score
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if is_cross_attention:
                attn = controller.forward(attention_probs, is_cross_attention, place_in_unet)
            # ----------------------------------------------------------------------------------------------------------------------------------
            # 2) after value calculating
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def main(args):
    print(f'\n step 1. dtype and device')
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    print(f'\n step 2. preparing accelerator')
    unique_time = str(datetime.time(datetime.now())).replace(':', '_')
    unique_time = unique_time.replace('.', '_')
    experiment_base_folder = os.path.join('../result', f'test_{unique_time}')
    os.makedirs(experiment_base_folder, exist_ok=True)
    args.logging_dir = os.path.join(experiment_base_folder, args.logging_dir)
    accelerator, unwrap_model = train_util.prepare_accelerator(args)

    print(f'\n step 3. original stable diffusion model')
    print(f' (3.1) tokenizer')
    version = args.tokenizer_version
    tokenizer = CLIPTokenizer.from_pretrained(version)

    print(f' (3.2) text encoder, vae, unet')
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

    print(f' (3.2) text encoder, vae, unet')
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

    print(f' (4.3) network and original model to device and state')
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 6. inference scheduler')
    scheduler_cls = EulerAncestralDiscreteScheduler
    sched_init_args = {}
    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"
    scheduler = scheduler_cls(num_train_timesteps=1000,
                              beta_start=0.00085,
                              beta_end=0.012,
                              beta_schedule="scaled_linear",
                              **sched_init_args, )
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        scheduler.config.clip_sample = True
    # ------------------------------------------------------------------------------------------------------------
    attention_storer = AttentionStore()
    register_attention_control(unet, attention_storer)
    # ------------------------------------------------------------------------------------------------------------
    pipeline = StableDiffusionLongPromptWeightingPipeline(text_encoder=text_encoder, vae=vae, unet=unet,
                                                          tokenizer=tokenizer, scheduler=scheduler,
                                                          clip_skip=args.clip_skip,
                                                          safety_checker=None,
                                                          feature_extractor=None,
                                                          requires_safety_checker=False, )
    pipeline.to(accelerator.device)
    from typing import Union, Optional, Callable
    import PIL
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    height: int = args.height
    width: int = args.width
    num_inference_steps: int = args.sample_steps
    guidance_scale: float = args.scale
    strength = 0.8
    num_images_per_prompt = 1
    eta: float = 0.0
    latents: Optional[torch.FloatTensor] = None,
    max_embeddings_multiples: Optional[int] = 3
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    is_cancelled_callback: Optional[Callable[[], bool]] = None,
    callback_steps: int = 1

    with torch.no_grad():
        generator = torch.Generator(device='cuda')
        generator.manual_seed(args.seed)

        # 0. Default height and width to unet
        height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(prompt, height, width, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = pipeline._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt

        def _encode_prompt(self,
                           prompt,
                           device,
                           num_images_per_prompt,
                           do_classifier_free_guidance,
                           negative_prompt,
                           max_embeddings_multiples, ):

            batch_size = len(prompt) if isinstance(prompt, list) else 1

            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=self.clip_skip,
            )
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
            text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

            if do_classifier_free_guidance:
                bs_embed, seq_len, _ = uncond_embeddings.shape
                uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
                uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            return text_embeddings

        text_embeddings = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
        )
        dtype = text_embeddings.dtype

        # 4. Preprocess zaradress and mask
        image = None
        if image is not None:
            print(f'image : {image} | type(image) : {type(image)}')
            image = image.to(device=pipeline.device, dtype=dtype)
        mask = None

        # 5. set timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = pipeline.get_timesteps(num_inference_steps, strength, device,
                                                                image is None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        def prepare_latents(image, timestep, batch_size, height, width,
                            dtype, device, generator, latents):
            if image is None:
                shape = (batch_size,
                         unet.in_channels,
                         height // pipeline.vae_scale_factor,
                         width // pipeline.vae_scale_factor,)
                if latents is None:
                    if device.type == "mps":
                        # randn does not work reproducibly on mps
                        latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
                    else:
                        latents = torch.randn(shape,
                                              generator=generator,
                                              device=device, dtype=dtype)
                else:
                    latents = torch.randn(shape,
                                          generator=generator,
                                          device=device, dtype=dtype)
                    if latents.shape != shape:
                        raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                    latents = latents.to(device)

                # scale the initial noise by the standard deviation required by the scheduler
                latents = latents * pipeline.scheduler.init_noise_sigma
                return latents, None, None
            else:
                init_latent_dist = pipeline.vae.encode(image).latent_dist
                init_latents = init_latent_dist.sample(generator=generator)
                init_latents = 0.18215 * init_latents
                init_latents = torch.cat([init_latents] * batch_size, dim=0)
                init_latents_orig = init_latents
                shape = init_latents.shape

                # add noise to latents using the timesteps
                if device.type == "mps":
                    noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
                else:
                    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                latents = pipeline.scheduler.add_noise(init_latents, noise, timestep)
                return latents, init_latents_orig, noise

        latents, init_latents_orig, noise = prepare_latents(
            image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            latents, )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            if mask is not None:
                init_latents_proper = pipeline.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))
            # call the callback, if provided

            # 9. Post-processing
            image = pipeline.decode_latents(latents)
            # 10. Run safety checker
            image, has_nsfw_concept = pipeline.run_safety_checker(image, device, text_embeddings.dtype)

            # 11. Convert to PIL
            image = pipeline.numpy_to_pil(image)[0]
            p = prompt.split(' ')
            p = '_'.join(p)
            image.save(os.path.join(experiment_base_folder, f'prompt_{p}_timestep_{int(t)}.png'))
    del network_sy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. session id
    parser.add_argument("--cache_latents", action="store_true", )
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--save_precision", type=str, default="fp16")
    parser.add_argument("--device", default='cuda:2', type=str)
    parser.add_argument("--seed", default=1021623518, type=int)

    # step 2. preparing accelerator
    parser.add_argument("--logging_dir", type=str, default='logging', )
    parser.add_argument("--log_with", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="add prefix for each log directory")
    parser.add_argument("--gradient_accumulation_steps", default=1)

    # step 3. original stable diffusion model
    parser.add_argument("--tokenizer_version", type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0")
    parser.add_argument("--v_parameterization", action="store_true", )
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=r'pretrained/AnythingV5Ink_v5PrtRE.safetensors')
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None, )
    parser.add_argument("--mem_eff_attn", action="store_true", )
    parser.add_argument("--xformers", action="store_true", )
    parser.add_argument("--vae", type=str, default=None)

    # step 4. lora state dict
    parser.add_argument("--lora_file_dir", type=str,
                        default=r'pretrained_lora_models/target/izumi_reina_v1.safetensors')

    parser.add_argument("--network_module", type=str, default='networks.lora')
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value)")
    parser.add_argument("--network_dim", type=int, default=2, help="network dimensions")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) ", )
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) ", )
    parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder")
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) ", )

    parser.add_argument("--full_fp16", action="store_true", )
    parser.add_argument("--sample_sampler", default='ddim')
    parser.add_argument("--clip_skip", default=1, type=int)
    parser.add_argument("--output_dir", default='test', type=str)
    parser.add_argument("--lowram", action='store_true',
                        help="load stable diffusion checkpoint weights to VRAM instead of RAM")
    parser.add_argument("--base_weights_multiplier", type=float, default=None,
                        nargs="*", help="multiplier for network weights to merge into the model before training", )

    # step 7. inference
    parser.add_argument("--prompt", default='reina, standing, on a grass', type=str)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--negative_prompt", type=str,
                        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', )
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--scale", type=int, default=7)
    args = parser.parse_args()
    main(args)





