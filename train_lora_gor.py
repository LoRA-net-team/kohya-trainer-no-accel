import wandb
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
from typing import Union, Optional, Callable
import PIL


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
            if context is not None :
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


def prepare_latents(image, timestep, batch_size, height, width,
                    dtype, device, generator, latents, pipeline):
    if image is None:
        shape = (batch_size,
                 pipeline.unet.in_channels,
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
    experiment_base_folder = os.path.join('result', f'test_{unique_time}')
    os.makedirs(experiment_base_folder, exist_ok=True)
    args.logging_dir = os.path.join(experiment_base_folder, args.logging_dir)
    accelerator, unwrap_model = train_util.prepare_accelerator(args)

    if accelerator.is_main_process:
        accelerator.init_trackers("weight-heatmap", config=vars(args))


    print(f'\n step 3. original stable diffusion model')
    print(f' (3.1) tokenizer')
    version = args.tokenizer_version
    tokenizer = CLIPTokenizer.from_pretrained(version)

    print(f'\n step 4. lora state dict')
    lora_weight = load_file(args.lora_file_dir)

    print(f'\n step 5. blockwise')
    lora_block_weights = [args.target_block_weight]
    for lora_block_weight in lora_block_weights :
        print(f' (5.1) text encoder, vae, unet')
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)
        network_sy = PretrainedLoRANetwork(text_encoder=text_encoder,
                                           unet=unet,
                                           pretrained_lora_weight=lora_weight,
                                           lora_block_weights = lora_block_weight,
                                           multiplier=1.0,)
        print(f' (5.2) change original forward')
        network_sy.apply_to(text_encoder, unet,
                            apply_text_encoder=True, apply_unet=True)
        print(f' (5.3) network and original model to device and state')
        network_sy.requires_grad_(False)
        network_sy.to(accelerator.device, dtype=weight_dtype)
        unet.requires_grad_(False)
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=weight_dtype)
        print(f' (5.4) inference scheduler')
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

        def cov(tensor, rowvar=True, bias=False):
            """Estimate a covariance matrix (np.cov)"""
            tensor = tensor if rowvar else tensor.transpose(-1, -2)
            tensor = tensor - tensor.mean(dim=-1, keepdim=True)
            factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
            return factor * tensor @ tensor.transpose(-1, -2).conj()

        def corrcoef(tensor, rowvar=True):
            """Get Pearson product-moment correlation coefficients (np.corrcoef)"""
            covariance = cov(tensor, rowvar=rowvar)
            variance = covariance.diagonal(0, -1, -2)
            if variance.is_complex():
                variance = variance.real
            stddev = variance.sqrt()
            covariance /= stddev.unsqueeze(-1)
            covariance /= stddev.unsqueeze(-2)
            if covariance.is_complex():
                covariance.real.clip_(-1, 1)
                covariance.imag.clip_(-1, 1)
            else:
                covariance.clip_(-1, 1)
            return covariance


        print(f' (5.5) lora weights')
        lora_blocks = network_sy.unet_loras + network_sy.text_encoder_loras
        for lora_block in lora_blocks :
            lora_name = lora_block.lora_name
            lora_down = lora_block.lora_down
            lora_up = lora_block.lora_up
            # Only Linear Layer ...
            if lora_down.weight.ndim == 2 :

                lora_up_cor = torch.corrcoef(lora_up.weight)
                #lora_weight = lora_up.weight @ lora_down.weight
                #covariance = cov(lora_weight, rowvar=True)

                import seaborn as sns

                #print(f'lora_down.weight : {lora_down.weight.shape} | lora_up.weight : {lora_up.weight.shape}')
                """
                
                variance = covariance.diagonal(0, -1, -2)
                if variance.is_complex():
                    variance = variance.real
                stddev = variance.sqrt()
                
                covariance /= stddev.unsqueeze(-1)
                covariance /= stddev.unsqueeze(-2)
                if covariance.is_complex():
                    covariance.real.clip_(-1, 1)
                    covariance.imag.clip_(-1, 1)
                else:
                    covariance.clip_(-1, 1)
                #weight_cor = torch.corrcoef(lora_weight)
                """
                print(f'lora_name : {lora_name} | lora_up_cor : {lora_up_cor.shape}')
                cor_img = sns.heatmap(lora_up_cor.cpu(),
                                      cmap='BuPu',
                                      annot=False,
                                      yticklabels=False,
                                      xticklabels=False,
                                      cbar=False,
                                      vmin=-1, vmax=1)
                figure = cor_img.get_figure()
                figure.savefig(f'{lora_name}.jpg', dpi=400)
                for tracker in accelerator.trackers:
                    tracker.log({"validation": [wandb.Image(figure, caption=f"{lora_name}")]})





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. session id
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--save_precision", type=str, default="fp16")

    # step 2. preparing accelerator
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="specify WandB API key to log in before starting training (optional).", )
    parser.add_argument("--logging_dir", type=str, default='logging', )
    parser.add_argument("--log_with", type=str, default="wandb",
                        choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="add prefix for each log directory")
    parser.add_argument("--gradient_accumulation_steps", default=1)

    # step 3. tokenizer
    parser.add_argument("--tokenizer_version", type=str, default='openai/clip-vit-large-patch14')

    # step 4. lora state dict
    parser.add_argument("--lora_file_dir", type=str,
     default=r'result/4_sailormoon_4_training_image/block_wise_11111111111111111/model/test_sailormoon-000002.safetensors')

    # step 5. blockwise
    import ast
    def arg_as_list(s):
        v = ast.literal_eval(s)
        return v
    parser.add_argument("--target_block_weight", type=arg_as_list,
                        default=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    # (5.1) text encoder, vae, unet
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=r'pretrained/AnythingV5Ink_v5PrtRE.safetensors')
    parser.add_argument("--mem_eff_attn", action="store_true", )
    parser.add_argument("--xformers", action="store_true", )

    # ------------------------------------------------------------------------------------------------------
    parser.add_argument("--lowram", action='store_true',
                        help="load stable diffusion checkpoint weights to VRAM instead of RAM")
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--target_block_weight_use", action="store_false")
    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0")
    parser.add_argument("--v_parameterization", action="store_true", )
    parser.add_argument("--clip_skip", default=1, type=int)
    parser.add_argument("--prompt", default='sailormoon, standing, on a grass', type=str)
    parser.add_argument("--negative_prompt", type=str,
                        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--sample_steps", type=int, default = 20)
    parser.add_argument("--scale", type=int, default=7)
    parser.add_argument("--seed", default=1021623518, type=int)
    # ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    main(args)








