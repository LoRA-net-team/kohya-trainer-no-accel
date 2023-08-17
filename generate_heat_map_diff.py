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
from attention_store import AttentionStore
from transformers import CLIPTokenizer
from networks.lora_block_weighing import PretrainedLoRANetwork
from attention_store import AttentionStore


def register_attention_control(unet, controller):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, context=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            batch_size, sequence_length, _ = hidden_states.shape
            query = self.to_q(hidden_states)
            # query = controller.forward(query, is_cross_attention, place_in_unet)
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


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    torch_img = 2.0 * image - 1.0
    return torch_img


@torch.no_grad()
def image2latent(image, vae, device, weight_dtype):
    with torch.no_grad():
        image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)
        latents = vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
    return latents


def train(args):
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
    experiment_base_folder = os.path.join('result', f'attentionmap_test_{unique_time}')
    print(f' - experiment_base_folder : {experiment_base_folder}')
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
    print(f' (3.3) network')
    lora_weight = load_file(args.lora_file_dir)
    lora_block_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    network_sy = PretrainedLoRANetwork(text_encoder=text_encoder,
                                       unet=unet,
                                       pretrained_lora_weight=lora_weight,
                                       lora_block_weights=lora_block_weight,
                                       multiplier=1.0, )
    print(f' (3.4) change original forward')
    network_sy.apply_to(text_encoder, unet, apply_text_encoder=True, apply_unet=True)
    print(f' (3.4) network and original model to device and state')
    network_sy.requires_grad_(False)
    network_sy.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 4. attention storer')
    attention_storer = AttentionStore()
    register_attention_control(unet, attention_storer)

    print(f'\n step 7. noising prediction')

    def generate_text_embedding(prompt):
        text_input = tokenizer([prompt],
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt", )
        text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
        return text_embeddings

    text_embeddings_src = generate_text_embedding('reina')
    text_embeddings_trg = generate_text_embedding('girl')

    print(f' (7.3) time step')
    t = 0

    print(f'\n step 5. call image')
    # 1) latent 1
    trg_img_folder = 'result/target'
    images = os.listdir(trg_img_folder)
    for image in images:
        name, ext = os.path.splitext(image)
        img_dir = os.path.join(trg_img_folder, image)
        pil_img_1 = Image.open(img_dir)
        latent_1 = image2latent(pil_img_1, vae, accelerator.device, weight_dtype)
        noise_pred = unet(latent_1, t, encoder_hidden_states=text_embeddings_src)["sample"]
        attention_maps_src = attention_storer.step_store
        cross_maps = []
        res1 = int(math.sqrt(256))
        res2 = int(math.sqrt(1024))
        res3 = int(math.sqrt(4096))
        map1_list, map2_list, map3_list = [], [], []
        for location in attention_maps_src.keys():
            attention_map_src_list = attention_maps_src[location]
            for attn_map_src in attention_map_src_list:
                maps = []
                h = w = int(math.sqrt(attn_map_src.size(1)))
                attention_probs_src = attn_map_src.permute(2, 0, 1)  # [sen_len, 8, pix_len]
                for map_src in attention_probs_src:  # total 77 number of
                    map_src = map_src.view(map_src.size(0), h, w)  # [8, h, w]
                    # map_trg = map_trg.view(map_trg.size(0), h, w)
                    maps.append(map_src)
                    # maps.append(map_trg)
                maps = torch.stack(maps, 0)  # [77, 8, h, w ]
                trg_maps = maps[1, :, :, :]  # [8, h, w]
                if trg_maps.size(1) == res1:
                    map1_list.append(trg_maps)

        def list2tensor(map_list):
            # list = [ (8,h,w), ... ]
            # out = [8*num=40, h, w]
            out = torch.cat(map_list, dim=0)
            out = out.sum(0) / out.shape[0]
            #boolen_map = (out > original_max_score) * 1
            #out = out * boolen_map
            # print(f'{name} : {out.max()}')
            out = 255 * out / out.max()
            #out = (out > 0) * 255
            out = out.to(torch.uint8)
            return out.cpu()

        def reshape_map(map):
            # map = [res,res]
            map_ = map.unsqueeze(-1)
            map_ = map_.expand(*map.size(), 3)
            return map_

        def tensor2img(map) -> Image:
            image = map.numpy().astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            return image

        im = list2tensor(map1_list)
        im = im.unsqueeze(0).unsqueeze(0)
        import torch.nn.functional as F
        im = F.interpolate(im.float().detach(), size=(512, 512), mode='bicubic')
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        im = im.cpu().detach().squeeze()
        plt.clf()

        plt.axis('off')
        # plt.imshow(im, 'jet')
        # plt.savefig(os.path.join(experiment_base_folder, f'{name}_binary_attn_map.png'))
        import cv2
        image_pil = Image.open(img_dir)
        image = np.array(image_pil)
        plt.imshow(image, alpha=0.5)
        plt.imshow(im, 'jet', alpha=0.5)
        plt.savefig(os.path.join(experiment_base_folder, f'{name}_binary_attn_map.png'))
        # heatmap_img = im
        # heatmap_img = cv2.imread(os.path.join(experiment_base_folder, f'{name}_binary_attn_map.png'), cv2.IMREAD_COLOR)

        # print(f'heatmap_img : {heatmap_img.shape} / base_img : {base_img.shape} ')
        # super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, base_img, 0.5, 0)
        # cv2.imwrite(os.path.join(experiment_base_folder, f'{name}_supervised_binary_attn_map.png'), super_imposed_img)
        attention_storer.reset()
    """
    res1 = int(math.sqrt(256))
    res2 = int(math.sqrt(1024))
    res3 = int(math.sqrt(4096))
    map1_list, map2_list, map3_list = [],[],[]
    for map in cross_maps :
        if map.shape[0] == int(res1**2) :
            map1 = torch.reshape(map, (res1, res1))
            map1_list.append(map1)
        elif map.shape[0] == int(res2**2) :
            map2 = torch.reshape(map, (res2, res2))
            map2_list.append(map2)
        elif map.shape[0] == res3**2 :
            map3 = torch.reshape(map, (res3, res3))
            map3_list.append(map3)

    """


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
    parser.add_argument("--mem_eff_attn", action="store_true", )
    parser.add_argument("--xformers", action="store_true", )
    parser.add_argument("--v2",
                        action="store_true",
                        help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0")
    parser.add_argument("--v_parameterization", action="store_true", )
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=r'pretrained/AnythingV5Ink_v5PrtRE.safetensors')
    parser.add_argument("--lowram", action='store_true',
                        help="load stable diffusion checkpoint weights to VRAM instead of RAM")
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--lora_file_dir", type=str,
                        default=r'pretrained_lora_models/target/izumi_reina_v1.safetensors')

    # step 5. call zaradress
    parser.add_argument("--trg_img_dir_1", default=r'result/full.png', type=str)
    parser.add_argument("--compare_img_folder", default=r'result/target', type=str)
    # step 6.
    parser.add_argument("--prompt", default='eye', type=str)
    args = parser.parse_args()
    train(args)

