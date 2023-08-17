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
import torch.nn.functional as F
import torch.nn.functional as F
def generate_text_embedding(prompt, tokenizer, text_encoder, accelerator):
    text_input = tokenizer([prompt],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt", )
    cls_token = 49406
    pad_token = 49407
    trg_indexs = []
    trg_index = 0
    token_ids = text_input.input_ids[0]
    attns = text_input.attention_mask[0]
    for token_id, attn in zip(token_ids, attns):
        if token_id != cls_token and token_id != pad_token and attn == 1:
            trg_indexs.append(trg_index)
        trg_index += 1
    text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
    return text_embeddings, trg_indexs
def list2tensor(map_list, threshold):
    out = torch.cat(map_list, dim=0)
    out = out.sum(0) / out.shape[0]
    out = torch.where(out > threshold, 1, 0)
    out = 255 * out / out.max()
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
def register_attention_control(unet, controller):
    def ca_forward(self, place_in_unet, layer_name):
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
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1],
                                                         dtype=query.dtype, device=query.device),
                                             query,
                                             key.transpose(-1, -2),
                                             beta=0, alpha=self.scale,)
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if is_cross_attention:
                attn = controller.forward(attention_probs,
                                          is_cross_attention,
                                          place_in_unet,
                                          layer_name)
            # ----------------------------------------------------------------------------------------------------------------------------------
            # 2) after value calculating
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, place_in_unet, parent_name):
        if net_.__class__.__name__ == 'CrossAttention':
            layer_name = parent_name
            net_.forward = ca_forward(net_, place_in_unet, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                parent_name = f'{parent_name}.{name__}'
                count = register_recr(net__, count, place_in_unet, parent_name)
        return count

    for name, module in unet.named_modules() :
        if "down" in name and module.__class__.__name__ == 'CrossAttention' :
            module.forward = ca_forward(module, 'down', name)
        if "mid" in name and module.__class__.__name__ == 'CrossAttention' :
            module.forward = ca_forward(module, 'mid', name)
        if "up" in name and module.__class__.__name__ == 'CrossAttention' :
            module.forward = ca_forward(module, 'mid', name)

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

def draw_attention_score_on_image(trg_list,
                                  img_dir,
                                  thresholds, save_folder,
                                  src_name, layer_name):
    if len(trg_list) > 0:
        for threshold in thresholds:
            im = list2tensor(trg_list, threshold)
            im = im.unsqueeze(0).unsqueeze(0)
            im = F.interpolate(im.float().detach(), size=(512, 512), mode='bicubic')
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
            im = im.cpu().detach().squeeze()
            plt.clf()
            plt.axis('off')
            image_pil = Image.open(img_dir)
            image = np.array(image_pil)
            plt.imshow(image, alpha=0.5)
            plt.imshow(im, 'jet', alpha=0.5)
            save_dir = os.path.join(save_folder,
                                    f'{src_name}_{layer_name}_thredshold_{threshold}_binary_attn_map.png')
            plt.savefig(save_dir)
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
    experiment_base_folder = os.path.join('result', f'attentionmap_test_{unique_time}')
    print(f' result will be saved on : {experiment_base_folder}')
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
    lora_block_weight = args.lora_block_weight
    print(f'lora_block_weight : {lora_block_weight}')
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
    """
    print(f'\n step 5. key collecting')
    parameter_dict = {}
    for name, module in unet.named_modules() :
        if "down" in name and 'attn2' in name and module.__class__.__name__ == 'CrossAttention' :
            for parameter_name, param in module.named_parameters() :
                if 'to_k' in parameter_name :
                    full_name = f'{name}.{parameter_name}'
                    parameter_dict[full_name] = param
        if "mid" in name and 'attn2' in name and module.__class__.__name__ == 'CrossAttention' :
            for parameter_name, param in module.named_parameters() :
                if 'to_k' in parameter_name :
                    full_name = f'{name}.{parameter_name}'
                    parameter_dict[full_name] = param
        if "up" in name and 'attn2' in name and module.__class__.__name__ == 'CrossAttention' :
            for parameter_name, param in module.named_parameters() :
                if 'to_k' in parameter_name :
                    full_name = f'{name}.{parameter_name}'
                    parameter_dict[full_name] = param

    new_parameter_dict = {}
    for k, param in parameter_dict.items() :
        try :
            new_key = str(param.shape)
            new_parameter_dict[new_key].append(param)
        except :
            new_key = str(param.shape)
            new_parameter_dict[new_key] = []
            new_parameter_dict[new_key].append(param)

    test_parameter_dict = {}
    shape_index = {}
    for layer_name, param in parameter_dict.items() :
        trg_shape = param.shape
        for k, param_list in new_parameter_dict.items() :
            if str(trg_shape) == k :
                try :
                    shape_index[str(trg_shape)] += 1
                except :
                    shape_index[str(trg_shape)] = 0
                i = shape_index[str(trg_shape)] + 1
                i = (i)%len(param_list)
                test_parameter_dict[layer_name] = param_list[i]


    print(f'\n step 6. key value change')
    for name, module in unet.named_modules() :
        if "down" in name and'attn2' in name and module.__class__.__name__ == 'CrossAttention' :
            for param_name, param in module.named_parameters() :
                full_name = f'{name}.{param_name}'
                if 'to_k' in full_name :
                    module.to_k.weight = test_parameter_dict[full_name]
        if "up" in name and'attn2' in name and module.__class__.__name__ == 'CrossAttention' :
            for param_name, param in module.named_parameters() :
                full_name = f'{name}.{param_name}'
                if 'to_k' in full_name :
                    module.to_k.weight = test_parameter_dict[full_name]
        if "mid" in name and'attn2' in name and module.__class__.__name__ == 'CrossAttention' :
            for param_name, param in module.named_parameters() :
                full_name = f'{name}.{param_name}'
                if 'to_k' in full_name :
                    module.to_k.weight = test_parameter_dict[full_name]
    """
    print(f'\n step 5. generate cross attention map')
    t = 0
    text_embeddings_src, trg_indexs_src = generate_text_embedding(args.target_token,
                                                                  tokenizer,
                                                                  text_encoder,
                                                                  accelerator)
    trg_img_folder = args.trg_img_folder
    images = os.listdir(trg_img_folder)
    for image in images:
        name, ext = os.path.splitext(image)
        if 'A_sailormoon,_wearing_a_headphone' in name :
            img_dir = os.path.join(trg_img_folder, image)
            pil_img_1 = Image.open(img_dir)
            latent_1 = image2latent(pil_img_1, vae, accelerator.device, weight_dtype)
            noise_pred = unet(latent_1, t, encoder_hidden_states=text_embeddings_src)["sample"]
            attention_maps_src = attention_storer.step_store
            for layer_name in attention_maps_src.keys():
                map_list = []
                attention_map_src_list = attention_maps_src[layer_name]
                for attn_map_src in attention_map_src_list:
                    maps = []
                    h = w = int(math.sqrt(attn_map_src.size(1)))
                    attention_probs_src = attn_map_src.permute(2, 0, 1)  # [sen_len, 8, pix_len]
                    for map_src in attention_probs_src:  # total 77 number of
                        map_src = map_src.view(map_src.size(0), h, w)  # [8, h, w]
                        maps.append(map_src)
                    maps = torch.stack(maps, 0)  # [77, 8, h, w ]
                    for index in trg_indexs_src :
                        trg_maps = maps[index, :, :, :]  # [8, h, w]
                        map_list.append(trg_maps)
                draw_attention_score_on_image(map_list,
                                              img_dir=img_dir,
                                              thresholds=[args.thredshold],
                                              save_folder=experiment_base_folder,
                                              src_name=name,
                                              layer_name=layer_name)
            attention_storer.reset()

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
    import ast
    def arg_as_list(s):
        v = ast.literal_eval(s)
        return v
    parser.add_argument("--lora_block_weight", type = arg_as_list)

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
                        default=r'result/fineblock_lora_block_sailormoon_2023-08-10_1691662940_7337515/block_wise_11111000000111111/model/test_sailormoon-000009.safetensors')
    # step 5. generate cross attention map
    parser.add_argument("--target_token", default='headphone', type=str)
    parser.add_argument("--trg_img_folder", default='result/fineblock_lora_block_sailormoon_2023-08-10_1691662940_7337515/block_wise_11111000000111111/sample/epoch10', type=str)
    parser.add_argument("--thredshold", default=0.05, type=float)
    args = parser.parse_args()
    main(args)