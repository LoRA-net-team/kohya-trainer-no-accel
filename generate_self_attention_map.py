import importlib
import os
from library import train_util
from datetime import datetime
from safetensors.torch import load_file
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
import argparse
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
            query = controller.forward(query, is_cross_attention, place_in_unet)


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
            #if is_cross_attention:
            #    attn = controller.forward(attention_probs, is_cross_attention, place_in_unet)
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
    lora_block_weight = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
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
    prompt = args.prompt
    text_input = tokenizer([prompt],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt", )
    print(f'text_input : {text_input}')
    text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]

    print(f' (7.3) time step')
    t = 0

    print(f'\n step 5. call zaradress')
    # 1) latent 1
    pil_img_1 = Image.open(args.trg_img_dir_1)
    latent_1 = image2latent(pil_img_1, vae, accelerator.device, weight_dtype)
    noise_pred = unet(latent_1, t, encoder_hidden_states=text_embeddings)["sample"]
    query_collecting_dict_1 = attention_storer.step_store

    attention_storer.reset()

    images = os.listdir(args.compare_img_folder)
    for image in images :
        name, ext = os.path.splitext(image)
        if name == 'prompt_hmreina_lora_weight_00000000001000000' :
            image_dir = os.path.join(args.compare_img_folder, image)
            pil_img_2 = Image.open(image_dir)
            # 2) latent 2
            latent_2 = image2latent(pil_img_2, vae, accelerator.device, weight_dtype)
            noise_pred = unet(latent_2, t, encoder_hidden_states=text_embeddings)["sample"]
            query_collecting_dict_2 = attention_storer.step_store
            attentions_256, attentions_1024, attentions_4096 = [],[],[]
            for k in query_collecting_dict_1.keys() :
                queries = query_collecting_dict_1[k]
                for i, query in enumerate(queries) :
                    batch, pix_len, dim = query.shape

                    if pix_len == 256 :
                        compare_query = query_collecting_dict_2[k][i]
                        attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], compare_query.shape[1], dtype=query.dtype, device=query.device),
                                                         query, compare_query.transpose(-1, -2),
                                                         beta=0)
                        attention_probs = attention_scores.softmax(dim=-1)
                        attention_probs = attention_probs.to(query.dtype)
                        attentions_256.append(attention_probs.squeeze())

                    elif pix_len == 1024 : # 1024
                        compare_query = query_collecting_dict_2[k][i]
                        attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], compare_query.shape[1], dtype=query.dtype, device=query.device),
                                                         query, compare_query.transpose(-1, -2),
                                                         beta=0)
                        attention_probs = attention_scores.softmax(dim=-1)
                        attention_probs = attention_probs.to(query.dtype)
                        attentions_1024.append(attention_probs.squeeze())

                    elif pix_len == 4096 :
                        compare_query = query_collecting_dict_2[k][i]
                        attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], compare_query.shape[1], dtype=query.dtype, device=query.device),
                                                         query, compare_query.transpose(-1, -2),
                                                         beta=0)
                        attention_probs = attention_scores.softmax(dim=-1)
                        attention_probs = attention_probs.to(query.dtype)
                        attentions_4096.append(attention_probs.squeeze())


            maps = torch.stack(attentions_256, 0)  # shape: (10, 256,256)
            out = maps.sum(0) / maps.shape[0]  # [256,256]
            num = 0
            for i in out :
                i = i.view(16,16)
                image = 255 * i / i.max()
                image = image.to(torch.uint8)
                image_ = image.unsqueeze(-1)
                image = image_.expand(*image.shape, 3)
                image = image.cpu().detach()
                image = image.numpy().astype(np.uint8)
                image = Image.fromarray(image).resize((512, 512))
                image.save(os.path.join(experiment_base_folder, f'{name}_{num}.png'))
                num += 1

            #print(maps.shape)
            """
            
            def attentions2img (attentions) :
                maps = torch.stack(attentions, 0)  # shape: (tokens, heads, height, width)
                out = maps.sum(0) / maps.shape[0]  # [256,256]
                zaradress = 255 * out / out.max()
                zaradress = zaradress.to(torch.uint8)
                image_ = zaradress.unsqueeze(-1)
                zaradress = image_.expand(*zaradress.shape, 3)
                zaradress = zaradress.cpu().detach()
                zaradress = zaradress.numpy().astype(np.uint8)
                zaradress = Image.fromarray(zaradress).resize((512, 512))
                return zaradress
            
            image_256 = attentions2img(attentions_256)
            image_1024 = attentions2img(attentions_1024)
            image_4096 = attentions2img(attentions_4096)
            image_256.save(os.path.join(experiment_base_folder, f'{name}_attention_map_256.png'))
            image_1024.save(os.path.join(experiment_base_folder, f'{name}_attention_map_1024.png'))
            image_4096.save(os.path.join(experiment_base_folder, f'{name}_attention_map_4096.png'))
            attention_storer.reset()
            """
    #input_latent = torch.cat((latent_1,latent_2), dim=0)
    #input_text = torch.cat((text_embeddings,text_embeddings), dim=0)
    #


    """
    for trg_img in os.listdir(args.trg_img_folder) :
        name, ext = os.path.splitext(trg_img)
        trg_img_dir = os.path.join(args.trg_img_folder, trg_img)
        pil_img = Image.open(trg_img_dir)
        latent = image2latent(pil_img, vae, accelerator.device, weight_dtype)

        print(f' (7.4) noise pred')
        noise_pred = unet(latent, t, encoder_hidden_states=text_embeddings)["sample"]

        print(f'\n step 7. check attention dictionary')
        attention_collect_dict = attention_storer.step_store

        print(f'\n step 8. see attention collect dictionary')
        maps_64 = []
        for k, v in attention_collect_dict.items() :
            # "down_cross": [], "mid_cross": [], "up_cross": [],
            # "down_self": [],  "mid_self": [],  "up_self": []}
            for attention_probs in v :
                factor = int(math.sqrt(64 * 64 // attention_probs.shape[1]))
                h = w = int(math.sqrt(attention_probs.size(1)))
                maps = []
                attention_probs_ = attention_probs.permute(2, 0, 1)  # sen_len, head_num, pix_len
                for map_ in attention_probs_:  # total 77 number of
                    map_ = map_.view(map_.size(0), h, w)
                    maps.append(map_)
                maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
                maps_firsttoken = maps[1,:,:,:]
                if maps_firsttoken.shape[1] == 16 :
                    maps_64.append(maps_firsttoken)
        out = torch.cat(maps_64, dim=0)
        out = out.sum(0) / out.shape[0]
        print(f'out (64,64) : {out.shape}')
        zaradress = 255 * out / out.max()
        image_ = zaradress.unsqueeze(-1)
        print(f'zaradress : {zaradress.shape}')
        zaradress = image_.expand(*zaradress.shape, 3)
        zaradress = zaradress.cpu().detach()
        zaradress = zaradress.numpy().astype(np.uint8)
        zaradress = Image.fromarray(zaradress).resize((512,512))
        img_save_dir = os.path.join(experiment_base_folder, f'{name}_attentionmap.png')
        zaradress.save(img_save_dir)

        attention_storer.reset()
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
    parser.add_argument("--trg_img_dir_1",default=r'result/full.png',type=str)
    parser.add_argument("--compare_img_folder", default=r'result/target', type=str)
    # step 6.
    parser.add_argument("--prompt", default='reina', type=str)
    args = parser.parse_args()
    train(args)





