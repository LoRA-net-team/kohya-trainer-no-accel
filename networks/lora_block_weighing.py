import os
from typing import List, Tuple, Union
import torch
from networks.lora import get_block_index

BLOCKS = ["text_model",
          "unet_down_blocks_0_attentions_0",
          "unet_down_blocks_0_attentions_1",
          "unet_down_blocks_1_attentions_0",
          "unet_down_blocks_1_attentions_1",
          "unet_down_blocks_2_attentions_0",
          "unet_down_blocks_2_attentions_1",
          "unet_mid_block_attentions_0",
          "unet_up_blocks_1_attentions_0",
          "unet_up_blocks_1_attentions_1",
          "unet_up_blocks_1_attentions_2",
          "unet_up_blocks_2_attentions_0",
          "unet_up_blocks_2_attentions_1",
          "unet_up_blocks_2_attentions_2",
          "unet_up_blocks_3_attentions_0",
          "unet_up_blocks_3_attentions_1",
          "unet_up_blocks_3_attentions_2", ]

class PretrainedLoRAModule(torch.nn.Module):

    def __init__(self, lora_name,
                 org_module: torch.nn.Module,
                 multiplier=1.0, lora_dim=4, alpha=1,
                 pretrained_lora_weight=None,
                 lora_block_weights=None):
        super().__init__()
        self.lora_name = lora_name
        ratio = 1
        if lora_block_weights != None :
            for i, block in enumerate(BLOCKS):
                if block in self.lora_name :
                    ratio = int(lora_block_weights[i])


        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える
        self.lora_down.weight.data = pretrained_lora_weight[f'{lora_name}.lora_down.weight']
        self.lora_up.weight.data = pretrained_lora_weight[f'{lora_name}.lora_up.weight']
        self.multiplier = multiplier * ratio
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def merge_to(self, sd, dtype, device):
        # get up/down weight
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)

        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + self.multiplier * conved * self.scale

        # set weight to org_module
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def set_region(self, region):
        self.region = region
        self.region_mask = None
    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


class PretrainedLoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    def __init__(self,
                 text_encoder,
                 unet,
                 pretrained_lora_weight,
                 lora_block_weights,
                 multiplier=1.0,) -> None:
        super().__init__()
        self.multiplier = multiplier

        # create module instances
        def create_modules(is_unet, root_module: torch.nn.Module, target_replace_modules) -> List[PretrainedLoRAModule]:
            prefix = PretrainedLoRANetwork.LORA_PREFIX_UNET if is_unet else PretrainedLoRANetwork.LORA_PREFIX_TEXT_ENCODER
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d or is_conv2d_1x1:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            for i, block in enumerate(BLOCKS):
                                if block in lora_name:
                                    ratio = int(lora_block_weights[i])
                            if ratio == 1 :

                                module_alpha = pretrained_lora_weight[f'{lora_name}.alpha']
                                module_down = pretrained_lora_weight[f'{lora_name}.lora_down.weight']
                                module_up = pretrained_lora_weight[f'{lora_name}.lora_up.weight']
                                network_dim = module_down.size(0)
                                alpha = module_alpha


                                lora = PretrainedLoRAModule(lora_name,
                                                            child_module,
                                                            self.multiplier,
                                                            network_dim,
                                                            alpha,
                                                            pretrained_lora_weight,
                                                            lora_block_weights,)
                                loras.append(lora)
            return loras, skipped
        self.text_encoder_loras, skipped_te = create_modules(False,
                                                             text_encoder,
                                                             PretrainedLoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        self.unet_loras, skipped_un = create_modules(True,
                                                     unet,
                                                     PretrainedLoRANetwork.UNET_TARGET_REPLACE_MODULE)
    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(PretrainedLoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(PretrainedLoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        print(f"weights are merged")

    # 層別学習率用に層ごとの学習率に対する倍率を定義する
    def set_block_lr_weight(
        self,
        up_lr_weight: List[float] = None,
        mid_lr_weight: float = None,
        down_lr_weight: List[float] = None,
    ):
        self.block_lr = True
        self.down_lr_weight = down_lr_weight
        self.mid_lr_weight = mid_lr_weight
        self.up_lr_weight = up_lr_weight
    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from BlockwiseLora.library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    # mask is a tensor with values from 0 to 1
    def set_region(self, sub_prompt_index, is_last_network, mask):
        if mask.max() == 0:
            mask = torch.ones_like(mask)

        self.mask = mask
        self.sub_prompt_index = sub_prompt_index
        self.is_last_network = is_last_network

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.set_network(self)

    def set_current_generation(self, batch_size, num_sub_prompts, width, height, shared):
        self.batch_size = batch_size
        self.num_sub_prompts = num_sub_prompts
        self.current_size = (height, width)
        self.shared = shared

        # create masks
        mask = self.mask
        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w
        ref_weight = self.text_encoder_loras[0].lora_down.weight if self.text_encoder_loras else self.unet_loras[0].lora_down.weight
        dtype = ref_weight.dtype
        device = ref_weight.device

        def resize_add(mh, mw):
            # print(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(device, dtype=dtype)
            mask_dic[mh * mw] = m

        h = height // 8
        w = width // 8
        for _ in range(4):
            resize_add(h, w)
            if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                resize_add(h + h % 2, w + w % 2)
            h = (h + 1) // 2
            w = (w + 1) // 2

        self.mask_dic = mask_dic
