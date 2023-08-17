import math
import os
from typing import List, Tuple
import torch
import re

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

low_lr_blocks = ["unet_down_blocks_0_attentions_0","unet_down_blocks_0_attentions_1",
                 "unet_down_blocks_1_attentions_0", "unet_down_blocks_1_attentions_1",
                 "unet_up_blocks_2_attentions_0","unet_up_blocks_2_attentions_1","unet_up_blocks_2_attentions_2",
                 "unet_up_blocks_3_attentions_0","unet_up_blocks_3_attentions_1","unet_up_blocks_3_attentions_2", ]

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")

class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self, lora_name, org_module: torch.nn.Module, training_option=1.0, lora_dim=4, alpha=1):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()

        self.lora_name = lora_name
        self.training_option = training_option
        if self.training_option :
            if org_module.__class__.__name__ == "Conv2d":
                in_dim = org_module.in_channels
                out_dim = org_module.out_channels
            else:
                in_dim = org_module.in_features
                out_dim = org_module.out_features

            # if limit_rank:
            #   self.lora_dim = min(lora_dim, in_dim, out_dim)
            #   if self.lora_dim != lora_dim:
            #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
            # else:
            self.lora_dim = lora_dim
            """
            if 'unet_down_blocks_1' in lora_name or 'unet_up_blocks_2' in lora_name :
                self.lora_dim = int(self.lora_dim / 2)
            if 'unet_down_blocks_2' in lora_name or 'unet_up_blocks_1' in lora_name :
                self.lora_dim = int(self.lora_dim / 4)
            if 'unet_mid' in lora_name:
                self.lora_dim = int(self.lora_dim / 8)
            print(f'{lora_name} lora_dim: {self.lora_dim}')
            """
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
            # same as microsoft's
            torch.nn.init.zeros_(self.lora_up.weight)
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))


        self.multiplier = 1.0
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
        if self.training_option :
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        else :
            return self.org_forward(x)

def create_network(lora_block_training_options,
                   network_dim,
                   network_alpha,
                   vae,
                   text_encoder,
                   unet, **kwargs):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # block dim/alpha/lr
    block_dims = kwargs.get("block_dims", None)
    down_lr_weight = kwargs.get("down_lr_weight", None)
    mid_lr_weight = kwargs.get("mid_lr_weight", None)
    up_lr_weight = kwargs.get("up_lr_weight", None)

    # 以上のいずれかに指定があればblockごとのdim(rank)を有効にする
    if block_dims is not None or down_lr_weight is not None or mid_lr_weight is not None or up_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha)

        # extract learning rate weight for each block
        if down_lr_weight is not None:
            # if some parameters are not set, use zero
            if "," in down_lr_weight:
                down_lr_weight = [(float(s) if s else 0.0) for s in down_lr_weight.split(",")]

        if mid_lr_weight is not None:
            mid_lr_weight = float(mid_lr_weight)

        if up_lr_weight is not None:
            if "," in up_lr_weight:
                up_lr_weight = [(float(s) if s else 0.0) for s in up_lr_weight.split(",")]

        down_lr_weight, mid_lr_weight, up_lr_weight = get_block_lr_weight(
            down_lr_weight, mid_lr_weight, up_lr_weight, float(kwargs.get("block_lr_zero_threshold", 0.0))
        )

        # remove block dim/alpha without learning rate
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = remove_block_dims_and_alphas(
            block_dims, block_alphas, conv_block_dims, conv_block_alphas, down_lr_weight, mid_lr_weight, up_lr_weight
        )

    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    network = LoRANetwork(text_encoder,
        unet,
        lora_block_training_options=lora_block_training_options,
        lora_dim=network_dim,
        alpha=network_alpha,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        varbose=True,)


    if up_lr_weight is not None or mid_lr_weight is not None or down_lr_weight is not None:
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network


# このメソッドは外部から呼び出される可能性を考慮しておく
# network_dim, network_alpha にはデフォルト値が入っている。
# block_dims, block_alphas は両方ともNoneまたは両方とも値が入っている
# conv_dim, conv_alpha は両方ともNoneまたは両方とも値が入っている
def get_block_dims_and_alphas(
    block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
):
    num_total_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + 1

    def parse_ints(s):
        return [int(i) for i in s.split(",")]

    def parse_floats(s):
        return [float(i) for i in s.split(",")]

    # block_dimsとblock_alphasをパースする。必ず値が入る
    if block_dims is not None:
        block_dims = parse_ints(block_dims)
        assert (
            len(block_dims) == num_total_blocks
        ), f"block_dims must have {num_total_blocks} elements / block_dimsは{num_total_blocks}個指定してください"
    else:
        print(f"block_dims is not specified. all dims are set to {network_dim} / block_dimsが指定されていません。すべてのdimは{network_dim}になります")
        block_dims = [network_dim] * num_total_blocks

    if block_alphas is not None:
        block_alphas = parse_floats(block_alphas)
        assert (len(block_alphas) == num_total_blocks),\
                f"block_alphas must have {num_total_blocks} elements / block_alphasは{num_total_blocks}個指定してください"
    else:
        print(
            f"block_alphas is not specified. all alphas are set to {network_alpha} / block_alphasが指定されていません。すべてのalphaは{network_alpha}になります"
        )
        block_alphas = [network_alpha] * num_total_blocks

    # conv_block_dimsとconv_block_alphasを、指定がある場合のみパースする。指定がなければconv_dimとconv_alphaを使う
    if conv_block_dims is not None:
        conv_block_dims = parse_ints(conv_block_dims)
        assert (
            len(conv_block_dims) == num_total_blocks
        ), f"conv_block_dims must have {num_total_blocks} elements / conv_block_dimsは{num_total_blocks}個指定してください"

        if conv_block_alphas is not None:
            conv_block_alphas = parse_floats(conv_block_alphas)
            assert (
                len(conv_block_alphas) == num_total_blocks
            ), f"conv_block_alphas must have {num_total_blocks} elements / conv_block_alphasは{num_total_blocks}個指定してください"
        else:
            if conv_alpha is None:
                conv_alpha = 1.0
            print(
                f"conv_block_alphas is not specified. all alphas are set to {conv_alpha} / conv_block_alphasが指定されていません。すべてのalphaは{conv_alpha}になります"
            )
            conv_block_alphas = [conv_alpha] * num_total_blocks
    else:
        if conv_dim is not None:
            print(
                f"conv_dim/alpha for all blocks are set to {conv_dim} and {conv_alpha} / すべてのブロックのconv_dimとalphaは{conv_dim}および{conv_alpha}になります"
            )
            conv_block_dims = [conv_dim] * num_total_blocks
            conv_block_alphas = [conv_alpha] * num_total_blocks
        else:
            conv_block_dims = None
            conv_block_alphas = None

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 層別学習率用に層ごとの学習率に対する倍率を定義する、外部から呼び出される可能性を考慮しておく
def get_block_lr_weight(
    down_lr_weight, mid_lr_weight, up_lr_weight, zero_threshold
) -> Tuple[List[float], List[float], List[float]]:
    # パラメータ未指定時は何もせず、今までと同じ動作とする
    if up_lr_weight is None and mid_lr_weight is None and down_lr_weight is None:
        return None, None, None

    max_len = LoRANetwork.NUM_OF_BLOCKS  # フルモデル相当でのup,downの層の数

    def get_list(name_with_suffix) -> List[float]:
        import math

        tokens = name_with_suffix.split("+")
        name = tokens[0]
        base_lr = float(tokens[1]) if len(tokens) > 1 else 0.0

        if name == "cosine":
            return [math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr for i in reversed(range(max_len))]
        elif name == "sine":
            return [math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr for i in range(max_len)]
        elif name == "linear":
            return [i / (max_len - 1) + base_lr for i in range(max_len)]
        elif name == "reverse_linear":
            return [i / (max_len - 1) + base_lr for i in reversed(range(max_len))]
        elif name == "zeros":
            return [0.0 + base_lr] * max_len
        else:
            print(
                "Unknown lr_weight argument %s is used. Valid arguments:  / 不明なlr_weightの引数 %s が使われました。有効な引数:\n\tcosine, sine, linear, reverse_linear, zeros"
                % (name)
            )
            return None

    if type(down_lr_weight) == str:
        down_lr_weight = get_list(down_lr_weight)
    if type(up_lr_weight) == str:
        up_lr_weight = get_list(up_lr_weight)

    if (up_lr_weight != None and len(up_lr_weight) > max_len) or (down_lr_weight != None and len(down_lr_weight) > max_len):
        print("down_weight or up_weight is too long. Parameters after %d-th are ignored." % max_len)
        print("down_weightもしくはup_weightが長すぎます。%d個目以降のパラメータは無視されます。" % max_len)
        up_lr_weight = up_lr_weight[:max_len]
        down_lr_weight = down_lr_weight[:max_len]

    if (up_lr_weight != None and len(up_lr_weight) < max_len) or (down_lr_weight != None and len(down_lr_weight) < max_len):
        print("down_weight or up_weight is too short. Parameters after %d-th are filled with 1." % max_len)
        print("down_weightもしくはup_weightが短すぎます。%d個目までの不足したパラメータは1で補われます。" % max_len)

        if down_lr_weight != None and len(down_lr_weight) < max_len:
            down_lr_weight = down_lr_weight + [1.0] * (max_len - len(down_lr_weight))
        if up_lr_weight != None and len(up_lr_weight) < max_len:
            up_lr_weight = up_lr_weight + [1.0] * (max_len - len(up_lr_weight))

    if (up_lr_weight != None) or (mid_lr_weight != None) or (down_lr_weight != None):
        print("apply block learning rate / 階層別学習率を適用します。")
        if down_lr_weight != None:
            down_lr_weight = [w if w > zero_threshold else 0 for w in down_lr_weight]
            print("down_lr_weight (shallower -> deeper, 浅い層->深い層):", down_lr_weight)
        else:
            print("down_lr_weight: all 1.0, すべて1.0")

        if mid_lr_weight != None:
            mid_lr_weight = mid_lr_weight if mid_lr_weight > zero_threshold else 0
            print("mid_lr_weight:", mid_lr_weight)
        else:
            print("mid_lr_weight: 1.0")

        if up_lr_weight != None:
            up_lr_weight = [w if w > zero_threshold else 0 for w in up_lr_weight]
            print("up_lr_weight (deeper -> shallower, 深い層->浅い層):", up_lr_weight)
        else:
            print("up_lr_weight: all 1.0, すべて1.0")

    return down_lr_weight, mid_lr_weight, up_lr_weight


# lr_weightが0のblockをblock_dimsから除外する、外部から呼び出す可能性を考慮しておく
def remove_block_dims_and_alphas(
    block_dims, block_alphas, conv_block_dims, conv_block_alphas, down_lr_weight, mid_lr_weight, up_lr_weight
):
    # set 0 to block dim without learning rate to remove the block
    if down_lr_weight != None:
        for i, lr in enumerate(down_lr_weight):
            if lr == 0:
                block_dims[i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[i] = 0
    if mid_lr_weight != None:
        if mid_lr_weight == 0:
            block_dims[LoRANetwork.NUM_OF_BLOCKS] = 0
            if conv_block_dims is not None:
                conv_block_dims[LoRANetwork.NUM_OF_BLOCKS] = 0
    if up_lr_weight != None:
        for i, lr in enumerate(up_lr_weight):
            if lr == 0:
                block_dims[LoRANetwork.NUM_OF_BLOCKS + 1 + i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[LoRANetwork.NUM_OF_BLOCKS + 1 + i] = 0

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 外部から呼び出す可能性を考慮しておく
def get_block_index(lora_name: str) -> int:
    block_idx = -1  # invalid lora name

    m = RE_UPDOWN.search(lora_name)
    if m:
        g = m.groups()
        i = int(g[1])
        j = int(g[3])
        if g[2] == "resnets":
            idx = 3 * i + j
        elif g[2] == "attentions":
            idx = 3 * i + j
        elif g[2] == "upsamplers" or g[2] == "downsamplers":
            idx = 3 * i + 2

        if g[0] == "down":
            block_idx = 1 + idx  # 0に該当するLoRAは存在しない
        elif g[0] == "up":
            block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx

    elif "mid_block_" in lora_name:
        block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12

    return block_idx


class LoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12
    # is it possible to apply conv_in and conv_out? -> yes, newer LoCon supports it (^^;)
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        lora_block_training_options = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        lora_dim=4,
        alpha=1,
        conv_lora_dim=None,
        conv_alpha=None,
        block_dims=None,
        block_alphas=None,
        conv_block_dims=None,
        conv_block_alphas=None,
        modules_dim=None,
        modules_alpha=None,
        module_class=LoRAModule,
        varbose=False,
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        super().__init__()
        self.lora_block_training_options = lora_block_training_options
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        elif block_dims is not None:
            print(f"create LoRA network from block_dims")
            print(f"block_dims: {block_dims}")
            print(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                print(f"conv_block_dims: {conv_block_dims}")
                print(f"conv_block_alphas: {conv_block_alphas}")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            if self.conv_lora_dim is not None:
                print(f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")


        def create_modules(is_unet,
                           root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
            prefix = LoRANetwork.LORA_PREFIX_UNET if is_unet else LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                block_idx = get_block_index(lora_name)
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha = conv_block_alphas[block_idx]

                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha
                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            for i, block in enumerate(BLOCKS):
                                if block in lora_name:
                                    training_option = self.lora_block_training_options[i]
                                    if training_option == 1 :
                                        lora = module_class(lora_name, child_module, training_option, dim, alpha)
                                        loras.append(lora)

            return loras, skipped

        self.text_encoder_loras, skipped_te = create_modules(False, text_encoder,
                                                             LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras, skipped_un = create_modules(True, unet, target_modules)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            print(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weight dim (rank) {len(skipped)}:")
            for name in skipped:
                print(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

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
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
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

    def get_lr_weight(self, lora: LoRAModule) -> float:
        lr_weight = 1.0
        block_idx = get_block_index(lora.lora_name)
        if block_idx < 0:
            return lr_weight

        if block_idx < LoRANetwork.NUM_OF_BLOCKS:
            if self.down_lr_weight != None:
                lr_weight = self.down_lr_weight[block_idx]
        elif block_idx == LoRANetwork.NUM_OF_BLOCKS:
            if self.mid_lr_weight != None:
                lr_weight = self.mid_lr_weight
        elif block_idx > LoRANetwork.NUM_OF_BLOCKS:
            if self.up_lr_weight != None:
                lr_weight = self.up_lr_weight[block_idx - LoRANetwork.NUM_OF_BLOCKS - 1]

        return lr_weight

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr, blockwise_lr):
        self.requires_grad_(True)
        all_params = []
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        def enumerate_unet_params(loras):
            low_lr_params = []
            high_lr_params = []
            for lora in loras:
                if lora.lora_name in  low_lr_blocks :
                    low_lr_params.extend(lora.parameters())
                else :
                    high_lr_params.extend(lora.parameters())
            return low_lr_params, high_lr_params

        if self.text_encoder_loras:
            param_data = {"params" : enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            if self.block_lr  :
                block_idx_to_lora = {}
                for lora in self.unet_loras:
                    idx = get_block_index(lora.lora_name)
                    if idx not in block_idx_to_lora:
                        block_idx_to_lora[idx] = []
                    block_idx_to_lora[idx].append(lora)

                # blockごとにパラメータを設定する
                for idx, block_loras in block_idx_to_lora.items():
                    param_data = {"params": enumerate_params(block_loras)}

                    if unet_lr is not None:
                        param_data["lr"] = unet_lr * self.get_lr_weight(block_loras[0])
                    elif default_lr is not None:
                        param_data["lr"] = default_lr * self.get_lr_weight(block_loras[0])
                    if ("lr" in param_data) and (param_data["lr"] == 0):
                        continue
                    all_params.append(param_data)

            else:
                if blockwise_lr :
                    low_lr_params, high_lr_params = enumerate_unet_params(self.unet_loras)
                    if len(low_lr_params) > 0 :
                        low_lr_param_data = {"params": low_lr_params}
                        if unet_lr is not None:
                            low_lr_param_data["lr"] = unet_lr / 10
                            all_params.append(low_lr_param_data)
                    if len(high_lr_params) > 0 :
                        high_lr_param_data = {"params": high_lr_params}
                        if unet_lr is not None:
                            high_lr_param_data["lr"] = unet_lr
                            all_params.append(high_lr_param_data)
                else :
                    param_data = {"params": enumerate_params(self.unet_loras)}
                    if unet_lr is not None:
                        param_data["lr"] = unet_lr
                    all_params.append(param_data)

        return all_params

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
            from library import train_util

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