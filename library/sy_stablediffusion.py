import inspect
import re
from typing import Callable, List, Optional, Union
import numpy as np
import PIL
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import SchedulerMixin, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.utils import logging
from transformers import CLIPVisionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask, scale_factor=8):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"])
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask
def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights
def parse_prompt_attention(text):
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1
    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier
    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)
        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)
    if len(res) == 0:
        res = [["", 1.0]]
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res
def get_prompts_with_weights(pipe: StableDiffusionPipeline, prompt: List[str], max_length: int):
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights
def get_unweighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,):
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos
            if clip_skip is None or clip_skip == 1:
                text_embedding = pipe.text_encoder(text_input_chunk)[0]
            else:
                enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)
            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    text_embedding = text_embedding[:, 1:-1]
            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = pipe.text_encoder(text_input)[0]
        else:
            enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings

def get_weighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
):
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]
    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = pipe.tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(prompt_tokens,prompt_weights,max_length,bos,
                                                           eos,no_boseos_middle=no_boseos_middle,chunk_length=pipe.tokenizer.model_max_length,)

    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(uncond_tokens,uncond_weights,max_length,
                                                               bos,eos,no_boseos_middle=no_boseos_middle,chunk_length=pipe.tokenizer.model_max_length,)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)
    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None

class MultimodalStableDiffusion(StableDiffusionPipeline):
    # if version.parse(version.parse(diffusers.__version__).base_version) >= version.parse("0.9.0"):
    def __init__(self,
                 vae: AutoencoderKL,
                 img_encoder: CLIPVisionModel,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: SchedulerMixin,
                 clip_skip: int,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPFeatureExtractor,
                 requires_safety_checker: bool = True,):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,)
        self.img_encoder = img_encoder
        self.clip_skip = clip_skip
        self.__init__additional__()

    def __init__additional__(self):
        if not hasattr(self, "vae_scale_factor"):
            setattr(self, "vae_scale_factor", 2 ** (len(self.vae.config.block_out_channels) - 1))

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self,prompt,
                       device,num_images_per_prompt,do_classifier_free_guidance,
                       negative_prompt,max_embeddings_multiples,):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`.")
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

    def check_inputs(self, prompt, height, width, strength, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            print(height, width)
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device, is_text2img):
        if is_text2img:
            return self.scheduler.timesteps.to(device), num_inference_steps
        else:
            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(device)
            return timesteps, num_inference_steps - t_start

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, image, timestep, batch_size, height, width, dtype, device, generator, latents=None):
        if image is None:
            shape = (
                batch_size,
                self.unet.in_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )

            if latents is None:
                if device.type == "mps":
                    # randn does not work reproducibly on mps
                    latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
                else:
                    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            else:
                if latents.shape != shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents, None, None
        else:
            init_latent_dist = self.vae.encode(image).latent_dist
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
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
            return latents, init_latents_orig, noise

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        clip_processored:torch = None, # shape = batch, 3channel, h, w
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # ----------------------------------------------------------------------------------------------------------
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, strength, callback_steps)
        # ----------------------------------------------------------------------------------------------------------
        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # ----------------------------------------------------------------------------------------------------------
        # 3. Encode input prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self._encode_prompt(prompt,device,
                                              num_images_per_prompt,do_classifier_free_guidance,
                                              negative_prompt,max_embeddings_multiples,)
        dtype = text_embeddings.dtype

        # ----------------------------------------------------------------------------------------------------------
        # 4. Preprocess zaradress and mask
        img_features = self.img_encoder(clip_processored).pooler_output
        # ----------------------------------------------------------------------------------------------------------
        # 4. Preprocess zaradress and mask
        if isinstance(image, PIL.Image.Image):
            image = preprocess_image(image)
        if image is not None:
            image = image.to(device=self.device, dtype=dtype)
        if isinstance(mask_image, PIL.Image.Image):
            mask_image = preprocess_mask(mask_image, self.vae_scale_factor)
        if mask_image is not None:
            mask = mask_image.to(device=self.device, dtype=dtype)
            mask = torch.cat([mask] * batch_size * num_images_per_prompt)
        else:
            mask = None

        # ----------------------------------------------------------------------------------------------------------
        # 5. condition
        if do_classifier_free_guidance :
            con, uncon = text_embeddings.chunk(2)
            cls_embbedding = con[:, 0:1, :]
            remainder_embedding = con[:, 1:-1, :]
            b, l, d = con.shape
            img_features = img_features.repeat(b, 1, 1)
            cls_img = torch.cat((cls_embbedding, img_features), dim=1)
            con = torch.cat((cls_img, remainder_embedding), dim=1)
            text_embeddings = torch.cat((con, uncon), dim = 0)
        else :
            cls_embbedding = text_embeddings[:, 0:1, :]
            remainder_embedding = text_embeddings[:, 1:-1, :]
            b, l, d = text_embeddings.shape
            img_features = img_features.repeat(b, 1, 1)
            cls_img = torch.cat((cls_embbedding, img_features), dim=1)
            text_embeddings = torch.cat((cls_img, remainder_embedding), dim=1)

        # ----------------------------------------------------------------------------------------------------------
        # 6. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, image is None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # ----------------------------------------------------------------------------------------------------------
        # 7. Prepare latent variables
        latents, init_latents_orig, noise = self.prepare_latents(image,latent_timestep,batch_size * num_images_per_prompt,
                                                                 height,width,dtype,device,generator,latents,)
        # ----------------------------------------------------------------------------------------------------------
        # 8. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # ----------------------------------------------------------------------------------------------------------
        # 9. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if mask is not None:
                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)
                if is_cancelled_callback is not None and is_cancelled_callback():
                    return None
        image = self.decode_latents(latents)
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return image, has_nsfw_concept
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
