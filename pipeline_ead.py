import copy
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import PIL.Image
from einops import rearrange
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import LCMScheduler
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def ddcm_sampler(
    scheduler, x_s, x_t, timestep, e_s, e_t, x_0, noise, eta, timestep_idx, to_next=True
):
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # if scheduler.step_index is None:
    #     scheduler._init_step_index(timestep)
    # prev_step_index = scheduler.step_index + 1

    prev_step_index = timestep_idx + 1
    if prev_step_index < len(scheduler.timesteps):
        prev_timestep = scheduler.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev
    std_dev_t = eta * variance
    noise = std_dev_t ** (0.5) * noise

    e_c = (x_s - alpha_prod_t ** (0.5) * x_0) / (1 - alpha_prod_t) ** (0.5)

    pred_x0 = x_0 + (
        (x_t - x_s) - beta_prod_t ** (0.5) * (e_t - e_s)
    ) / alpha_prod_t ** (0.5)
    # pred_x0 = (
    #     x_t - beta_prod_t**0.5 * (e_c + (e_t - e_s) * weights)
    # ) / alpha_prod_t ** (0.5)
    eps = e_c + (e_t - e_s)
    dir_xt = (beta_prod_t_prev - std_dev_t) ** (0.5) * eps

    # Noise is not used for one-step sampling.
    if len(scheduler.timesteps) > 1:
        prev_xt = alpha_prod_t_prev ** (0.5) * pred_x0 + dir_xt + noise
        prev_xs = alpha_prod_t_prev ** (0.5) * x_0 + dir_xt + noise
    else:
        prev_xt = pred_x0
        prev_xs = x_0

    # if to_next:
    #     scheduler._step_index += 1
    return prev_xs, prev_xt, pred_x0


## following MultiDiffusion: https://github.com/omerbt/MultiDiffusion/blob/master/panorama.py ##
## the window size is changed for 360-degree panorama generation ##
def get_views(panorama_height, panorama_width, window_size=[64, 128], stride=16):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size[0]) // stride + 1
    num_blocks_width = (panorama_width - window_size[1]) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size[0]
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size[1]
        views.append((h_start, h_end, w_start, w_end))
    return views


class EditPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LCMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        controller=None,
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.controller = controller
        self.processor_manager = ProcessorManager()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil"
                )
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        denoise_model,
        generator=None,
    ):
        image = image.to(device=device, dtype=dtype)

        batch_size = image.shape[0]

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt * num_images_per_prompt,
                dim=0,
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timestep
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        clean_latents = init_latents
        if denoise_model:
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
        else:
            latents = noise

        return latents, clean_latents

    # TODO
    # 1.
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        source_prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        positive_prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        original_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        source_guidance_scale: Optional[float] = 1,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        denoise: Optional[bool] = True,
        latent_size: int = 64,
        stride: int = 16,
        height: int = 512,
        width: int = 3072,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds_tuple = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        source_prompt_embeds_tuple = self.encode_prompt(
            source_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            positive_prompt,
            None,
        )
        if prompt_embeds_tuple[1] is not None:
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
        else:
            prompt_embeds = prompt_embeds_tuple[0]
        if source_prompt_embeds_tuple[1] is not None:
            source_prompt_embeds = torch.cat(
                [source_prompt_embeds_tuple[1], source_prompt_embeds_tuple[0]]
            )
        else:
            source_prompt_embeds = source_prompt_embeds_tuple[0]

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            original_inference_steps=original_inference_steps,
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        if denoise is False:
            strength = 1
        num_denoise_num = math.trunc(num_inference_steps * strength)
        num_start = num_inference_steps - num_denoise_num

        # 6. Prepare latent variables
        latents, clean_latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            denoise,
            generator,
        )
        source_latents = latents
        mutual_latents = latents

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        generator = extra_step_kwargs.pop("generator", None)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        z_0_list = []
        # 9. get multiple views for pano
        # latents are sampled from standard normal distribution (torch.randn()) with a size of Bx4x64x256,
        # where B denotes the batch size.
        mad_thereshold = 25
        views = get_views(height, width, window_size=[latent_size] * 2, stride=stride)
        count = torch.zeros_like(
            latents, requires_grad=False, device=self.device, dtype=self.dtype
        )
        value = torch.zeros_like(
            latents, requires_grad=False, device=self.device, dtype=self.dtype
        )

        # unet custom attention processor
        processor = AttnProcessor(
            batch_size=6 if do_classifier_free_guidance else 3,
            latent_h=height // 8,
            views=views,
            latent_w=width // 8,
            stride=stride,
            is_cons=False,
            num_start=num_start,
            num_steps=num_inference_steps,
            controller=self.controller,
        )
        self.unet.set_attn_processor(processor)
        self.set_attn_processor_mad(processor, "all")

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i >= mad_thereshold:
                    processor.apply_mad = False
                else:
                    pass
                batched_latent_views = []
                batched_source_latent_views = []
                batched_mutual_latent_views = []
                batched_clean_latent_views = []
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    latent_view = latents[:, :, h_start:h_end, w_start:w_end].detach()
                    source_view = source_latents[
                        :, :, h_start:h_end, w_start:w_end
                    ].detach()
                    mutual_view = mutual_latents[
                        :, :, h_start:h_end, w_start:w_end
                    ].detach()
                    clean_view = clean_latents[:, :, h_start:h_end, w_start:w_end]
                    batched_latent_views.append(latent_view)
                    batched_source_latent_views.append(source_view)
                    batched_mutual_latent_views.append(mutual_view)
                    batched_clean_latent_views.append(clean_view)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat(batched_latent_views * 2)
                    if do_classifier_free_guidance
                    else latents
                )
                source_latent_model_input = (
                    torch.cat(batched_source_latent_views * 2)
                    if do_classifier_free_guidance
                    else batched_source_latent_views
                )
                mutual_latent_model_input = (
                    torch.cat(batched_mutual_latent_views * 2)
                    if do_classifier_free_guidance
                    else batched_mutual_latent_views
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                source_latent_model_input = self.scheduler.scale_model_input(
                    source_latent_model_input, t
                )
                mutual_latent_model_input = self.scheduler.scale_model_input(
                    mutual_latent_model_input, t
                )

                # predict the noise residual
                if do_classifier_free_guidance:
                    concat_latent_model_input = torch.vstack(
                        [
                            source_latent_model_input.chunk(2)[0],
                            latent_model_input.chunk(2)[0],
                            mutual_latent_model_input.chunk(2)[0],
                            source_latent_model_input.chunk(2)[1],
                            latent_model_input.chunk(2)[1],
                            mutual_latent_model_input.chunk(2)[1],
                        ],
                        # dim=0,
                    )
                    concat_prompt_embeds = torch.vstack(
                        [
                            source_prompt_embeds[0].repeat(len(views), 1, 1),
                            prompt_embeds[0].repeat(len(views), 1, 1),
                            source_prompt_embeds[0].repeat(len(views), 1, 1),
                            source_prompt_embeds[1].repeat(len(views), 1, 1),
                            prompt_embeds[1].repeat(len(views), 1, 1),
                            source_prompt_embeds[1].repeat(len(views), 1, 1),
                        ],
                        # dim=0,
                    )
                else:
                    concat_latent_model_input = torch.cat(
                        [
                            source_latent_model_input.repeat(len(views)),
                            latent_model_input.repeat(len(views)),
                            mutual_latent_model_input.repeat(len(views)),
                        ],
                        dim=0,
                    )
                    concat_prompt_embeds = torch.cat(
                        [
                            source_prompt_embeds,
                            prompt_embeds,
                            source_prompt_embeds,
                        ],
                        dim=0,
                    )

                concat_noise_pred = self.unet(
                    concat_latent_model_input,
                    t,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_hidden_states=concat_prompt_embeds,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    (
                        source_noise_pred_uncond,
                        noise_pred_uncond,
                        mutual_noise_pred_uncond,
                        source_noise_pred_text,
                        noise_pred_text,
                        mutual_noise_pred_text,
                    ) = concat_noise_pred.chunk(6, dim=0)

                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    source_noise_pred = (
                        source_noise_pred_uncond
                        + source_guidance_scale
                        * (source_noise_pred_text - source_noise_pred_uncond)
                    )
                    mutual_noise_pred = (
                        mutual_noise_pred_uncond
                        + source_guidance_scale
                        * (mutual_noise_pred_text - mutual_noise_pred_uncond)
                    )

                else:
                    (source_noise_pred, noise_pred, mutual_noise_pred) = (
                        concat_noise_pred.chunk(3, dim=0)
                    )

                noise = torch.randn(
                    noise_pred.shape,
                    dtype=latents.dtype,
                    device=latents.device,
                    generator=generator,
                )

                _, latent_view_dn, pred_x0_view_dn = ddcm_sampler(
                    self.scheduler,
                    torch.vstack(batched_source_latent_views),
                    torch.vstack(batched_latent_views),
                    t,
                    source_noise_pred,
                    noise_pred,
                    torch.vstack(batched_clean_latent_views),
                    noise=noise,
                    eta=eta,
                    to_next=False,
                    timestep_idx=i,
                    **extra_step_kwargs,
                )

                source_latents_view_dn, mutual_latents_view_dn, pred_xm = ddcm_sampler(
                    self.scheduler,
                    torch.vstack(batched_source_latent_views),
                    torch.vstack(batched_mutual_latent_views),
                    t,
                    source_noise_pred,
                    mutual_noise_pred,
                    torch.vstack(batched_clean_latent_views),
                    noise=noise,
                    eta=eta,
                    timestep_idx=i,
                    **extra_step_kwargs,
                )

                # -----------------------------------------------------------------------------
                # TODO
                # need to be optimized
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value[:, :, h_start:h_end, w_start:w_end] += latent_view_dn[
                        view_idx
                    ]
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step (average the latents)
                latents = torch.where(count > 0, value / count, value)

                count.zero_()
                value.zero_()
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value[:, :, h_start:h_end, w_start:w_end] += pred_x0_view_dn[
                        view_idx
                    ]
                    count[:, :, h_start:h_end, w_start:w_end] += 1
                pred_x0 = torch.where(count > 0, value / count, value)
                count.zero_()
                value.zero_()

                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value[:, :, h_start:h_end, w_start:w_end] += source_latents_view_dn[
                        view_idx
                    ]
                    count[:, :, h_start:h_end, w_start:w_end] += 1
                source_latents = torch.where(count > 0, value / count, value)
                count.zero_()
                value.zero_()

                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    value[:, :, h_start:h_end, w_start:w_end] += mutual_latents_view_dn[
                        view_idx
                    ]
                    count[:, :, h_start:h_end, w_start:w_end] += 1
                mutual_latents = torch.where(count > 0, value / count, value)
                count.zero_()
                value.zero_()

                # -----------------------------------------------------------------------------

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        mutual_latents, latents = callback(
                            i, t, source_latents, latents, mutual_latents, alpha_prod_t
                        )
                z_0_list.append(pred_x0)

        # 8.5 save z_0
        # save_dir = os.path.join(os.getcwd(), "z_0_list")  # 保存目录
        # z_0_loop = tqdm(enumerate(z_0_list))

        # save_dir = os.path.join(save_dir, f"{prompt}_{source_prompt}".replace(" ", "_"))
        # os.makedirs(save_dir, exist_ok=True)
        # for idx, img in z_0_loop:
        #     file_name = f"z_0_{idx}.png"
        #     image = self.vae.decode(
        #         img / self.vae.config.scaling_factor, return_dict=False
        #     )[0]
        #     do_denormalize = [True] * image.shape[0]
        #     out_image = self.image_processor.postprocess(
        #         image, output_type=output_type, do_denormalize=do_denormalize
        #     )[0]
        #     print(type(out_image))
        #     # 保存图像
        #     out_image.save(os.path.join(save_dir, file_name))

        # 9. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(
                pred_x0 / self.vae.config.scaling_factor, return_dict=False
            )[0]
            has_nsfw_concept = None
        else:
            image = pred_x0
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def set_attn_processor_mad(self, processor, block_name):
        def fn_recursive_attn_processor(
            place_in_unet: str,
            module: torch.nn.Module,
            processor,
            count: int,
        ):
            """
            Recursively traverse the module and set the apply_mad attribute of the AttentionRefine processor.

            Args:
                name (str): The name of the module.
                module (torch.nn.Module): The module to traverse.
                processor (AttentionRefine or dict of AttentionRefine): The processor to set. If a dict, it must contain the processor for each place_in_unet.
            """
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    assert module.__class__.__name__ == "Attention", "not attention! "

                    module.processor = AttnProcessor(
                        processor.latent_h,
                        processor.latent_w,
                        processor.views,
                        processor.bs,
                        processor.stride,
                        processor.latent_size,
                        processor.mad,
                        processor.is_cons,
                        processor.self_replace_steps,
                        processor.num_start,
                        processor.start_steps,
                        processor.num_steps,
                        processor.controller,
                    )
                    module.processor = self.processor_manager.get_processor(
                        processor, place_in_unet=place_in_unet, apply_mad=True
                    )
                    return 1

                else:
                    raise NotImplementedError
            count = 0
            for _, child in module.named_children():
                count += fn_recursive_attn_processor(
                    place_in_unet, child, processor, count
                )
            return count

        count = 0
        for _, module in self.unet.down_blocks.named_children():
            count += fn_recursive_attn_processor(
                "down",
                module,
                processor,
                count,
            )
        for _, module in self.unet.mid_block.named_children():
            count += fn_recursive_attn_processor("mid", module, processor, count)
        for _, module in self.unet.up_blocks.named_children():
            count += fn_recursive_attn_processor("up", module, processor, count)

        self.controller.num_att_layers = count
        print(f"num_att_layers={count}")


class AttnProcessor:
    """
    Cross frame attention processor with scaled_dot_product attention of Pytorch 2.0.
    """

    def __init__(
        self,
        latent_h,
        latent_w,
        views,
        batch_size=1,
        stride=16,
        latent_size=64,
        mad=False,
        is_cons=False,
        self_replace_steps=0.7,
        num_start=0.3,
        start_steps=0,
        num_steps=10,
        controller=None,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        if controller is None:
            raise NotImplementedError

        self.controller = controller

        self.latent_h = latent_h
        self.latent_w = latent_w
        self.views = views
        self.bs = batch_size
        self.stride = stride
        self.mad = mad
        self.is_cons = is_cons
        self.latent_size = latent_size
        self.self_replace_steps = self_replace_steps
        self.num_start = num_start
        self.start_steps = start_steps
        self.cur_step = 0
        self.num_steps = num_steps
        self.place_in_unet = None
        self.apply_mad = None

    def compute_current_sizes(self, batch):
        bs, sequence_length, inner_dim = batch.shape
        views_len = bs // self.bs
        spatial_size = int(math.sqrt(sequence_length))
        down_factor = self.latent_size // spatial_size
        latent_h = self.latent_h // down_factor
        latent_w = self.latent_w // down_factor
        return views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim

    def merge_all_batched_qkv_views_into_canvas(self, batch_q, batch_k, batch_v):
        """
        Merges all batched query, key, and value views into a single canvas.

        Args:
            batch_q (torch.Tensor): The batched query tensor.
            batch_k (torch.Tensor): The batched key tensor.
            batch_v (torch.Tensor): The batched value tensor.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The merged query tensor.
                - torch.Tensor: The merged key tensor.
                - torch.Tensor: The merged value tensor.
                - int: The down factor used for merging.
        """
        views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim = (
            self.compute_current_sizes(batch_q)
        )
        if down_factor <= 0:
            raise ValueError("down_factor must be greater than zero")

        batch_q_views = rearrange(
            batch_q, "(b v) (h w) d -> b v d h w", v=views_len, h=spatial_size
        )
        batch_k_views = rearrange(
            batch_k, "(b v) (h w) d -> b v d h w", v=views_len, h=spatial_size
        )
        batch_v_views = rearrange(
            batch_v, "(b v) (h w) d -> b v d h w", v=views_len, h=spatial_size
        )
        canvas_q = torch.zeros(
            (self.bs, inner_dim, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype,
        )
        canvas_k = torch.zeros(
            (self.bs, inner_dim, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype,
        )
        canvas_v = torch.zeros(
            (self.bs, inner_dim, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype,
        )
        count = torch.zeros(
            (self.bs, inner_dim, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype,
        )
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas_q[:, :, h_start:h_end, w_start:w_end] += batch_q_views[:, view_idx]
            canvas_k[:, :, h_start:h_end, w_start:w_end] += batch_k_views[:, view_idx]
            canvas_v[:, :, h_start:h_end, w_start:w_end] += batch_v_views[:, view_idx]
            count[:, :, h_start:h_end, w_start:w_end] += 1
        batch_q = torch.where(count > 0, canvas_q / count, canvas_q)
        batch_k = torch.where(count > 0, canvas_k / count, canvas_k)
        batch_v = torch.where(count > 0, canvas_v / count, canvas_v)
        return batch_q, batch_k, batch_v, down_factor

    def merge_batched_q_views_into_canvas(self, batch):
        views_len, spatial_size, down_factor, latent_h, latent_w, inner_dim = (
            self.compute_current_sizes(batch)
        )
        batch_views = rearrange(
            batch, "(b v) (h w) d -> b v d h w", v=views_len, h=spatial_size
        )
        canvas = torch.zeros(
            (self.bs, inner_dim, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype,
        )
        count = torch.zeros(
            (self.bs, inner_dim, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype,
        )
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas[:, :, h_start:h_end, w_start:w_end] += batch_views[:, view_idx]
            count[:, :, h_start:h_end, w_start:w_end] += 1
        batch = torch.where(count > 0, canvas / count, canvas)
        return batch, down_factor

    def split_canvas_into_views(self, canvas, down_factor):
        canvas_views = []
        for view_idx, (h_start, h_end, w_start, w_end) in enumerate(self.views):
            h_start, h_end = h_start // down_factor, h_end // down_factor
            w_start, w_end = w_start // down_factor, w_end // down_factor
            canvas_views.append(canvas[:, :, h_start:h_end, w_start:w_end, :])
        canvas = torch.cat(canvas_views, dim=1)
        canvas = rearrange(canvas, "b v h w d -> (b v) (h w) d")
        return canvas

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        place_in_unet = self.place_in_unet
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        residual = hidden_states
        inner_dim = hidden_states.shape[-1]
        input_ndim = hidden_states.ndim
        self.device = hidden_states.device
        self.dtype = hidden_states.dtype
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = inner_dim // attn.heads
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if not is_cross_attention:
            query, key, value = self.controller.self_attn_forward(
                query, key, value, attn.heads, bs=len(self.views)
            )
            query = attn.batch_to_head_dim(query)
            key = attn.batch_to_head_dim(key)
            value = attn.batch_to_head_dim(value)
            if self.apply_mad:
                query, key, value, down_factor = (
                    self.merge_all_batched_qkv_views_into_canvas(query, key, value)
                )
                query = rearrange(
                    query, "b (nh hd) h w -> b nh (h w) hd", nh=attn.heads, hd=head_dim
                ).contiguous()
                key = rearrange(
                    key, "b (nh hd) h w -> b nh (h w) hd", nh=attn.heads, hd=head_dim
                ).contiguous()
                value = rearrange(
                    value, "b (nh hd) h w -> b nh (h w) hd", nh=attn.heads, hd=head_dim
                ).contiguous()
            else:
                query = rearrange(
                    query, "b hw (nh nd) -> b nh hw nd", nh=attn.heads, nd=head_dim
                )
                key = rearrange(
                    key, "b hw (nh nd) -> b nh hw nd", nh=attn.heads, nd=head_dim
                )
                value = rearrange(
                    value, "b hw (nh nd) -> b nh hw nd", nh=attn.heads, nd=head_dim
                )
            _query = rearrange(query, "b nh hw nd -> (b nh) hw nd", nh=attn.heads)
            _key = rearrange(key, "b nh hw nd -> (b nh) hw nd", nh=attn.heads)
            _value = rearrange(value, "b nh hw nd -> (b nh) hw nd", nh=attn.heads)
            attention_probs = attn.get_attention_scores(_query, _key, attention_mask)
        if is_cross_attention:
            query = rearrange(query, "(b n) h w -> b h (n w)", n=attn.heads)
            key = rearrange(key, "(b n) h w -> b h (n w)", n=attn.heads)
            value = rearrange(value, "(b n) h w -> b h (n w)", n=attn.heads)
            query, down_factor = self.merge_batched_q_views_into_canvas(query)
            query = rearrange(
                query, "b (nh hd) h w -> b nh (h w) hd", nh=attn.heads, hd=head_dim
            )
            key = key[: self.bs]
            key = rearrange(key, "b p (nh nd) -> b nh p nd", nh=attn.heads, nd=head_dim)
            value = value[: self.bs]
            value = rearrange(
                value, "b p (nh nd) -> b nh p nd", nh=attn.heads, nd=head_dim
            )

            _query = rearrange(query, "b nh hw nd -> (b nh) hw nd", nh=attn.heads)
            _key = rearrange(key, "b nh hw nd -> (b nh) hw nd", nh=attn.heads)
            _value = rearrange(value, "b nh hw nd -> (b nh) hw nd", nh=attn.heads)
            attention_probs = attn.get_attention_scores(_query, _key, attention_mask)

            attention_probs = self.controller(
                attention_probs,
                is_cross_attention,
                place_in_unet=place_in_unet,
            )
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = torch.bmm(attention_probs, _value)
        # hidden_states = F.scaled_dot_product_attention(
        #     query,
        #     key,
        #     value,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     scale=attn.scale,
        # )
        if not is_cross_attention and not self.apply_mad:
            hidden_states = rearrange(
                hidden_states, "(b nh) hw nd -> b hw (nh nd)", nh=attn.heads
            )
        else:
            latent_h = self.latent_h // down_factor
            hidden_states = rearrange(
                hidden_states,
                "(b nh) (h w) hd -> b 1 h w (nh hd)",
                h=latent_h,
                nh=attn.heads,
            )
            hidden_states = self.split_canvas_into_views(hidden_states, down_factor)

        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        self.cur_att_layer = 0
        self.cur_step += 1
        return hidden_states


class ProcessorManager:
    def __init__(self):
        self._processor_cache = {}  # 用于缓存基础processor

    def get_processor(self, base_processor, place_in_unet, apply_mad):
        """获取或创建processor实例"""
        # 使用关键参数作为缓存key
        cache_key = tuple(
            tuple(x) if isinstance(x, list) else x
            for x in (
                base_processor.latent_h,
                base_processor.latent_w,
                base_processor.views,
                base_processor.bs,
                base_processor.stride,
                base_processor.latent_size,
                base_processor.mad,
                base_processor.is_cons,
                base_processor.self_replace_steps,
                base_processor.num_start,
                base_processor.start_steps,
                base_processor.num_steps,
                hash(str(base_processor.controller)),  # 如果controller是复杂对象
            )
        )
        if cache_key not in self._processor_cache:
            # 创建新的基础processor
            self._processor_cache[cache_key] = AttnProcessor(
                base_processor.latent_h,
                base_processor.latent_w,
                base_processor.views,
                base_processor.bs,
                base_processor.stride,
                base_processor.latent_size,
                base_processor.mad,
                base_processor.is_cons,
                base_processor.self_replace_steps,
                base_processor.num_start,
                base_processor.start_steps,
                base_processor.num_steps,
                base_processor.controller,
            )

        # 创建轻量级processor,只设置place_in_unet
        new_processor = AttnProcessor.__new__(AttnProcessor)
        for key, value in vars(self._processor_cache[cache_key]).items():
            setattr(new_processor, key, value)
        new_processor.place_in_unet = place_in_unet
        new_processor.apply_mad = apply_mad
        return new_processor
