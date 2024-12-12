from diffusers import LCMScheduler
from pipeline_ead import EditPipeline
import os
import torch
from PIL import Image
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Dict
import abc
import ptp_utils
import utils
import numpy as np
import seq_aligner
import math

LOW_RESOURCE = False
MAX_NUM_WORDS = 77

is_colab = utils.is_google_colab()
colab_instruction = (
    ""
    if is_colab
    else """
Colab Instuction"""
)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"
device_print = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"
device = "cuda" if torch.cuda.is_available() else "cpu"

if is_colab:
    scheduler = LCMScheduler.from_config(model_id_or_path, subfolder="scheduler")
    pipe = EditPipeline.from_pretrained(
        model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype
    )
else:
    scheduler = LCMScheduler.from_config(
        model_id_or_path,
        use_auth_token=os.environ.get("USER_TOKEN"),
        subfolder="scheduler",
    )
    pipe = EditPipeline.from_pretrained(
        model_id_or_path,
        use_auth_token=os.environ.get("USER_TOKEN"),
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )

tokenizer = pipe.tokenizer
encoder = pipe.text_encoder

if torch.cuda.is_available():
    pipe = pipe.to("cuda")


class LocalBlend:

    def get_mask(self, x_t, maps, word_idx, thresh, i):
        maps = maps * word_idx.reshape(1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, 1 : self.len - 1]).mean(0, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask

    def save_image(self, mask, i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"inter/{caption}"):
            os.mkdir(f"inter/{caption}")
        ptp_utils.save_images(image, f"inter/{caption}/{i}.jpg")

    def __call__(
        self,
        i,
        x_s,
        x_t,
        x_m,
        attention_store,
        alpha_prod,
        temperature=0.15,
        use_xm=False,
    ):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        h, w = x_t.shape[2], x_t.shape[3]
        h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        maps = [
            item.reshape(
                2,
                -1,
                1,
                h // int((h * w / item.shape[-2]) ** 0.5),
                w // int((h * w / item.shape[-2]) ** 0.5),
                MAX_NUM_WORDS,
            )
            for item in maps
        ]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0, :]
        maps_m = maps[1, :]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
            thresh_e = self.thresh_e
        thresh_m = self.thresh_m
        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m = self.get_mask(x_t, maps_s, (self.alpha_m - self.alpha_me), thresh_m, i)
        mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        if self.save_inter:
            self.save_image(mask_e, i, "mask_e")
            self.save_image(mask_m, i, "mask_m")
            self.save_image(mask_me, i, "mask_me")

        if self.alpha_e.sum() == 0:
            x_t_out = x_t
        else:
            x_t_out = torch.where(mask_e, x_t, x_m)
        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
            x_t_out = torch.where(mask_me, x_m, x_t_out)

        return x_m, x_t_out

    def __init__(self, thresh_e=0.3, thresh_m=0.3, save_inter=False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter

    def set_map(self, ms, alpha, alpha_e, alpha_m, len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def self_attn_forward(
        self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs
    ):
        b = q.shape[0] // num_heads
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i > 0):
            use_xm = self.cur_step + self.start_steps + 1 == self.num_steps
            x_m, x_t = self.local_blend(
                i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm
            )
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def attn_batch(
        self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs
    ):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out

    def self_attn_forward(self, q, k, v, num_heads):
        if q.shape[0] // num_heads == 3:
            if self.self_replace_steps <= (
                (self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps
            ):
                q = torch.cat([q[: num_heads * 2], q[num_heads : num_heads * 2]])
                k = torch.cat([k[: num_heads * 2], k[:num_heads]])
                v = torch.cat([v[: num_heads * 2], v[:num_heads]])
            else:
                q = torch.cat([q[:num_heads], q[:num_heads], q[:num_heads]])
                k = torch.cat([k[:num_heads], k[:num_heads], k[:num_heads]])
                v = torch.cat([v[: num_heads * 2], v[:num_heads]])
            return q, k, v
        else:
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            if self.self_replace_steps <= (
                (self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps
            ):
                qu = torch.cat([qu[: num_heads * 2], qu[num_heads : num_heads * 2]])
                qc = torch.cat([qc[: num_heads * 2], qc[num_heads : num_heads * 2]])
                ku = torch.cat([ku[: num_heads * 2], ku[:num_heads]])
                kc = torch.cat([kc[: num_heads * 2], kc[:num_heads]])
                vu = torch.cat([vu[: num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[: num_heads * 2], vc[:num_heads]])
            else:
                qu = torch.cat([qu[:num_heads], qu[:num_heads], qu[:num_heads]])
                qc = torch.cat([qc[:num_heads], qc[:num_heads], qc[:num_heads]])
                ku = torch.cat([ku[:num_heads], ku[:num_heads], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads], kc[:num_heads], kc[:num_heads]])
                vu = torch.cat([vu[: num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[: num_heads * 2], vc[:num_heads]])

            return (
                torch.cat([qu, qc], dim=0),
                torch.cat([ku, kc], dim=0),
                torch.cat([vu, vc], dim=0),
            )

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce, attn_masa = attn[0], attn[1], attn[2]
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_repalce)
            attn_base_store = self.replace_cross_attention(attn_base, attn_repalce)
            if self.cross_replace_steps >= (
                (self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps
            ):
                attn[1] = attn_base_store
            attn_store = torch.cat([attn_base_store, attn_replace_new])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
            super(AttentionControlEdit, self).forward(
                attn_store, is_cross, place_in_unet
            )
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        start_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Tuple[float, float]]
        ],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
    ):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts) + 1
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend
        )
        self.mapper = (
            seq_aligner.get_replacement_mapper(prompts, tokenizer)
            .to(device)
            .to(torch_dtype)
        )


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        prompt_specifiers,
        num_steps: int,
        start_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super(AttentionRefine, self).__init__(
            prompts,
            num_steps,
            start_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(
            prompts, prompt_specifiers, tokenizer, encoder, device
        )
        self.mapper, alphas, ms = (
            self.mapper.to(device),
            alphas.to(device).to(torch_dtype),
            ms.to(device).to(torch_dtype),
        )
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(ms, alphas, alpha_e, alpha_m, t_len)


def get_equalizer(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    values: Union[List[float], Tuple[float, ...]],
):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def inference(
    img,
    source_prompt,
    target_prompt,
    local,
    mutual,
    positive_prompt,
    negative_prompt,
    guidance_s,
    guidance_t,
    num_inference_steps,
    width,
    height,
    seed,
    strength,
    cross_replace_steps,
    self_replace_steps,
    thresh_e,
    thresh_m,
    denoise,
    user_instruct="",
    api_key="",
):

    if user_instruct != "" and api_key != "":
        (
            source_prompt,
            target_prompt,
            local,
            mutual,
            replace_steps,
            num_inference_steps,
        ) = get_params(api_key, user_instruct)
        cross_replace_steps = replace_steps
        self_replace_steps = replace_steps

    torch.manual_seed(seed)
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    if denoise is False:
        strength = 1
    num_denoise_num = math.trunc(num_inference_steps * strength)
    num_start = num_inference_steps - num_denoise_num
    # create the CAC controller.
    local_blend = LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, save_inter=False)
    controller = AttentionRefine(
        [source_prompt, target_prompt],
        [[local, mutual]],
        num_inference_steps,
        num_start,
        cross_replace_steps=cross_replace_steps,
        self_replace_steps=self_replace_steps,
        local_blend=local_blend,
    )
    ptp_utils.register_attention_control(pipe, controller)

    results = pipe(
        prompt=target_prompt,
        source_prompt=source_prompt,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=img,
        num_inference_steps=15,
        eta=1,
        strength=strength,
        guidance_scale=guidance_t,
        source_guidance_scale=guidance_s,
        denoise_model=denoise,
        callback=None,
    )

    return replace_nsfw_images(results)


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images[0]


css = """.cycle-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.cycle-diffusion-div div h1{font-weight:900;margin-bottom:7px}.cycle-diffusion-div p{margin-bottom:10px;font-size:94%}.cycle-diffusion-div p a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
intro = """
<div style="display: flex;align-items: center;justify-content: center">
    <img src="https://sled-group.github.io/InfEdit/image_assets/InfEdit.png" width="80" style="display: inline-block">
    <h1 style="margin-left: 12px;text-align: center;margin-bottom: 7px;display: inline-block">InfEdit</h1>
    <h3 style="display: inline-block;margin-left: 10px;margin-top: 6px;font-weight: 500">Inversion-Free Image Editing
with Natural Language</h3>
</div>
"""

param_bot_prompt = """
You are a helpful assistant named InfEdit that provides input parameters to the image editing model based on user instructions. You should respond in valid json format.

User:
```
{image descrption and editing commands | example: 'The image shows an apple on the table and I want to change the apple to a banana.'}
```

After receiving this, you will need to generate the appropriate params as input to the image editing models.

Assistant:
```
{
“source_prompt”: “{a string describes the input image, it needs to includes the thing user want to change | example: 'an apple on the table'}”,
“target_prompt”: “{a string that matches the source prompt, but it needs to includes the thing user want to change | example: 'a banana on the table'}”,
“target_sub”: “{a special substring from the target prompt}”,
“mutual_sub”: “{a special mutual substring from source/target prompt}”
“attention_control”: {a number between 0 and 1}
“steps”: {a number between 8 and 50}
}
```

You need to fill in the "target_sub" and "mutual_sub" by the guideline below.

If the editing instruction is not about changing style or background:
- The "target_sub" should be a special substring from the target prompt that highlights what you want to edit, it should be as short as possible and should only be noun ("banana" instead of "a banana"). 
- The "mutual_sub" should be kept as an empty string.
P.S. When you want to remove something, it's always better to use "empty", "nothing" or some appropriate words to replace it. Like remove an apple on the table, you can use "an apple on the table" and "nothing on the table" as your prompts, and use "nothing" as your target_sub.
P.S. You should think carefully about what you want to modify, like "short hair" to "long hair", your target_sub should be "hair" instead of "long".
P.S. When you are adding something, the target_sub should be the thing you want to add.

If it's about style editing:
- The "target_sub" should be kept as an empty string.
- The "mutual_sub" should be kept as an empty string.

If it's about background editing:
- The "target_sub" should be kept as an empty string.
- The "mutual_sub" should be a common substring from source/target prompt, and is the main object/character (noun) in the image. It should be as short as possible and only be noun ("banana" instead of "a banana", "man" instead of "running man").

A specific case, if it's about change an object's abstract information, like pose, view or shape and want to keep the semantic feature same, like a dog to a running dog,
- The "target_sub" should be a special substring from the target prompt that highlights what you want to edit, it should be as short as possible and should only be noun ("dog" instead of "a running dog"). 
- The "mutual_sub" should be as same as target_sub because we want to "edit the dog but also keep the dog as same".


You need to choose a specific value of “attention_control” by the guideline below.
A larger value of “attention_control” means more consistency between the source image and the output.

- the editing is on the feature level, like color, material and so on, and want to ensure the characteristics of the original object as much as possible, you should choose a large value. (Example: for color editing, you can choose 1, and for material you can choose 0.9)
- the editing is on the object level, like edit a "cat" to a "dog", or a "horse" to a "zebra", and want to make them to be similar, you need to choose a relatively large value, we say 0.7 for example.
- the editing is changing the style but want to keep the spatial features, you need to choose a relatively large value, we say 0.7 for example.
- the editing need to change something's shape, like edit an "apple" to a "banana", a "flower" to a "knife", "short" hair to "long" hair, "round" to "square", which have very different shapes, you need to choose a relatively small value, we say 0.3 for example.
- the editing is tring to change the spatial information, like change the pose and so on, you need to choose a relatively small value, we say 0.3 for example.
- the editing should not consider the consistency with the input image, like add something new, remove something, or change the background, you can directly use 0.


You need to choose a specific value of “steps” by the guideline below.
More steps mean that the edit effect is more pronounced.
- If the editing is super easy, like changing something to something with very similar features, you can choose 8 steps.
- In most cases, you can choose 15 steps.
- For style editing and remove tasks, you can choose a larger value, like 25 steps.
- If you feel the task is extremely difficult (like some kinds of styles or removing very tiny stuffs), you can directly use 50 steps.
"""


def get_params(api_key, user_instruct):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    print("user_instruct", user_instruct)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": param_bot_prompt},
            {"role": "user", "content": user_instruct},
        ],
        response_format={"type": "json_object"},
    )
    param_dict = response.choices[0].message.content
    print("param_dict", param_dict)
    import json

    param_dict = json.loads(param_dict)
    return (
        param_dict["source_prompt"],
        param_dict["target_prompt"],
        param_dict["target_sub"],
        param_dict["mutual_sub"],
        param_dict["attention_control"],
        param_dict["steps"],
    )


output = inference(
    Image.open("images/corgi.jpg"),
    "corgi",
    "tiger",
    "cat",
    "",
    "",
    "",
    1,
    2,
    15,
    512,
    512,
    0,
    1,
    0.7,
    0.7,
    0.6,
    0.6,
    False,
)

output.save("output.png")
