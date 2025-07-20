import base64, cv2, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image

from transformers import CLIPVisionModelWithProjection

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)

from controlnet_aux import MidasDetector

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250
TARGET_RES = 1024  # SDXL рекомендует 1024×1024

logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #

def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
controlnets = [
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=DTYPE,
        use_safetensors=True
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=DTYPE
    )
]

PIPELINE = StableDiffusionXLControlNetPipeline.from_pretrained(
    # "RunDiffusion/Juggernaut-XL-v9",
    # "SG161222/RealVisXL_V5.0",
    # "misri/cyberrealisticPony_v90Alt1",
    "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    controlnet=controlnets,
    torch_dtype=DTYPE,
    # variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
    requires_safety_checker=False,
    add_watermarker=False,
    use_safetensors=True,
    resume_download=True,
)
# PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
#     PIPELINE.scheduler.config)
PIPELINE.scheduler = DPMSolverMultistepScheduler.from_config(
    PIPELINE.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++",   # важно: именно "dpmsolver++"
    solver_order=2,
    lower_order_final=True
)
# PIPELINE.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
#     "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
#     torch_dtype=torch.float16,
# ).to(DEVICE)

PIPELINE.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to(DEVICE)

PIPELINE.enable_xformers_memory_efficient_attention()
PIPELINE.to(DEVICE)

PIPELINE.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
)

midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")

CURRENT_LORA = "None"


def multiple_up(x: int, base: int = 64) -> int:
    """Округление вверх до кратного base."""
    return ((x + base - 1) // base) * base


def pad_image_center(img: Image.Image, target_w: int, target_h: int, fill=(0, 0, 0)):
    """
    Центрированный паддинг. Возвращает padded_image и смещения (left, top)
    для последующего обратного crop.
    """
    w, h = img.size
    if w == target_w and h == target_h:
        return img, (0, 0)
    left = (target_w - w) // 2
    top = (target_h - h) // 2
    new_im = Image.new("RGB", (target_w, target_h), fill)
    new_im.paste(img, (left, top))
    return new_im, (left, top)


def crop_back(img: Image.Image, orig_w: int, orig_h: int, left: int, top: int):
    return img.crop((left, top, left + orig_w, top + orig_h))


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required"}
        reference_url = payload.get("reference_url")
        if not reference_url:
            return {"error": "'reference_url' is required"}

        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        negative_prompt = payload.get("negative_prompt", "")
        guidance_scale = float(payload.get("guidance_scale", 7.5))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)

        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # control scales
        depth_scale = float(payload.get("depth_conditioning_scale", 0.8))
        canny_scale = float(payload.get("canny_conditioning_scale", 0.4))

        ip_adapter_scale = float(payload.get("ip_adapter_scale", 0.8))

        # # ---------- препроцессинг входа ------------
        # image_pil = url_to_pil(image_url)
        # reference_pil = url_to_pil(reference_url)
        # orig_w, orig_h = image_pil.size
        # depth_image = midas(image_pil)
        # canny_image = make_canny_condition(image_pil)

        # ---------- препроцессинг входа (без ресайза, только паддинг) ------------
        image_pil = url_to_pil(image_url)
        reference_pil = url_to_pil(reference_url)

        orig_w, orig_h = image_pil.size

        # Размеры, кратные 64 (или можно 8, но 64 надёжнее при нескольких контролях)
        pad_w = multiple_up(orig_w, 64)
        pad_h = multiple_up(orig_h, 64)

        # Генерируем depth/canny на оригинальном размере
        depth_raw = midas(image_pil)          # PIL (RGB)
        canny_raw = make_canny_condition(image_pil)

        # Паддим (центрируем) каждую карту
        # Для depth лучше средний серый фон (128) чтобы не создавать край-контраст
        depth_padded, (off_x, off_y) = pad_image_center(
            depth_raw, pad_w, pad_h, fill=(128, 128, 128))
        canny_padded, _ = pad_image_center(
            canny_raw, pad_w, pad_h, fill=(0, 0, 0))

        # Если где-то ещё нужен сам image_pil как условие (не в этом коде) — тоже можем паддить
        # img_padded, _ = pad_image_center(image_pil, pad_w, pad_h, fill=(0, 0, 0))

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[depth_padded, canny_padded],
            ip_adapter_image=reference_pil,
            control_image=[depth_padded, canny_padded],
            controlnet_conditioning_scale=[depth_scale, canny_scale],
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
        # Обрезаем результат к исходному размеру
        if pad_w != orig_w or pad_h != orig_h:
            images = [crop_back(im, orig_w, orig_h, off_x, off_y) for im in images]

        return {
            "images_base64": [pil_to_b64(i) for i in images],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps, "seed": seed,
            "lora": CURRENT_LORA if CURRENT_LORA != "None" else None,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
