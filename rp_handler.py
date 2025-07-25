import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image

from transformers import CLIPVisionModelWithProjection

from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel, UniPCMultistepScheduler,
    DPMSolverMultistepScheduler
)

from controlnet_aux import MidasDetector, ZoeDetector

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


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-zoe-depth-sdxl-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True
)

PIPELINE = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    # "RunDiffusion/Juggernaut-XL-v9",
    # "SG161222/RealVisXL_V5.0",
    # "misri/cyberrealisticPony_v90Alt1",
    "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    controlnet=controlnet,
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
    algorithm_type="sde-dpmsolver++",   # важно: именно "dpmsolver++"
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

# midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")

CURRENT_LORA = "None"


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
        strength = float(payload.get("strength", 0.5))

        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)

        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # control scales
        depth_scale = float(payload.get("depth_conditioning_scale", 0.8))

        ip_adapter_scale = float(payload.get("ip_adapter_scale", 0.8))

        # ---------- препроцессинг входа ------------
        image_pil = url_to_pil(image_url)
        reference_pil = url_to_pil(reference_url)
        controlnet_img = processor_zoe(
            image_pil, output_type='cv2')

        height, width, _ = controlnet_img.shape
        ratio = np.sqrt(1024. * 1024. / (width * height))
        new_width = (int(width * ratio) // 8) * 8
        new_height = (int(height * ratio) // 8) * 8
        controlnet_img = cv2.resize(controlnet_img,
                                    (new_width,
                                     new_height))
        controlnet_img = Image.fromarray(controlnet_img)

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            ip_adapter_image=reference_pil,
            control_image=controlnet_img,
            controlnet_conditioning_scale=depth_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=new_height,
            width=new_width,
            strength=strength
        ).images

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


# docker build --tag sintecs/sd_design:migrate_from_rp --build-arg COG_REPO=fofr --build-arg COG_MODEL=style-transfer --build-arg COG_VERSION=f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0 .