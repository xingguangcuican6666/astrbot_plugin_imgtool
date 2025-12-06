
import aiohttp, json, re
from typing import Dict, List, Tuple
from astrbot.api import logger
from .base import ImageProviderAdapter

class OpenAIImagesAdapter(ImageProviderAdapter):
    async def generate_image(
        self, *, base_url: str, api_key: str, model: str, prompt: str,
        negative_prompt: str | None = None, image_size: str | None = None,
        batch_size: int | None = None, seed: int | None = None,
        num_inference_steps: int | None = None, guidance_scale: float | None = None,
        cfg: float | None = None, image: str | None = None,
        image2: str | None = None, image3: str | None = None,
        extra_headers: Dict[str, str] | None = None,
    ):
        # 允许：/images/generations 完整端点；或 /v1 / 根域名，自动补齐
        b = (base_url or "").rstrip("/")
        if not b:
            b = "https://api.example.com/v1"  # 留给上层或子类覆盖默认
        if b.endswith("/images/generations"):
            url = b
        else:
            if not re.search(r"/v\d+$", b):
                b = b + "/v1"
            url = b + "/images/generations"
        logger.info(f"[imgtool] POST {url}")

        # 模型族，决定可发送字段
        m = (model or "").lower()
        if "kolors" in m:
            family = "kolors"
        elif "image-edit" in m:
            family = "qwen_edit"
        elif "qwen-image" in m:
            family = "qwen_image"
        else:
            family = "generic"
        logger.info(f"[imgtool] model_family={family}")

        payload: Dict = {"model": model, "prompt": prompt}
        if negative_prompt: payload["negative_prompt"] = negative_prompt
        if seed is not None: payload["seed"] = int(seed)
        if num_inference_steps: payload["num_inference_steps"] = int(num_inference_steps)

        if family == "kolors":
            if image_size: payload["image_size"] = image_size
            if batch_size: payload["batch_size"] = max(1, min(4, int(batch_size)))
            if guidance_scale is not None: payload["guidance_scale"] = float(guidance_scale)
            if image: payload["image"] = image
        elif family == "qwen_image":
            if image_size: payload["image_size"] = image_size
            if cfg is not None: payload["cfg"] = float(cfg)
            if image: payload["image"] = image
        elif family == "qwen_edit":
            if image: payload["image"] = image
            if "2509" in m:
                if image2: payload["image2"] = image2
                if image3: payload["image3"] = image3
        else:
            if image_size: payload["image_size"] = image_size
            if cfg is not None: payload["cfg"] = float(cfg)
            if image: payload["image"] = image

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers: headers.update(extra_headers)
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, json=payload) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} when POST {url}: {txt}")
                data = json.loads(txt)

        urls: List[str] = []
        for item in (data.get("images") or data.get("data") or []):
            if isinstance(item, dict):
                u = item.get("url")
                if u: urls.append(u)
                if not u and "b64_json" in item:
                    urls.append(f"data:image/png;base64,{item['b64_json']}")
        return urls, data
