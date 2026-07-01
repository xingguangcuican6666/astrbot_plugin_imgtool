
import aiohttp, json
from typing import Any, Dict, List
from astrbot.api import logger
from .base import ImageProviderAdapter

class OpenRouterAdapter(ImageProviderAdapter):
    async def generate_image(
        self, *, base_url: str, api_key: str, model: str, prompt: str,
        negative_prompt: str | None = None, image_size: str | None = None,
        batch_size: int | None = None, seed: int | None = None,
        num_inference_steps: int | None = None, guidance_scale: float | None = None,
        cfg: float | None = None, image: str | None = None,
        image2: str | None = None, image3: str | None = None,
        extra_headers: Dict[str, str] | None = None,
        provider_options: Dict[str, Any] | None = None,
    ):
        url = (base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
        text = prompt if not negative_prompt else f"{prompt}\n\n[Negative prompt]: {negative_prompt}"
        content = [{"type": "text", "text": text}]
        for u in (image, image2, image3):
            if u: content.append({"type": "image_url", "image_url": {"url": u}})
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "modalities": ["image", "text"]}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers: headers.update(extra_headers)
        logger.info(f"[imgtool] POST {url}")
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, json=payload) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} when POST {url}: {txt}")
                data = json.loads(txt)
        urls: List[str] = []
        try:
            ch = (data.get("choices") or [])[0] or {}
            msg = ch.get("message") or {}
            imgs = msg.get("images") or []
            for it in imgs:
                if isinstance(it, dict):
                    obj = it.get("image_url") or it.get("image") or {}
                    u = obj.get("url") if isinstance(obj, dict) else None
                    if u: urls.append(u)
            if not urls:
                parts = msg.get("content")
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") in ("image_url", "output_image", "image"):
                            obj = p.get("image_url") or p.get("image") or {}
                            u = obj.get("url") if isinstance(obj, dict) else None
                            if u: urls.append(u)
        except Exception as e:
            from astrbot.api import logger as _log
            _log.error(f"[imgtool] parse OpenRouter response failed: {e}", exc_info=True)
        return urls, data
