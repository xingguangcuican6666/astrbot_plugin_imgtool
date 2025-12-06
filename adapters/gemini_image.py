import aiohttp
import json
from typing import Dict, List, Tuple

from astrbot.api import logger
from .base import ImageProviderAdapter


class GeminiImageAdapter(ImageProviderAdapter):
    async def generate_image(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        image_size: str | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        cfg: float | None = None,
        image: str | None = None,
        image2: str | None = None,
        image3: str | None = None,
        extra_headers: Dict[str, str] | None = None,
    ) -> Tuple[List[str], Dict]:
        """
        Adapter for Google Gemini 3 Pro Image (preview).

        Maps the plugin's generic image parameters onto the Gemini
        generateContent JSON API and returns data: URLs for images.
        """
        # Decide endpoint
        base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        model_name = (model or "gemini-3-pro-image-preview").strip()

        if base.endswith(":generateContent"):
            url = base
        elif "/models/" in base:
            url = base
            if not url.endswith(":generateContent"):
                url = url + ":generateContent"
        else:
            url = f"{base}/models/{model_name}"
            if not url.endswith(":generateContent"):
                url = url + ":generateContent"

        if not api_key:
            raise RuntimeError("Gemini image provider requires api_key")

        # Gemini HTTP API typically uses ?key=API_KEY
        params: Dict[str, str] = {"key": api_key}

        # Build prompt text (include negative prompt if any)
        text = prompt or ""
        if negative_prompt:
            if text:
                text = f"{text}\n\n[Negative prompt]: {negative_prompt}"
            else:
                text = f"[Negative prompt]: {negative_prompt}"

        # Build parts: text + optional reference images
        parts: List[Dict[str, object]] = []
        if text:
            parts.append({"text": text})

        def _data_url_to_inline(u: str) -> Dict[str, object] | None:
            if not isinstance(u, str):
                return None
            if not u.startswith("data:"):
                return None
            try:
                head, b64 = u.split(",", 1)
            except ValueError:
                return None
            mime = "image/png"
            head = head.strip()
            if head.startswith("data:") and ";" in head:
                mime = head[5:].split(";", 1)[0] or mime
            elif head.startswith("data:") and len(head) > 5:
                mime = head[5:]
            return {
                "inlineData": {
                    "mimeType": mime,
                    "data": b64,
                }
            }

        for u in (image, image2, image3):
            if not u:
                continue
            part = _data_url_to_inline(u)
            if part:
                parts.append(part)

        contents = [
            {
                "role": "user",
                "parts": parts or [{"text": text or prompt}],
            }
        ]

        # Map size: treat colon-form "W:H" as aspectRatio, otherwise as imageSize (1K/2K/4K)
        aspect_ratio = "1:1"
        resolution = "1K"
        if image_size:
            s = str(image_size).strip()
            if ":" in s:
                aspect_ratio = s
            else:
                resolution = s

        generation_config: Dict[str, object] = {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": resolution,
            },
        }

        payload: Dict[str, object] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        logger.info(f"[imgtool] POST {url}")
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, params=params, json=payload) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} when POST {url}: {txt}")
                data = json.loads(txt)

        urls: List[str] = []

        try:
            candidates = data.get("candidates") or []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                content = cand.get("content") or {}
                parts_list = content.get("parts") or []
                for p in parts_list:
                    if not isinstance(p, dict):
                        continue
                    inline = p.get("inlineData") or p.get("inline_data")
                    if not isinstance(inline, dict):
                        continue
                    mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                    b64 = inline.get("data")
                    if isinstance(b64, str):
                        urls.append(f"data:{mime};base64,{b64}")

            # Fallback: top-level parts (if present)
            if not urls:
                top_parts = data.get("parts") or []
                for p in top_parts:
                    if not isinstance(p, dict):
                        continue
                    inline = p.get("inlineData") or p.get("inline_data")
                    if not isinstance(inline, dict):
                        continue
                    mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                    b64 = inline.get("data")
                    if isinstance(b64, str):
                        urls.append(f"data:{mime};base64,{b64}")
        except Exception as e:
            from astrbot.api import logger as _log

            _log.error(f"[imgtool] parse Gemini image response failed: {e}", exc_info=True)

        return urls, data

