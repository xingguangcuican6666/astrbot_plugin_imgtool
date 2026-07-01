import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode, urlparse

import aiohttp

from astrbot.api import logger

from .base import ImageProviderAdapter


DEFAULT_XFYUN_URL = "https://maas-api.cn-huabei-1.xf-yun.com/v2.1/tti"
DEFAULT_SCHEDULER = "DPM++ 2M Karras"
SUPPORTED_SIZES = {
    "768x768",
    "1024x1024",
    "576x1024",
    "768x1024",
    "1024x576",
    "1024x768",
}


class XfyunSparkAdapter(ImageProviderAdapter):
    def _build_auth_url(self, base_url: str, api_key: str, api_secret: str) -> str:
        parsed = urlparse(base_url)
        host = parsed.netloc
        path = parsed.path or "/"
        date = format_datetime(datetime.now(timezone.utc), usegmt=True)
        signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/1.1"
        digest = hmac.new(
            api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(digest).decode("utf-8")
        authorization_origin = (
            f'api_key="{api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
        query = urlencode(
            {
                "authorization": authorization,
                "date": date,
                "host": host,
            }
        )
        return f"{base_url}?{query}"

    def _parse_size(self, image_size: str | None) -> Tuple[int, int]:
        size = (image_size or "1024x1024").strip().lower()
        if size not in SUPPORTED_SIZES:
            choices = ", ".join(sorted(SUPPORTED_SIZES))
            raise RuntimeError(f"讯飞生图仅支持以下分辨率：{choices}")
        width_s, height_s = size.split("x", 1)
        return int(width_s), int(height_s)

    def _is_flux_model(self, domain: str) -> bool:
        return "flux.1-dev" in (domain or "").lower()

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
        provider_options: Dict[str, Any] | None = None,
    ) -> Tuple[List[str], Dict]:
        opts = provider_options or {}
        app_id = str(opts.get("app_id") or "").strip()
        api_secret = str(opts.get("api_secret") or "").strip()
        uid = str(opts.get("uid") or "").strip()
        patch_id = opts.get("patch_id")
        scheduler = str(opts.get("scheduler") or DEFAULT_SCHEDULER).strip() or DEFAULT_SCHEDULER

        if not app_id or not api_key or not api_secret:
            raise RuntimeError("讯飞 provider 需要同时配置 app_id、api_key、api_secret。")
        if not model or not str(model).strip():
            raise RuntimeError("讯飞 provider 需要配置 model（讯飞文档中的 modelID/domain）。")
        if batch_size and int(batch_size) > 1:
            logger.info("[imgtool] xfyun only supports single image; batch_size will be forced to 1")
        if image or image2 or image3:
            raise RuntimeError("讯飞文生图接口暂不支持参考图参数。")
        if cfg is not None:
            raise RuntimeError("讯飞文生图接口暂不支持 cfg 参数。")

        raw_url = (base_url or DEFAULT_XFYUN_URL).strip()
        if not raw_url:
            raw_url = DEFAULT_XFYUN_URL

        domain = str(model).strip()
        width, height = self._parse_size(image_size)

        chat_params: Dict[str, Any] = {
            "domain": domain,
            "width": width,
            "height": height,
            "seed": int(seed if seed is not None else 42),
            "num_inference_steps": int(num_inference_steps if num_inference_steps is not None else 20),
            "guidance_scale": float(guidance_scale if guidance_scale is not None else 5.0),
            "scheduler": scheduler,
        }

        if self._is_flux_model(domain):
            # 官方文档注明 FLUX.1-dev 参数固定为默认值，避免下发自定义值触发错误。
            chat_params = {"domain": domain}

        header: Dict[str, Any] = {"app_id": app_id}
        if uid:
            header["uid"] = uid
        if patch_id:
            header["patch_id"] = patch_id

        payload: Dict[str, Any] = {
            "message": {
                "text": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        }
        if negative_prompt:
            payload["negative_prompts"] = {"text": negative_prompt}

        body: Dict[str, Any] = {
            "header": header,
            "parameter": {"chat": chat_params},
            "payload": payload,
        }

        url = self._build_auth_url(raw_url, api_key=api_key, api_secret=api_secret)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        logger.info(f"[imgtool] POST {raw_url} (xfyun)")
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, json=body) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} when POST {raw_url}: {txt}")
                data = json.loads(txt)

        header_data = data.get("header") or {}
        code = int(header_data.get("code") or 0)
        if code != 0:
            message = header_data.get("message") or "unknown error"
            sid = header_data.get("sid") or ""
            sid_text = f", sid={sid}" if sid else ""
            raise RuntimeError(f"讯飞生图失败：code={code}, message={message}{sid_text}")

        urls: List[str] = []
        choices = ((data.get("payload") or {}).get("choices") or {})
        text_items = choices.get("text") or []
        if isinstance(text_items, list):
            for item in text_items:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str) and content:
                    urls.append(f"data:image/png;base64,{content}")

        return urls, data
