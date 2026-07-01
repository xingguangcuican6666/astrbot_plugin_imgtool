import aiohttp
import json
from typing import Any, Dict, List, Tuple

from astrbot.api import logger
from .base import ImageProviderAdapter


class DoubaoSeedreamAdapter(ImageProviderAdapter):
    """即梦（Doubao Seedream）图像生成适配器。

    仅负责把插件的通用参数映射到 Doubao 的 CreateImage 接口：
    - base_url 需要在配置里填写为完整的 HTTP 接口地址；
    - api_key 作为 Bearer Token 使用；
    - prompt / image / size / seed / guidance_scale 直接透传。
    """

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
        # base_url 为空时使用 Doubao 官方默认的 CreateImage 接口地址。
        if base_url and base_url.strip():
            url = base_url.strip()
        else:
            url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

        if not api_key:
            raise RuntimeError("Doubao 即梦需要 api_key，请在插件配置中填写。")

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            # Doubao Ark 系列接口通常采用 Bearer Token 认证。
            "Authorization": f"Bearer {api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)

        # 组装 prompt：若有负面提示词，则附加在文本中，保证即便 API 不单独支持 negative_prompt 也能生效。
        text = prompt or ""
        if negative_prompt:
            if text:
                text = f"{text}\n\n[Negative prompt]: {negative_prompt}"
            else:
                text = f"[Negative prompt]: {negative_prompt}"

        payload: Dict[str, object] = {
            "model": model,
            "prompt": text,
        }

        # 分辨率：CreateImage 文档中使用的是 size 字段（字符串）。
        if image_size:
            payload["size"] = str(image_size)

        # 参考图：即梦 image 字段支持 URL 或 Base64，且可为单图或多图。
        imgs: List[str] = []
        for u in (image, image2, image3):
            if isinstance(u, str) and u:
                imgs.append(u)
        if imgs:
            if len(imgs) == 1:
                payload["image"] = imgs[0]
            else:
                payload["image"] = imgs

        # 随机种子与文本权重。
        if seed is not None:
            payload["seed"] = int(seed)
        if guidance_scale is not None:
            payload["guidance_scale"] = float(guidance_scale)

        logger.info(f"[imgtool] POST {url} (doubao-seedream)")
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, json=payload) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} when POST {url}: {txt}")
                data = json.loads(txt)

        # 即梦返回结构与 OpenAI Images 类似：顶层 data 数组，每项包含 url 或 b64_json。
        urls: List[str] = []
        items = data.get("data") or data.get("images") or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                u = item.get("url")
                if isinstance(u, str) and u:
                    urls.append(u)
                    continue
                b64 = item.get("b64_json")
                if isinstance(b64, str) and b64:
                    urls.append(f"data:image/jpeg;base64,{b64}")

        return urls, data
