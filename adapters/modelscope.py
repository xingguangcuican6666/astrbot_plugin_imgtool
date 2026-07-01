import aiohttp, json, asyncio
from typing import Any, Dict, List, Tuple
from astrbot.api import logger
from .base import ImageProviderAdapter


class ModelScopeAdapter(ImageProviderAdapter):
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
        # é»˜è®¤ç«¯ç‚¹ï¼šbase_url ç•™ç©ºæ—¶ä½¿ç”¨å®˜æ–¹çš„ç»Ÿä¸€æŽ¨ç† API
        base = (base_url or "https://api-inference.modelscope.cn/v1").rstrip("/")
        url = base + "/images/generations"

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # é‡‡ç”¨å¼‚æ­¥ä»»åŠ¡æ¨¡å¼ï¼Œå†è½®è¯¢ /tasks/{task_id}
            "X-ModelScope-Async-Mode": "true",
        }
        if extra_headers:
            headers.update(extra_headers)

        # ç»„è£…è¯·æ±‚ä½“ï¼ˆéµå¾ªå®˜æ–¹è‰æ¡ˆæ–‡æ¡£ï¼‰
        params: Dict[str, object] = {}
        if image_size:
            params["size"] = image_size
        if batch_size is not None:
            params["n"] = int(batch_size)
        if seed is not None:
            params["seed"] = int(seed)
        if num_inference_steps is not None:
            params["num_inference_steps"] = int(num_inference_steps)
        if guidance_scale is not None:
            params["guidance_scale"] = float(guidance_scale)
        if cfg is not None:
            # æŸäº›æ¨¡åž‹å¯èƒ½æŠŠ cfg è§†ä¸ºé£Žæ ¼/æç¤ºå¼ºåº¦ï¼ŒæŒ‰åŽŸå­—æ®µä¼ é€?
            params["cfg"] = float(cfg)

        payload: Dict[str, object] = {
            "model": model,
            "prompt": prompt,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if params:
            payload["parameters"] = params

        # å‚è€ƒå›¾ï¼šè‰æ¡ˆç¤ºä¾‹ä¸º {"ref_image": {"url": "..."}}
        # æ’ä»¶ä¸Šæ¸¸ä¼šå°½é‡æŠŠæœ¬åœ°/è¿œç¨‹å›¾è½¬æ¢ä¸º data URL ï¼Œè¿™é‡Œç›´æŽ¥ä½œä¸º url ä¼ é€?
        if image:
            payload["ref_image"] = {"url": image}

        logger.info(f"[imgtool] POST {url}")
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=headers, json=payload) as resp:
                txt = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} when POST {url}: {txt}")
                first = json.loads(txt)

        # è‹¥ç›´æŽ¥è¿”å›žåŒæ­¥ç»“æžœï¼ˆå°‘æ•°æ¨¡åž‹å¯èƒ½è¿”å›žï¼‰ï¼Œå°è¯•è§£æžå¸¸è§ç»“æž„
        direct_urls: List[str] = []
        for item in (first.get("images") or first.get("data") or []):
            if isinstance(item, dict):
                u = item.get("url")
                if u:
                    direct_urls.append(u)
                elif "b64_json" in item:
                    direct_urls.append(f"data:image/png;base64,{item['b64_json']}")
        if direct_urls:
            return direct_urls, first

        # å¼‚æ­¥ä»»åŠ¡ï¼šéœ€è¦æœ‰ task_id
        task_id = first.get("task_id") or first.get("id")
        if not task_id:
            # æ—¢æ— åŒæ­¥æ•°æ®ï¼Œåˆæ—  task_id ï¼Œåªèƒ½è¿”å›žåŽŸå§‹å“åº”ä¾¿äºŽæŽ’æŸ?
            logger.error(f"[imgtool] ModelScope unexpected response: {first}")
            return [], first

        # è½®è¯¢ä»»åŠ¡çŠ¶æ€?ï¼šæŒ‰å®˜æ–¹æ ·ä¾‹ï¼Œéœ€æ˜Žç¡® task type
        task_url = f"{base}/tasks/{task_id}"
        poll_headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "X-ModelScope-Task-Type": "image_generation",
        }
        if extra_headers:
            poll_headers.update(extra_headers)

        success_status = {"SUCCEEDED", "SUCCESS", "SUCCEED", "DONE"}
        max_wait_s = 180
        delay = 1.0
        elapsed = 0.0
        last_payload: Dict = {}

        async with aiohttp.ClientSession(timeout=timeout) as sess:
            while elapsed < max_wait_s:
                async with sess.get(task_url, headers=poll_headers) as resp:
                    txt = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status} when GET {task_url}: {txt}")
                    info = json.loads(txt)

                last_payload = info
                status = (info.get("task_status") or info.get("status") or "").upper()

                if status in success_status:
                    urls: List[str] = []

                    # 1) é«˜çº§ AIGC æŽ¥å?£å¸¸è§å­—æ®µï¼šoutput_images
                    output_images = info.get("output_images") or []
                    if isinstance(output_images, list):
                        for o in output_images:
                            if isinstance(o, str):
                                urls.append(o)
                            elif isinstance(o, dict):
                                u = o.get("url") or o.get("image_url") or o.get("image")
                                if isinstance(u, str):
                                    urls.append(u)
                                elif isinstance(u, dict):
                                    uu = u.get("url")
                                    if isinstance(uu, str):
                                        urls.append(uu)

                    # 2) æ™®é€šç»“æž„ï¼šresults / images / data
                    if not urls:
                        results = info.get("results") or []
                        if isinstance(results, list):
                            for r in results:
                                if isinstance(r, dict):
                                    u = r.get("url")
                                    if not u:
                                        # å…¼å®¹ image_url å¯¹è±¡ç»“æž„
                                        obj = r.get("image_url") or r.get("image") or {}
                                        u = obj.get("url") if isinstance(obj, dict) else None
                                    if u:
                                        urls.append(u)

                    if not urls:
                        for item in (info.get("images") or info.get("data") or []):
                            if isinstance(item, dict):
                                u = item.get("url")
                                if u:
                                    urls.append(u)
                                elif "b64_json" in item:
                                    urls.append(f"data:image/png;base64,{item['b64_json']}")

                    return urls, info

                if status in {"FAILED", "ERROR"}:
                    msg = info.get("message") or info.get("error") or "task failed"
                    # æ—¥å¿—ä¿ç•™åŽŸå§‹ payload ï¼Œä¾¿äºŽæŽ’æŸ?
                    logger.error(f"[imgtool] ModelScope task failed: status={status}, payload={info}")
                    raise RuntimeError(f"ModelScope task failed: {msg}")

                await asyncio.sleep(delay)
                elapsed += delay
                # å¢žåŠ è½®è¯¢é—´éš”ï¼Œå°é¡º 3s
                delay = min(3.0, delay + 0.5)

        # è¶…æ—¶
        logger.error(f"[imgtool] ModelScope task timeout: {task_id}")
        return [], last_payload
