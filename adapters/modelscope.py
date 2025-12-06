import aiohttp, json, asyncio
from typing import Dict, List, Tuple
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
    ) -> Tuple[List[str], Dict]:
        # 默认端点
        b = (base_url or "https://api-inference.modelscope.cn/v1").rstrip("/")
        url = b + "/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # 采用异步任务模式，随后轮询 /tasks/{task_id}
            "X-ModelScope-Async-Mode": "true",
        }
        if extra_headers:
            headers.update(extra_headers)

        # 组装请求体（遵循给定草案文档）
        params: Dict = {}
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
            # 某些模型可能把 cfg 视为风格/提示强度，按原字段转发
            params["cfg"] = float(cfg)

        payload: Dict = {
            "model": model,
            "prompt": prompt,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if params:
            payload["parameters"] = params

        # 参考图：草案示例为 {"ref_image": {"url": "..."}}
        # 插件上游会尽量把本地/远程图转为 data URL，这里直接作为 url 传递
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

        # 若直接返回同步结果（容错），尝试解析常见结构
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

        # 异步任务：需要有 task_id
        task_id = first.get("task_id") or first.get("id")
        if not task_id:
            # 既无同步数据，又无 task_id，只能返回原始响应便于排查
            logger.error(f"[imgtool] ModelScope unexpected response: {first}")
            return [], first

        # 轮询任务状态
        task_url = f"{b}/tasks/{task_id}"
        poll_headers = {"Authorization": f"Bearer {api_key}"}
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
                if status in {"SUCCEEDED", "SUCCESS", "DONE"}:
                    urls: List[str] = []
                    # 结果可能在 results 数组里，或 data/images
                    results = info.get("results") or []
                    if isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict):
                                u = r.get("url")
                                if not u:
                                    # 兼容 image_url 对象结构
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
                    raise RuntimeError(f"ModelScope task failed: {msg}")
                await asyncio.sleep(delay)
                elapsed += delay
                # 增加轮询间隔，封顶 3s
                delay = min(3.0, delay + 0.5)

        # 超时
        logger.error(f"[imgtool] ModelScope task timeout: {task_id}")
        return [], last_payload

