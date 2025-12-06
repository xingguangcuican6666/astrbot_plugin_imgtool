from __future__ import annotations
import os
import re
import uuid
import aiohttp
import asyncio
import base64
import mimetypes
from urllib.parse import urlparse
from typing import List

import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger as logger
from astrbot.api.all import llm_tool # 若导入失败，可改成 from astrbot.api.all import llm_tool

from .adapters import ImageProviderAdapter
from .adapters import get_adapter

PLUGIN_ID = "astrbot_plugin_imgtool"


def _find_data_root(start_dir: str) -> str:
    """尽量将持久化文件放到 AstrBot/data 下；找不到就退回到插件目录的 _data。"""
    cur = os.path.abspath(start_dir)
    for _ in range(5):
        parent = os.path.dirname(cur)
        if os.path.basename(cur) == "data":
            return cur
        cur = parent
    # fallback: 插件目录下
    fallback = os.path.join(start_dir, "_data")
    os.makedirs(fallback, exist_ok=True)
    return fallback


@register(PLUGIN_ID, "your_name", "图像生成 LLM 工具（支持硅基流动 / 可扩展）", "0.1.0")
class ImgToolPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config


        plugin_dir = os.path.dirname(__file__)
        self.data_root = _find_data_root(plugin_dir)
        self.save_dir = os.path.join(self.data_root, self.config.get("save_dir", "images/astrbot_plugin_imgtool"))
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"[{PLUGIN_ID}] save_dir = {self.save_dir}")

    # --- 工具：把本地/HTTP(S) 图片转成 PNG 并内联为 data URL ---
    async def _to_data_url(self, src: str) -> str | None:
        if not src:
            return None
        s = src.strip()
        if s.startswith("data:image/"):
            return s
        # 本地文件 -> 尽量转成 PNG
        if "://" not in s:
            try:
                try:
                    from PIL import Image  # type: ignore
                    from io import BytesIO
                    with Image.open(s) as im:
                        if im.mode not in ("RGB", "RGBA"):
                            im = im.convert("RGBA")
                        buf = BytesIO()
                        im.save(buf, format="PNG")
                        raw = buf.getvalue()
                    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"
                except Exception:
                    # Pillow 不可用或失败则退回原始字节
                    with open(s, "rb") as f:
                        raw = f.read()
                    ctype = mimetypes.guess_type(s)[0] or "image/png"
                    return f"data:{ctype};base64,{base64.b64encode(raw).decode()}"
            except Exception as e:
                logger.error(f"[imgtool] read local image failed: {s} -> {e}", exc_info=True)
                return None
        # 远程 URL
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            # 通用头
            headers = {"User-Agent": "Mozilla/5.0 AstrBot-ImgTool/0.2"}
            # 针对常见防盗链域名补 Referer
            host = urlparse(s).hostname or ""
            if any(d in host for d in ("pstatp.com", "toutiaoimg.com", "byteimg.com")):
                headers["Referer"] = "https://www.toutiao.com/"
            # 360/奇虎图片 CDN 有时对直链返回 404/403，补上站内 Referer
            if "qhimg.com" in host:
                headers["Referer"] = "https://www.360.cn/"
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as sess:
                async with sess.get(s, allow_redirects=True) as resp:
                    if resp.status >= 400:
                        logger.error(f"[imgtool] fetch remote image failed: {s} -> HTTP {resp.status}")
                        return None
                    raw = await resp.read()
                    ctype = resp.headers.get("Content-Type") or "image/jpeg"
            # 尝试用 Pillow 统一转成 PNG（避免 webp 不稳）
            try:
                from PIL import Image  # type: ignore
                from io import BytesIO
                buf_in = BytesIO(raw)
                with Image.open(buf_in) as im:
                    if im.mode not in ("RGB", "RGBA"):
                        im = im.convert("RGBA")
                    buf_out = BytesIO()
                    im.save(buf_out, format="PNG")
                    raw_png = buf_out.getvalue()
                data_url = f"data:image/png;base64,{base64.b64encode(raw_png).decode()}"
                logger.info(f"[imgtool] fetched remote image -> data:image/png size={len(raw)}")
                return data_url
            except Exception:
                pass
            if not ctype.startswith("image/"):
                guess = mimetypes.guess_type(urlparse(s).path)[0]
                ctype = guess if (guess and guess.startswith("image/")) else "image/jpeg"
            data_url = f"data:{ctype};base64,{base64.b64encode(raw).decode()}"
            logger.info(f"[imgtool] fetched remote image -> data:{ctype} size={len(raw)}")
            return data_url
        except Exception as e:
            logger.error(f"[imgtool] fetch remote image failed: {s} -> {e}", exc_info=True)
            return None

    # 从当前消息事件里“捞”参考图（支持本条消息内的图片；引用消息里若有图片做最佳努力提取）
    def _collect_refs_from_event(self, event: AstrMessageEvent, max_n: int = 3) -> list[str]:
        imgs: list[str] = []
        try:
            comps = getattr(event.message_obj, "message", [])  # 消息链：List[组件]
            for seg in comps:
                cls = seg.__class__.__name__.lower()
                # 直接图片段（跨平台统一是 Image 组件）
                if "image" in cls:
                    u = getattr(seg, "url", None) or getattr(seg, "file", None) or getattr(seg, "path", None)
                    if isinstance(u, str) and u:
                        imgs.append(u)
                # 引用段里可能嵌着原消息的图片（不同平台字段不完全一致，尽力取出）
                if "reply" in cls:
                    for key in ("content", "message", "message_chain"):
                        inner = getattr(seg, key, None)
                        if isinstance(inner, list):
                            for s in inner:
                                if s.__class__.__name__.lower().find("image") >= 0:
                                    u = getattr(s, "url", None) or getattr(s, "file", None) or getattr(s, "path", None)
                                    if isinstance(u, str) and u:
                                        imgs.append(u)
                if len(imgs) >= max_n:
                    break
        except Exception as e:
            logger.error(f"collect refs failed: {e}", exc_info=True)
        return imgs[:max_n]

    # ------------- 公共工具方法 -------------
    def _pick_adapter(self) -> ImageProviderAdapter:
        provider = (self.config.get("provider") or "siliconflow").lower()
        return get_adapter(provider)

    def _platform_url_only(self, event: AstrMessageEvent) -> bool:
        # 钉钉仅支持 URL 图（其他平台多数均支持本地文件 & URL）。
        name = (event.get_platform_name() or "").lower()
        return name in {"dingtalk", "dingding"}

    async def _download_all(self, urls: List[str]) -> List[str]:
        saved: List[str] = []
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            for i, u in enumerate(urls):
                try:
                    if u.startswith("data:image/"):
                        # data URL -> 文件
                        head, b64 = u.split(",", 1)
                        ext = ".png"
                        if "image/jpeg" in head:
                            ext = ".jpg"
                        fname = uuid.uuid4().hex + ext
                        path = os.path.join(self.save_dir, fname)
                        with open(path, "wb") as f:
                            import base64
                            f.write(base64.b64decode(b64))
                        saved.append(path)
                        continue

                    async with sess.get(u) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"GET {u} -> {resp.status}")
                        data = await resp.read()
                        ext = ".png"
                        ctype = resp.headers.get("Content-Type", "")
                        if "jpeg" in ctype:
                            ext = ".jpg"
                        fname = uuid.uuid4().hex + ext
                        path = os.path.join(self.save_dir, fname)
                        with open(path, "wb") as f:
                            f.write(data)
                        saved.append(path)
                except Exception as e:
                    logger.error(f"download image failed: {u} -> {e}", exc_info=True)
        return saved

    async def _generate_and_reply(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        cfg: float | None = None,
        image: str | None = None,
        image2: str | None = None,
        image3: str | None = None,
    ) -> MessageEventResult:
        if not prompt or not prompt.strip():
            return event.plain_result("提示词为空。")

        api_key = self.config.get("api_key", "").strip()
        if not api_key:
            return event.plain_result("请先在插件配置中填写 api_key。")

        provider = (self.config.get("provider") or "siliconflow").lower()
        # 交给各 adapter 决定默认 base_url；仅 custom 严格校验完整端点（你的原始约束保留）
        extra_headers = None
        if provider == "openrouter":
            extra_headers = {}
            ref = (self.config.get("openrouter_referer") or "").strip()
            ttl = (self.config.get("openrouter_title") or "").strip()
            if ref: extra_headers["HTTP-Referer"] = ref
            if ttl: extra_headers["X-Title"] = ttl
        base_url = (self.config.get("base_url") or "").strip()
        if provider == "custom":
            b = base_url.rstrip("/")
            if not b or (not b.endswith("/images/generations")):
                return event.plain_result("base_url 必须以 /v1/images/generations 结尾（完整接口地址）。")
        # siliconflow/openrouter 留空 base_url 让各自 adapter 用默认
        if provider in {"siliconflow", "openrouter"}:
            base_url = base_url  # 可以留空，adapter 会兜底

        if not model:
            model = (self.config.get("model") or "Qwen/Qwen-Image").strip()
        if not size:
            size = (self.config.get("defaults", {}).get("image_size") or "1024x1024").strip()
        if steps is None:
            steps = int(self.config.get("defaults", {}).get("num_inference_steps", 20))
        if guidance_scale is None:
            guidance_scale = float(self.config.get("defaults", {}).get("guidance_scale", 7.5))
        if batch_size is None:
            batch_size = int(self.config.get("defaults", {}).get("batch_size", 1))

        adapter = self._pick_adapter()
        
        try:
            urls, raw = await adapter.generate_image(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_size=size,
                batch_size=batch_size,
                seed=seed,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                cfg=cfg,
                image=image,
                image2=image2,
                image3=image3,
                extra_headers=extra_headers,
            )
        except Exception as e:
            logger.error("image generation failed", exc_info=True)
            return event.plain_result(f"生成失败：{e}")

        # 按平台能力发送：优先本地文件，其次 URL（钉钉仅 URL）。
        if self._platform_url_only(event):
            chain = [Comp.Plain(f"已生成 {len(urls)} 张图（URL 有效期约 1h）：")] + [Comp.Image.fromURL(u) for u in urls]
            return event.chain_result(chain)

        saved_paths = await self._download_all(urls)
        if saved_paths:
            chain = [Comp.Plain(f"已生成 {len(saved_paths)} 张图：")] + [Comp.Image.fromFileSystem(p) for p in saved_paths]
            return event.chain_result(chain)
        else:
            # 兜底：仍发送 URL
            chain = [Comp.Plain(f"已生成 {len(urls)} 张图（未能保存本地）：")] + [Comp.Image.fromURL(u) for u in urls]
            return event.chain_result(chain)

    # ------------- 指令：/img -------------
    @filter.command("img")
    async def img(self, event: AstrMessageEvent, prompt: str):
        """/img <提示词>  直接生成图片（使用配置默认参数）。"""
        yield await self._generate_and_reply(event, prompt=prompt)

    # ------------- LLM 工具：imagine -------------
    @llm_tool(name="imagine")
    async def imagine(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "",
        model: str = "",
        negative_prompt: str = "",
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 0,
        batch_size: int = 1,
        cfg: float = 0.0,
        image: str = "",
        image2: str = "",
        image3: str = "",
        use_refs: bool = False,
    ) -> MessageEventResult:
        """生成图片。

        Args:
            prompt(string): 文生图提示词
            size(string): 图像分辨率，形如 "宽x高"，如 "1024x1024"
            model(string): 模型名称（为空则用插件配置）
            negative_prompt(string): 反向提示词
            steps(number): 推理步数（1-100）
            guidance_scale(number): 提示词遵循度（0-20）
            seed(number): 随机种子（0-9999999999）
            batch_size(number): 生成数量（1-4）
            cfg(number): 仅 Qwen-Image 支持的 CFG
            image(string): 参考图（base64 或 URL）
            image2(string): 参考图2（仅 Qwen-Image-Edit-2509）
            image3(string): 参考图3（仅 Qwen-Image-Edit-2509）
            use_refs(boolean): 是否自动使用本条消息里的图片/被引用消息里的图片作为参考图
        """
        # 1) 解析最终要用的模型名（如果函数参数没给，就用配置里的）
        final_model = (model or self.config.get("model") or "Qwen/Qwen-Image").strip()
        # 允许使用别名：例如传入 "edit" 自动映射为配置的编辑模型
        alias = final_model.lower()
        if alias in {"edit", "qwen-edit", "qwen_image_edit", "qwen-image-edit"}:
            fm = (self.config.get("edit_model") or "Qwen/Qwen-Image-Edit").strip()
            if fm:
                final_model = fm
        m = final_model.lower()

        # 2) 可选：自动从消息里采集参考图（消息链里就能拿到图片段）
        im1, im2, im3 = image or None, image2 or None, image3 or None
        if use_refs and not im1:
            refs = self._collect_refs_from_event(event, max_n=3)
            if refs:
                im1 = refs[0]
                if len(refs) > 1: im2 = refs[1]
                if len(refs) > 2: im3 = refs[2]
                logger.info(f"[imgtool] use_refs collected {len(refs)} refs: {refs}")

        # 3) 将参考图统一转换为 data URL（无论显式传还是 use_refs）
        #    —— 这是关键修复，避免 provider 去拉外链导致 500/防盗链
        async def _norm(u: str | None) -> str | None:
            if not u:
                return None
            if isinstance(u, str) and u.startswith("data:image/"):
                return u
            # 远程 URL：尝试抓取为 data URL，失败则回退为原始 URL（交给 Provider 侧拉取）
            if isinstance(u, str) and u.startswith(("http://", "https://")):
                du = await self._to_data_url(u)
                return du
            # 本地文件：只能转 data URL，失败返回 None
            return await self._to_data_url(u)
        im1 = await _norm(im1)
        im2 = await _norm(im2)
        im3 = await _norm(im3)
        # 若转换失败且启用了 use_refs，则从消息链回退拿图
        if use_refs and not im1:
            refs = self._collect_refs_from_event(event, max_n=3)
            if refs:
                cand1 = await _norm(refs[0])
                cand2 = await _norm(refs[1]) if len(refs) > 1 and not im2 else im2
                cand3 = await _norm(refs[2]) if len(refs) > 2 and not im3 else im3
                im1, im2, im3 = cand1 or im1, cand2, cand3
                logger.info(f"[imgtool] use_refs collected {len(refs)} refs: {refs}")
        # 打点，方便排查（只打印长度，避免刷日志）
        for tag, val in (("image", im1), ("image2", im2), ("image3", im3)):
            if val:
                logger.info(f"[imgtool] {tag}=data:... len={len(val)}")

        # 4) Edit 系列必须要参考图，做友好校验
        if ("image-edit" in m):
            if not im1:
                yield event.plain_result("使用编辑模型需要参考图：请附加图片或把 use_refs 设为 true。")
                return
            # 接受 data URL 或 http(s) 直链，其他情况视为无效
            if not (isinstance(im1, str) and (im1.startswith("data:image/") or im1.startswith("http://") or im1.startswith("https://"))):
                yield event.plain_result("参考图格式不支持，请上传图片或提供可直链 URL。")
                return

        yield await self._generate_and_reply(
            event,
            prompt=prompt,
            model=final_model,
            size=size or None,
            negative_prompt=negative_prompt or None,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed or None,
            batch_size=batch_size,
            cfg=cfg or None,
            image=im1,
            image2=im2,
            image3=im3,
        )
