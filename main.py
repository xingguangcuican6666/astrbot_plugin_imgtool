from __future__ import annotations
import os
import re
import uuid
import aiohttp
import base64
import mimetypes
from urllib.parse import urlparse
from typing import Any, List

import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger as logger

from .adapters import ImageProviderAdapter
from .adapters import get_adapter
from .tools import ImagineTool

PLUGIN_ID = "astrbot_plugin_imgtool"
XFYUN_SUPPORTED_SIZES = {
    "768x768",
    "1024x1024",
    "576x1024",
    "768x1024",
    "1024x576",
    "1024x768",
}


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

        # 迁移旧版平铺配置到按 provider 分组的配置里（兼容已有用户）
        try:
            self._migrate_legacy_config()
        except Exception as e:
            logger.error(f"[{PLUGIN_ID}] migrate config failed: {e}", exc_info=True)

        # 注册 LLM Tool：imagine（供 LLM 调用的图片生成工具）
        try:
            self.imagine_tool = ImagineTool(plugin=self)
            self.context.add_llm_tools(self.imagine_tool)
            logger.info(f"[{PLUGIN_ID}] LLM tool 'imagine' registered via add_llm_tools")
        except Exception as e:
            # 兼容旧版本：直接往 llm_tools.func_list 里追加
            try:
                self.imagine_tool = ImagineTool(plugin=self)
                tool_mgr = self.context.provider_manager.llm_tools
                tool_mgr.func_list.append(self.imagine_tool)
                logger.info(f"[{PLUGIN_ID}] LLM tool 'imagine' registered via legacy func_list")
            except Exception as ee:
                logger.error(
                    f"[{PLUGIN_ID}] register LLM tool 'imagine' failed: add_llm_tools={e}; legacy={ee}",
                    exc_info=True,
                )

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
    def _current_provider(self) -> str:
        return (self.config.get("provider") or "siliconflow").lower()

    def _get_provider_config(self, provider: str | None = None) -> dict:
        if provider is None:
            provider = self._current_provider()
        cfg = self.config.get(provider) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _prepare_flat_config_for_provider(self, provider: str | None = None) -> None:
        """将按 provider 分组的配置摊平到顶层，兼容旧版读取方式。"""
        if provider is None:
            provider = self._current_provider()
        cfg = self._get_provider_config(provider)
        if not isinstance(cfg, dict) or not cfg:
            return

        def _set_if_present(key: str, cfg_key: str | None = None) -> None:
            k = cfg_key or key
            if k in cfg and cfg[k] not in (None, ""):
                self.config[key] = cfg[k]

        _set_if_present("api_key")
        _set_if_present("model")
        _set_if_present("edit_model")
        if provider == "custom":
            _set_if_present("base_url")
        if provider == "gemini-image":
            _set_if_present("base_url")
        if provider == "xfyun":
            _set_if_present("base_url")
            _set_if_present("app_id")
            _set_if_present("api_secret")
            _set_if_present("uid")
            _set_if_present("scheduler")
            _set_if_present("patch_id")
        if provider == "openrouter":
            _set_if_present("openrouter_referer", "referer")
            _set_if_present("openrouter_title", "title")

    def _refresh_imagine_tool(self) -> None:
        tools = []
        imagine_tool = getattr(self, "imagine_tool", None)
        if imagine_tool is not None:
            tools.append(imagine_tool)

        try:
            tool_mgr = self.context.provider_manager.llm_tools
            func_list = getattr(tool_mgr, "func_list", None)
            if isinstance(func_list, list):
                for tool in func_list:
                    if getattr(tool, "name", "") == "imagine" and getattr(tool, "plugin", None) is self:
                        tools.append(tool)
        except Exception:
            pass

        seen = set()
        for tool in tools:
            if id(tool) in seen:
                continue
            seen.add(id(tool))
            refresh = getattr(tool, "refresh_schema", None)
            if not callable(refresh):
                continue
            try:
                refresh()
            except Exception as e:
                logger.error(f"[{PLUGIN_ID}] refresh imagine tool schema failed: {e}", exc_info=True)

    def _migrate_legacy_config(self) -> None:
        """把旧版全局配置转换为按 provider 分组的配置。"""
        changed = False

        provider = self._current_provider()
        legacy_api_key = (self.config.get("api_key") or "").strip()
        legacy_model = (self.config.get("model") or "").strip()
        legacy_edit_model = (self.config.get("edit_model") or "").strip()

        if provider in {"siliconflow", "openrouter", "modelscope", "gemini-image", "doubao", "xfyun", "custom"}:
            section = self._get_provider_config(provider)
            if legacy_api_key and not section.get("api_key"):
                section["api_key"] = legacy_api_key
                changed = True
            if legacy_model and not section.get("model"):
                section["model"] = legacy_model
                changed = True
            if legacy_edit_model and not section.get("edit_model") and provider in {"siliconflow", "openrouter", "custom"}:
                section["edit_model"] = legacy_edit_model
                changed = True
            self.config[provider] = section

        legacy_app_id = (self.config.get("app_id") or "").strip()
        legacy_api_secret = (self.config.get("api_secret") or "").strip()
        if provider == "xfyun":
            section = self._get_provider_config("xfyun")
            if legacy_app_id and not section.get("app_id"):
                section["app_id"] = legacy_app_id
                changed = True
            if legacy_api_secret and not section.get("api_secret"):
                section["api_secret"] = legacy_api_secret
                changed = True
            self.config["xfyun"] = section

        legacy_ref = (self.config.get("openrouter_referer") or "").strip()
        legacy_title = (self.config.get("openrouter_title") or "").strip()
        if legacy_ref or legacy_title:
            openrouter_cfg = self._get_provider_config("openrouter")
            if legacy_ref and not openrouter_cfg.get("referer"):
                openrouter_cfg["referer"] = legacy_ref
                changed = True
            if legacy_title and not openrouter_cfg.get("title"):
                openrouter_cfg["title"] = legacy_title
                changed = True
            self.config["openrouter"] = openrouter_cfg

        if changed and hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
                logger.info(f"[{PLUGIN_ID}] legacy config migrated to per-provider sections")
            except Exception as e:
                logger.error(f"[{PLUGIN_ID}] save migrated config failed: {e}", exc_info=True)

    def _xfyun_provider_options(self) -> dict[str, Any]:
        provider_cfg = self._get_provider_config("xfyun")
        width = 1024
        height = 1024
        default_size = str(self.config.get("defaults", {}).get("image_size") or "1024x1024").strip().lower()
        if default_size in XFYUN_SUPPORTED_SIZES:
            left, right = default_size.split("x", 1)
            width, height = int(left), int(right)
        return {
            "app_id": (self.config.get("app_id") or provider_cfg.get("app_id") or "").strip(),
            "api_secret": (self.config.get("api_secret") or provider_cfg.get("api_secret") or "").strip(),
            "uid": (self.config.get("uid") or provider_cfg.get("uid") or "").strip(),
            "patch_id": self.config.get("patch_id") or provider_cfg.get("patch_id") or "",
            "scheduler": (self.config.get("scheduler") or provider_cfg.get("scheduler") or "DPM++ 2M Karras").strip(),
            "width": width,
            "height": height,
        }

    # ------------- 公共工具方法 -------------
    def _pick_adapter(self) -> ImageProviderAdapter:
        return get_adapter(self._current_provider())

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

        provider = self._current_provider()
        self._prepare_flat_config_for_provider(provider)
        provider = (self.config.get("provider") or "siliconflow").lower()
        api_key = self.config.get("api_key", "").strip()
        provider_options: dict[str, Any] | None = None
        if provider == "xfyun":
            provider_options = self._xfyun_provider_options()
            app_id = str(provider_options.get("app_id") or "").strip()
            api_secret = str(provider_options.get("api_secret") or "").strip()
            if not app_id or not api_key or not api_secret:
                return event.plain_result("请先在讯飞配置中填写 app_id、api_key 和 api_secret。")
        elif not api_key:
            return event.plain_result("请先在插件配置中填写 api_key。")

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
            if provider == "xfyun":
                model = (self.config.get("model") or "").strip()
            else:
                model = (self.config.get("model") or "Qwen/Qwen-Image").strip()
        if not size:
            size = (self.config.get("defaults", {}).get("image_size") or "1024x1024").strip()
        if steps is None:
            steps = int(self.config.get("defaults", {}).get("num_inference_steps", 20))
        if guidance_scale is None:
            guidance_scale = float(self.config.get("defaults", {}).get("guidance_scale", 7.5))
        if batch_size is None:
            batch_size = int(self.config.get("defaults", {}).get("batch_size", 1))
        if provider == "xfyun":
            batch_size = 1
            if not model:
                return event.plain_result("请先在讯飞配置中填写 model（讯飞控制台中的 modelID/domain）。")

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
                provider_options=provider_options,
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

    # ------------- 指令：/imgprovider -------------
    @filter.command("imgprovider")
    async def imgprovider(self, event: AstrMessageEvent, provider: str = ""):
        """/imgprovider [provider]  查看或切换文生图 provider。

        不带参数：显示当前 provider 和可选列表；
        带参数：切换 provider 并保存配置（同时保留各自的 api_key 和模型配置）。
        """
        provider = (provider or "").strip().lower()
        valid = ["siliconflow", "openrouter", "modelscope", "gemini-image", "doubao", "xfyun", "custom"]

        if not provider:
            current = self._current_provider()
            choices = ", ".join(valid)
            yield event.plain_result(f"当前文生图 provider：{current}\n可选：{choices}")
            return

        if provider not in valid:
            choices = ", ".join(valid)
            yield event.plain_result(f"无效 provider：{provider}\n可选：{choices}")
            return

        self.config["provider"] = provider
        # 切换后即刻按新 provider 摊平一次，方便后续使用
        self._prepare_flat_config_for_provider(provider)
        self._refresh_imagine_tool()

        if hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
            except Exception as e:
                logger.error(f"[{PLUGIN_ID}] save provider config failed: {e}", exc_info=True)
                yield event.plain_result(f"已切换为 {provider}，但保存配置失败：{e}")
                return

        yield event.plain_result(f"已将文生图 provider 切换为：{provider}")

    # ------------- LLM 工具实现：imagine -------------
    # 注意：不再通过装饰器直接注册为 LLM 工具，而是由 ImagineTool 调用此方法。
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
    ) -> str:
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
            cfg(number): 仅 Qwen-Image 支持；讯飞不支持
            image(string): 参考图（base64 或 URL；仅当前 provider 支持时可用）
            image2(string): 参考图2（仅 Qwen-Image-Edit-2509 且当前 provider 支持时可用）
            image3(string): 参考图3（仅 Qwen-Image-Edit-2509 且当前 provider 支持时可用）
            use_refs(boolean): 是否自动使用本条消息里的图片/被引用消息里的图片作为参考图（仅当前 provider 支持时可用）
        """
        provider = self._current_provider()
        self._prepare_flat_config_for_provider(provider)
        # 1) 解析最终要用的模型名（如果函数参数没给，就用配置里的）
        if provider == "xfyun":
            final_model = (model or self.config.get("model") or "").strip()
        else:
            final_model = (model or self.config.get("model") or "Qwen/Qwen-Image").strip()
        # 允许使用别名：例如传入 "edit" 自动映射为配置的编辑模型
        alias = final_model.lower()
        if provider == "xfyun" and alias in {"edit", "qwen-edit", "qwen_image_edit", "qwen-image-edit"}:
            mer = event.plain_result("讯飞 provider 暂不支持编辑模型别名，请直接填写讯飞控制台中的文生图 modelID/domain。")
            await event.send(mer)
            return mer.get_plain_text()
        if alias in {"edit", "qwen-edit", "qwen_image_edit", "qwen-image-edit"}:
            fm = (self.config.get("edit_model") or "Qwen/Qwen-Image-Edit").strip()
            if fm:
                final_model = fm
        m = final_model.lower()

        if provider == "xfyun":
            if image or image2 or image3 or use_refs:
                mer = event.plain_result("讯飞 provider 暂不支持参考图或 use_refs。")
                await event.send(mer)
                return mer.get_plain_text()
            if cfg:
                mer = event.plain_result("讯飞 provider 暂不支持 cfg 参数。")
                await event.send(mer)
                return mer.get_plain_text()
            if not final_model:
                mer = event.plain_result("讯飞 provider 需要配置 model（讯飞控制台中的 modelID/domain）。")
                await event.send(mer)
                return mer.get_plain_text()
            sz = (size or self.config.get("defaults", {}).get("image_size") or "1024x1024").strip().lower()
            if sz not in XFYUN_SUPPORTED_SIZES:
                choices = ", ".join(sorted(XFYUN_SUPPORTED_SIZES))
                mer = event.plain_result(f"讯飞 provider 仅支持以下分辨率：{choices}")
                await event.send(mer)
                return mer.get_plain_text()
            batch_size = 1

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
                mer = event.plain_result("使用编辑模型需要参考图：请附加图片或把 use_refs 设为 true。")
                await event.send(mer)
                return mer.get_plain_text()
            # 接受 data URL 或 http(s) 直链，其他情况视为无效
            if not (isinstance(im1, str) and (im1.startswith("data:image/") or im1.startswith("http://") or im1.startswith("https://"))):
                mer = event.plain_result("参考图格式不支持，请上传图片或提供可直链 URL。")
                await event.send(mer)
                return mer.get_plain_text()

        mer = await self._generate_and_reply(
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
        await event.send(mer)
        return mer.get_plain_text()
