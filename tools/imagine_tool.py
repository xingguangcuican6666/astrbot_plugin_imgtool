from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrbot.api import FunctionTool
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@dataclass
class ImagineTool(FunctionTool):
    """
    文生图工具：调用插件的 imagine 方法生成图片。

    - 会实际向用户发送生成的图片消息
    - 同时将文本摘要返回给 LLM，用于后续对话
    """

    plugin: Any | None = None
    name: str = "imagine"
    description: str = "生成图片。根据提示词和可选参考图生成图片，并将结果发送给用户。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "文生图提示词",
                },
                "size": {
                    "type": "string",
                    "description": "图像分辨率，形如 '宽x高'，例如 '1024x1024'",
                },
                "model": {
                    "type": "string",
                    "description": "模型名称（为空则使用插件配置中的默认模型）",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "反向提示词，用于指定不希望出现的内容",
                },
                "steps": {
                    "type": "number",
                    "description": "推理步数，范围约 1-100",
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "提示词遵循度，范围约 0-20",
                },
                "seed": {
                    "type": "number",
                    "description": "随机种子，范围约 -9999999999 到 9999999999",
                },
                "batch_size": {
                    "type": "number",
                    "description": "生成数量，范围约 1-4",
                },
                "cfg": {
                    "type": "number",
                    "description": "仅 Qwen-Image 支持的 CFG 参数",
                },
                "image": {
                    "type": "string",
                    "description": "参考图（base64 或 URL）",
                },
                "image2": {
                    "type": "string",
                    "description": "参考图2（仅 Qwen-Image-Edit-2509 使用）",
                },
                "image3": {
                    "type": "string",
                    "description": "参考图3（仅 Qwen-Image-Edit-2509 使用）",
                },
                "use_refs": {
                    "type": "boolean",
                    "description": "是否自动使用本条消息或被引用消息中的图片作为参考图",
                },
            },
            "required": ["prompt"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> str:
        """通过插件的 imagine 方法执行实际生成逻辑，并返回给 LLM 的文本摘要。"""
        if self.plugin is None:
            raise ValueError("ImagineTool.plugin is not set.")

        agent_ctx = context.context
        event: AstrMessageEvent = agent_ctx.event

        # 直接复用插件中已经实现好的 imagine 逻辑：
        # - 内部会负责调用 _generate_and_reply 发送图片给用户
        # - 并返回给 LLM 用于总结的纯文本
        return await self.plugin.imagine(
            event,
            prompt=kwargs.get("prompt", ""),
            size=kwargs.get("size", ""),
            model=kwargs.get("model", ""),
            negative_prompt=kwargs.get("negative_prompt", ""),
            steps=kwargs.get("steps", 20),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            seed=kwargs.get("seed", 0),
            batch_size=kwargs.get("batch_size", 1),
            cfg=kwargs.get("cfg", 0.0),
            image=kwargs.get("image", ""),
            image2=kwargs.get("image2", ""),
            image3=kwargs.get("image3", ""),
            use_refs=kwargs.get("use_refs", False),
        )

