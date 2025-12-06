
from typing import Dict, List, Tuple
from astrbot.api import logger
from .custom_http import OpenAIImagesAdapter


class SiliconFlowAdapter(OpenAIImagesAdapter):
    async def generate_image(self, **kwargs):
        # 默认端点：忽略空 base_url
        if not kwargs.get("base_url"):
            kwargs["base_url"] = "https://api.siliconflow.cn/v1"

        # 修正：不再把 Qwen 系列映射到 SDXL。
        # 仅在遗漏组织前缀时，自动补齐为 "Qwen/..."，以匹配硅基流动的模型命名。
        model = (kwargs.get("model") or "").strip()
        ml = model.lower()
        if model and "/" not in model:
            # 用户可能传入了 "Qwen-Image-Edit-2509" 这样的短名
            if ml.startswith("qwen-"):
                fixed = f"Qwen/{model}"
                logger.info(f"[imgtool] siliconflow: normalize model '{model}' -> '{fixed}'")
                kwargs["model"] = fixed

        return await super().generate_image(**kwargs)
