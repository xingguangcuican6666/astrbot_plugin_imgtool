
from .base import ImageProviderAdapter
from .custom_http import OpenAIImagesAdapter as CustomHTTPAdapter
from .siliconflow import SiliconFlowAdapter
from .openrouter import OpenRouterAdapter
from .modelscope import ModelScopeAdapter
from .gemini_image import GeminiImageAdapter
from .doubao_seedream import DoubaoSeedreamAdapter

_REGISTRY: dict[str, ImageProviderAdapter] = {
    "siliconflow": SiliconFlowAdapter(),
    "openrouter": OpenRouterAdapter(),
    "modelscope": ModelScopeAdapter(),
    "gemini-image": GeminiImageAdapter(),
    "doubao": DoubaoSeedreamAdapter(),
    "custom": CustomHTTPAdapter(),
}


def get_adapter(name: str) -> ImageProviderAdapter:
    return _REGISTRY.get((name or "").lower(), CustomHTTPAdapter())
