
from .base import ImageProviderAdapter
from .custom_http import OpenAIImagesAdapter as CustomHTTPAdapter
from .siliconflow import SiliconFlowAdapter
from .openrouter import OpenRouterAdapter
from .modelscope import ModelScopeAdapter

_REGISTRY: dict[str, ImageProviderAdapter] = {
    "siliconflow": SiliconFlowAdapter(),
    "openrouter": OpenRouterAdapter(),
    "modelscope": ModelScopeAdapter(),
    "custom": CustomHTTPAdapter(),
}

def get_adapter(name: str) -> ImageProviderAdapter:
    return _REGISTRY.get((name or "").lower(), CustomHTTPAdapter())
