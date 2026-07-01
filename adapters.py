# 兼容旧导入路径：从新包 re-export
from .adapters import (
    ImageProviderAdapter,
    CustomHTTPAdapter,
    SiliconFlowAdapter,
    OpenRouterAdapter,
    ModelScopeAdapter,
    GeminiImageAdapter,
    XfyunSparkAdapter,
    get_adapter,
)
