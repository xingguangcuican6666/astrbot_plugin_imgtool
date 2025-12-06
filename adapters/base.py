
from typing import Dict, List, Tuple

class ImageProviderAdapter:
    """统一图像生成接口，便于扩展不同 Provider。"""
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
        raise NotImplementedError
