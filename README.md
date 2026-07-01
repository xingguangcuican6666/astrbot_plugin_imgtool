# astrbot_plugin_imgtool

一款 AstrBot 插件 —— 在插件配置面板暴露 `provider/base_url/model/api_key`，`provider` 使用下拉列表（siliconflow/openrouter/modelscope/gemini-image/doubao/xfyun/custom）；暴露 LLM 函数工具 `imagine`，可被函数调用（function-calling）自动触发；也可手动命令 `/img` 使用。

已适配提供商：

- SiliconFlow：OpenAI 风格 `/v1/images/generations`
- OpenRouter：OpenAI 风格 `/v1/images/generations`
- ModelScope（魔搭社区）：异步提交 `/v1/images/generations` + 轮询 `/v1/tasks/{task_id}`，支持参考图字段 `ref_image.url`
- Gemini Image：Google Gemini 图像生成接口
- Doubao：火山引擎即梦 CreateImage 接口
- 讯飞 Spark：`/v2.1/tti`，使用 `app_id + api_key + api_secret` 做 HTTP URL 鉴权

适配层解耦，后续按需扩展更多 Provider。支持在函数调用里传 `model=edit` 作为别名（将按配置项 `edit_model` 映射到实际模型，例如 `Qwen/Qwen-Image-Edit`）。参考图既可以 data URL，也可直接用 http(s) 直链；直链会在本地尝试抓取，失败则回退为把 URL 直接交给服务端拉取。为避免 Provider 返回的临时 URL 过期，插件会主动把图片下载保存到 `AstrBot/data/<save_dir>` 下。

## 安装

将本目录放到 `AstrBot/data/plugins/astrbot_plugin_imgtool`，然后在 WebUI 插件管理中启用并配置。

## 使用

- 指令：`/img 一个小岛，海鸥与灯塔……`
- LLM 工具：在对话中触发函数调用 `imagine`（参数同 README 顶部说明）。当以工具方式调用成功后，将向 Bot 返回包含 base64 图片的结果（`{"images": ["data:image/png;base64,...", ...]}`）。

### 使用 ModelScope（魔搭）

- 配置 `provider=modelscope`
- 可留空 `base_url`，适配器默认使用 `https://api-inference.modelscope.cn/v1`
- 会以异步方式提交任务（请求头 `X-ModelScope-Async-Mode: true`），再轮询 `/v1/tasks/{task_id}` 获取结果
- 参考图通过 `ref_image.url` 传入（若模型支持），建议使用公网可访问 URL；也支持 data URL

### 使用讯飞 Spark

- 配置 `provider=xfyun`
- 需要填写 `app_id`、`api_key`、`api_secret` 和 `model`
- `model` 对应讯飞控制台里的 `modelID/domain`
- 如控制台或接口返回要求 `patch_id`，请填写对应模型服务卡片里的 `resourceId`；多个值可用逗号分隔
- 可留空 `base_url`，适配器默认使用 `https://maas-api.cn-huabei-1.xf-yun.com/v2.1/tti`
- 支持 `negative_prompt`
- 支持的分辨率：`768x768`、`1024x1024`、`576x1024`、`768x1024`、`1024x576`、`1024x768`
- `batch_size` 在讯飞下会强制为 `1`
- 讯飞当前不支持参考图和 `cfg`

## 自定义 Provider

- 将 `provider` 设为 `custom`，配上自定义的 `base_url` 与 `api_key`；只要遵循 OpenAI 风格的 `/images/generations` JSON 即可（字段见 `adapters/*.py`）。
