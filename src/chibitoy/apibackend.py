# 云端图片生成 API 后端
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from chibitoy import config
from chibitoy.apiconfig import ApiConfig, ConfigError, ProviderConfig, load as load_config
from chibitoy.client import GenerationResult

class ApiError(RuntimeError):
    """云端 API 调用失败（可读消息，不包含密钥）"""


class ApiAuthError(ApiError):
    """鉴权失败：Key 错误、过期或无权限"""


class ApiRateLimited(ApiError):
    """被限流（HTTP 429 或厂商错误码）"""


class ApiTimeout(ApiError):
    """轮询超过 ``poll_timeout_seconds``"""


class ApiPostSubmitError(ApiError):
    """请求已提交之后才出现的失败"""


def _is_probably_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def _is_probably_jpeg(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff"


@dataclass
class GenerationRequest:
    """一次生图请求"""

    image_path: Path
    prompt: str
    negative_prompt: str = ""
    width: int = 1088
    height: int = 1088
    output_format: str = "png"
    seed: int | None = None
    inline_image: bool = True


@dataclass
class ProviderCapabilities:
    supports_negative_prompt: bool = False
    supports_seed: bool = False
    supports_reference_image: bool = True
    supports_size: bool = False
    controlnet: bool = False
    lora: bool = False
    notes: str = ""


class AliyunWanxProvider:
    PATH_SYNC_GENERATE = "/api/v1/services/aigc/multimodal-generation/generation"
    PATH_QUERY_TASK = "/api/v1/tasks/{task_id}"
    PATH_CREATE_TASK = "/api/v1/services/aigc/image2image/image-synthesis"
    sync_mode = True

    capabilities = ProviderCapabilities(
        supports_negative_prompt=True,
        supports_seed=False,
        supports_reference_image=True,
        supports_size=True,
        controlnet=False,
        lora=False,
        notes="图像编辑/风格重绘类：会保持输入照片的构图，因此**不保证**完整全身与 1.5–2 头身",
    )

    def __init__(self, spec: ProviderConfig, *, session: requests.Session | None = None):
        self.spec = spec
        self.session = session or requests.Session()


    def _headers(self, *, async_task: bool) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.spec.api_key}",
            "Content-Type": "application/json",
        }
        if async_task:
            headers["X-DashScope-Async"] = "enable"
        return headers

    def build_sync_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """同步多模态生图请求体（实测确认的形态）。"""
        prompt = request.prompt
        if request.negative_prompt and self.capabilities.supports_negative_prompt:
            prompt = f"{prompt}\n\n避免出现：{request.negative_prompt}"
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "input": {"messages": [{"role": "user", "content": [
                {"image": self._image_field(request)},
                {"text": prompt},
            ]}]},
        }
        if self.capabilities.supports_size:
            payload["parameters"] = {"size": f"{request.width}*{request.height}"}
        return payload

    def run_sync(self, request: GenerationRequest) -> str:
        """同步调用，返回生成的图片 URL。"""
        url = f"{self.spec.base_url}{self.PATH_SYNC_GENERATE}"
        try:
            response = self.session.post(
                url, headers={"Authorization": f"Bearer {self.spec.api_key}",
                              "Content-Type": "application/json"},
                json=self.build_sync_payload(request), timeout=self.spec.timeout_seconds)
        except requests.Timeout as exc:
            raise ApiPostSubmitError(
                f"调用超时（{self.spec.timeout_seconds}s），请求可能已被计费: {exc}") from exc
        except requests.RequestException as exc:
            raise ApiError(f"请求失败: {exc}") from exc

        if response.status_code in (401, 403):
            raise ApiAuthError("鉴权失败（401/403）：请检查 config/chibitoy.yaml 的 api_key 与模型权限。")
        if response.status_code == 429:
            raise ApiRateLimited("被限流（429）：请降低并发或稍后重试。")
        try:
            body = _parse_json(response)
        except ApiError as exc:
            raise ApiPostSubmitError(str(exc)) from exc
        if response.status_code >= 400:
            raise ApiError(f"调用失败（HTTP {response.status_code}）: {_short(body)}")
        try:
            return self._extract_image_url(body)
        except ApiError as exc:
            raise ApiPostSubmitError(str(exc)) from exc

    @staticmethod
    def _extract_image_url(body: dict[str, Any]) -> str:
        """从响应中取图片 URL；兼容两种已知形态。"""
        output = body.get("output") or {}
        for choice in output.get("choices") or []:
            for part in ((choice.get("message") or {}).get("content") or []):
                if isinstance(part, dict) and part.get("image"):
                    return str(part["image"])
        results = output.get("results") or []
        if results and results[0].get("url"):
            return str(results[0]["url"])
        if output.get("image_url"):
            return str(output["image_url"])
        raise ApiError(f"响应里没有图片 URL: {_short(body)}")

    def _image_field(self, request: GenerationRequest) -> str:
        if not request.inline_image:
            return str(request.image_path)
        data = request.image_path.read_bytes()
        suffix = "png" if _is_probably_png(data) else "jpeg"
        return f"data:image/{suffix};base64,{base64.b64encode(data).decode()}"

    def build_create_payload(self, request: GenerationRequest) -> dict[str, Any]:
        image_field: str
        if request.inline_image:
            data = request.image_path.read_bytes()
            suffix = "png" if _is_probably_png(data) else "jpeg"
            image_field = f"data:image/{suffix};base64,{base64.b64encode(data).decode()}"
        else:
            image_field = str(request.image_path)

        prompt = request.prompt
        if request.negative_prompt and self.capabilities.supports_negative_prompt:
            prompt = f"{prompt}\n\n避免出现：{request.negative_prompt}"

        payload: dict[str, Any] = {
            "model": self.spec.model,
            "input": {"function": "style_repaint", "image_url": image_field, "prompt": prompt},
            "parameters": {"n": 1},
        }
        if self.capabilities.supports_size:
            payload["parameters"]["size"] = f"{request.width}*{request.height}"
        return payload

    def create_task(self, request: GenerationRequest) -> str:
        url = f"{self.spec.base_url}{self.PATH_CREATE_TASK}"
        try:
            response = self.session.post(
                url, headers=self._headers(async_task=True),
                data=json.dumps(self.build_create_payload(request), ensure_ascii=False),
                timeout=self.spec.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise ApiTimeout(f"创建任务超时（{self.spec.timeout_seconds}s）: {exc}") from exc
        except requests.RequestException as exc:
            raise ApiError(f"创建任务请求失败: {exc}") from exc

        body = _parse_json(response)
        if response.status_code in (401, 403):
            raise ApiAuthError("鉴权失败（401/403）：请检查 config/chibitoy.yaml 里的 api_key 是否正确、是否有该模型权限。")
        if response.status_code == 429:
            raise ApiRateLimited("被限流（429）：请降低并发或稍后重试。")
        if response.status_code >= 400:
            raise ApiError(f"创建任务失败（HTTP {response.status_code}）: {_short(body)}")

        task_id = (body.get("output") or {}).get("task_id") or body.get("task_id")
        if not task_id:
            raise ApiError(f"响应里没有 task_id，无法轮询: {_short(body)}")
        return str(task_id)

    def query_task(self, task_id: str) -> dict[str, Any]:
        url = f"{self.spec.base_url}{self.PATH_QUERY_TASK.format(task_id=task_id)}"
        try:
            response = self.session.get(
                url, headers={"Authorization": f"Bearer {self.spec.api_key}"},
                timeout=self.spec.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ApiError(f"查询任务失败: {exc}") from exc
        return _parse_json(response)

    def wait_for_image_url(self, task_id: str) -> str:
        """轮询直到出图；返回图片 URL。"""
        deadline = time.monotonic() + self.spec.poll_timeout_seconds
        while True:
            body = self.query_task(task_id)
            output = body.get("output") or {}
            status = str(output.get("task_status") or "").upper()
            if status in ("SUCCEEDED", "SUCCESS"):
                results = output.get("results") or []
                url = results[0].get("url") if results else output.get("image_url")
                if not url:
                    raise ApiError(f"任务成功但响应里没有图片 URL: {_short(body)}")
                return str(url)
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise ApiError(f"任务失败（{status}）: {_short(output)}")
            if time.monotonic() >= deadline:
                raise ApiPostSubmitError(
                    f"轮询超时（{self.spec.poll_timeout_seconds}s），task_id={task_id}，"
                    f"最后一次状态={status or '未知'}"
                )
            time.sleep(self.spec.poll_interval_seconds)


class ApiBackend:
    """云端生图后端：把本地照片发给服务商，取回图片并落盘为 PNG。"""

    PROVIDERS = {"aliyun": AliyunWanxProvider}

    def __init__(self, cfg: ApiConfig, *, session: requests.Session | None = None):
        self.cfg = cfg
        spec = cfg.active_provider() # 缺服务商时
        provider_cls = self.PROVIDERS.get(spec.name)
        if provider_cls is None:
            raise ConfigError(
                f"暂未实现服务商 {spec.name!r}。已实现：{', '.join(sorted(self.PROVIDERS))}。"
                f"\n新增方式：在 src/chibitoy/apibackend.py 里实现一个 Provider 类并登记到 PROVIDERS。"
            )
        self.provider = provider_cls(spec, session=session)
        if not spec.has_key:
            raise ConfigError(
                f"服务商 {spec.name!r} 尚未配置 API Key。\n"
                f"请编辑 {cfg.path}，填写 providers.{spec.name}.api_key。"
            )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.provider.capabilities

    def build_request(self, image_path: Path, *, appearance: str = "") -> GenerationRequest:
        gen = self.cfg.generation
        prompt = gen.style_prompt
        if appearance:
            prompt = f"{prompt}, {appearance}"
        return GenerationRequest(
            image_path=image_path, prompt=prompt, negative_prompt=gen.negative_prompt,
            width=gen.width, height=gen.height, output_format=gen.output_format,
        )

    def generate(self, request: GenerationRequest, *, output_dir: Path) -> GenerationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        attempts = max(1, self.provider.spec.max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                if getattr(self.provider, "sync_mode", False):
                    task_id = f"sync-{int(time.time())}"
                    image_url = self.provider.run_sync(request)
                else:
                    task_id = self.provider.create_task(request)
                    image_url = self.provider.wait_for_image_url(task_id)
                data = self._download(image_url)
                path = config.unique_path(
                    output_dir, f"api_{request.image_path.stem}_{int(time.time())}.png")
                path.write_bytes(data)
                return GenerationResult(
                    prompt_id=task_id,
                    images=[{"filename": path.name, "source": "api", "url": image_url}],
                    saved_paths=[path],
                    elapsed_seconds=time.monotonic() - started,
                    raw_history={"provider": self.provider.spec.name,
                                 "model": self.provider.spec.model,
                                 "attempts": attempt},
                )
            except ApiAuthError:
                raise
            except ApiPostSubmitError:
                raise
            except (ApiRateLimited, ApiTimeout, ApiError) as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 30)) # 指数退避，封顶 30s
        assert last_error is not None
        raise type(last_error)(f"{attempts} 次尝试均失败，最后一次：{last_error}") from last_error

    def _download(self, url: str) -> bytes:
        try:
            response = self.provider.session.get(url, timeout=self.provider.spec.timeout_seconds)
        except requests.RequestException as exc:
            raise ApiPostSubmitError(f"下载生成结果失败: {exc}") from exc
        if response.status_code >= 400:
            raise ApiPostSubmitError(f"下载生成结果失败（HTTP {response.status_code}）")
        data = response.content
        if not (_is_probably_png(data) or _is_probably_jpeg(data)):
            raise ApiPostSubmitError(
                f"返回的内容不是有效图片（前 8 字节：{data[:8]!r}）")
        return data


def _parse_json(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        raise ApiError(f"响应不是合法 JSON（HTTP {response.status_code}）: "
                       f"{response.text[:200]!r}") from None
    if not isinstance(body, dict):
        raise ApiError(f"响应 JSON 顶层不是对象: {type(body).__name__}")
    return body


def _short(obj: Any, limit: int = 200) -> str:
    text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    return text[:limit]


def build_backend(mode: str | None = None, *, config_path: Path | str | None = None):
    """按配置返回后端对象（``ApiBackend`` 或 ``ComfyUIClient``）。

    ``mode`` 为空时用配置文件里的 ``backend.mode``。
    """
    cfg = load_config(config_path)
    chosen = (mode or cfg.mode).strip().lower()
    if chosen == "api":
        return cfg, ApiBackend(cfg)
    if chosen == "comfyui":
        from chibitoy.client import ComfyUIClient
        return cfg, ComfyUIClient()
    raise ConfigError(f"未知后端 {chosen!r}，只支持 api 或 comfyui。")
