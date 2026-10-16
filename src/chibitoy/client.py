from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from . import config
from .config import COMFY_BASE_URL


class ComfyUIError(RuntimeError):
    """ComfyUI 相关错误的基类"""


class ComfyUIUnavailable(ComfyUIError):
    """服务无法连接"""


class WorkflowRejected(ComfyUIError):
    """服务端拒绝了工作流（校验失败）"""

    def __init__(self, message: str, node_errors: dict[str, Any] | None = None):
        super().__init__(message)
        self.node_errors = node_errors or {}


class ExecutionFailed(ComfyUIError):
    """工作流已提交但执行失败"""


@dataclass
class GenerationResult:
    """一次生成任务的结果。"""

    prompt_id: str
    images: list[dict[str, Any]] = field(default_factory=list)
    saved_paths: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    raw_history: dict[str, Any] = field(default_factory=dict)


class ComfyUIClient:
    """ComfyUI 服务的最小可用客户端"""

    def __init__(self, base_url: str = COMFY_BASE_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self._session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self._session.request(method, url, **kwargs)
        except requests.ConnectionError as exc:
            raise ComfyUIUnavailable(
                f"无法连接 ComfyUI ({self.base_url})。请先启动服务：\n"
                f"    ./scripts/start_comfyui.sh --background"
            ) from exc
        except requests.Timeout as exc:
            raise ComfyUIUnavailable(f"请求 ComfyUI 超时: {url}") from exc
        return response

    def is_available(self) -> bool:
        """服务是否可访问。"""
        try:
            self._request("GET", "/system_stats", timeout=5)
            return True
        except ComfyUIError:
            return False

    def system_stats(self) -> dict[str, Any]:
        """获取服务、设备与版本信息"""
        response = self._request("GET", "/system_stats")
        response.raise_for_status()
        return response.json()

    def object_info(self, node_class: str | None = None) -> dict[str, Any]:
        """获取节点定义；不传参数时返回全部节点"""
        path = "/object_info" if node_class is None else f"/object_info/{node_class}"
        response = self._request("GET", path, timeout=60)
        response.raise_for_status()
        return response.json()

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """提交 API 格式工作流，返回 prompt_id

        Raises:
            WorkflowRejected: 服务端校验失败，附带逐节点错误信息
        """
        payload = {"prompt": workflow, "client_id": self.client_id}
        response = self._request("POST", "/prompt", json=payload)

        if response.status_code != 200:
            detail = ""
            node_errors: dict[str, Any] = {}
            try:
                body = response.json()
                detail = body.get("error", {}).get("message", "") or json.dumps(body)
                node_errors = body.get("node_errors", {}) or {}
            except ValueError:
                detail = response.text[:500]
            raise WorkflowRejected(
                f"ComfyUI 拒绝工作流 (HTTP {response.status_code}): {detail}",
                node_errors=node_errors,
            )

        body = response.json()
        if body.get("node_errors"):
            raise WorkflowRejected(
                f"工作流节点校验失败: {json.dumps(body['node_errors'], ensure_ascii=False)}",
                node_errors=body["node_errors"],
            )
        return body["prompt_id"]

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        """查询指定任务的历史记录，未完成时返回空字典"""
        response = self._request("GET", f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json().get(prompt_id, {})

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 900.0,
        poll_interval: float = 1.0,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """轮询等待任务结束，返回历史记录条目

        Raises:
            TimeoutError: 超时仍未完成
            ExecutionFailed: 执行过程中报错
        """
        started = time.monotonic()
        while True:
            entry = self.get_history(prompt_id)
            if entry:
                status = entry.get("status", {})
                status_str = status.get("status_str")
                if status_str == "error":
                    raise ExecutionFailed(self._format_execution_error(entry))
                # completed 字段为真即代表执行结束
                if status.get("completed") or status_str == "success":
                    return entry
                if entry.get("outputs") and status_str is None:
                    return entry

            if time.monotonic() - started > timeout:
                raise TimeoutError(
                    f"等待执行超时 ({timeout:.0f}s)，prompt_id={prompt_id}"
                )
            if on_progress is not None:
                on_progress(time.monotonic() - started)
            time.sleep(poll_interval)

    @staticmethod
    def _format_execution_error(entry: dict[str, Any]) -> str:
        """把 history 中的错误消息整理成可读文本"""
        messages = entry.get("status", {}).get("messages", [])
        lines: list[str] = []
        for item in messages:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            kind, data = item
            if kind == "execution_error":
                lines.append(
                    "节点 {node_type} (id={node_id}) 执行失败: {exception_type}: "
                    "{exception_message}".format(
                        node_type=data.get("node_type"),
                        node_id=data.get("node_id"),
                        exception_type=data.get("exception_type"),
                        exception_message=data.get("exception_message"),
                    )
                )
            elif kind in ("execution_interrupted",):
                lines.append("执行被中断")
        return "\n".join(lines) or "执行失败，但未返回详细错误信息"

    @staticmethod
    def collect_images(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
        """从历史记录中提取所有输出图片引用"""
        images: list[dict[str, Any]] = []
        for node_id, node_output in (history_entry.get("outputs") or {}).items():
            for image in node_output.get("images", []) or []:
                images.append({**image, "node_id": node_id})
        return images

    def download_image(self, image: dict[str, Any], dest_dir: Path) -> Path:
        """下载单张输出图片到本地目录，返回落盘路径"""
        params = {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
        response = self._request("GET", "/view", params=params, timeout=120)
        response.raise_for_status()

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = config.unique_path(dest_dir, image["filename"])
            dest.write_bytes(response.content)
        except OSError as exc:
            target = dest_dir / Path(image["filename"]).name
            raise ComfyUIError(
                f"无法写入产物 {target}: {type(exc).__name__}: {exc}"
                f"（常见原因：目标目录不可写、磁盘空间不足）"
            ) from exc
        return dest

    def upload_image(
        self, path: Path, subfolder: str = "", overwrite: bool = True
    ) -> str:
        """上传一张图片到 ComfyUI 输入目录，返回服务端可见的文件名

        上传后的文件名会出现在 LoadImage 节点的 ``image`` 候选项中，
        供后续工作流引用
        """
        path = Path(path)
        if not path.is_file():
            raise ComfyUIError(f"输入图片不存在: {path}")

        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")

        with open(path, "rb") as fh:
            response = self._request(
                "POST",
                "/upload/image",
                files={"image": (path.name, fh, mime)},
                data={"subfolder": subfolder, "overwrite": "true" if overwrite else "false"},
                timeout=120,
            )
        response.raise_for_status()
        body = response.json()
        name = body.get("name")
        if not name:
            raise ComfyUIError(f"图片上传失败，服务端响应异常: {body}")
        return name

    def interrupt(self) -> None:
        """中断当前正在执行的任务"""
        self._request("POST", "/interrupt")

    def generate(
        self,
        workflow: dict[str, Any],
        output_dir: Path | None = None,
        timeout: float = 900.0,
        on_progress: Any = None,
    ) -> GenerationResult:
        """提交工作流、等待完成，并按需下载输出图片"""
        started = time.monotonic()
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_completion(
            prompt_id, timeout=timeout, on_progress=on_progress
        )
        elapsed = time.monotonic() - started

        images = self.collect_images(history)
        saved: list[Path] = []
        if output_dir is not None:
            for image in images:
                saved.append(self.download_image(image, output_dir))

        return GenerationResult(
            prompt_id=prompt_id,
            images=images,
            saved_paths=saved,
            elapsed_seconds=elapsed,
            raw_history=history,
        )
