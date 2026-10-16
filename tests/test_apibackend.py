"""API 后端测试：**全部使用模拟响应，绝不发起真实请求**。

覆盖本阶段要求的 12 项中的前 12 项（第 13 项"现有功能回归"由整个测试套件承担）：
配置解析 / 缺 Key 提示 / 请求构造 / 模拟成功 / 模拟异步轮询 / 限流·超时·鉴权·服务端错误 /
无效图片 / 保存 / 后端选择 / 失败重试 / 密钥不入日志。
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
import requests

from chibitoy import apibackend, apiconfig

# --- 测试替身：一个可编排的假 HTTP 会话 -------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", text=""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32


class FakeSession:
    """可编排的假 HTTP 会话。

    先按队列出队；队列空了之后**按 URL 分派粘性响应** —— 查询任务的 URL 走
    ``query_response``，其余（图片下载）走 ``download_response``。
    这样测试不再依赖"轮询恰好发生几次"这种脆弱假设。
    """

    def __init__(self, *, post=None, get=None, query_response=None, download_response=None):
        self.post_queue = list(post or [])
        self.get_queue = list(get or [])
        self.query_response = query_response
        self.download_response = download_response
        self.calls: list[dict] = []

    def post(self, url, headers=None, data=None, json=None, timeout=None):
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers or {}),
                           "data": data, "json": json, "timeout": timeout})
        return self._next(self.post_queue, "POST 队列")

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers or {}),
                           "timeout": timeout})
        if self.get_queue:
            return self._next(self.get_queue, "GET 队列")
        sticky = self.query_response if "/tasks/" in url else self.download_response
        if sticky is None:
            raise AssertionError(f"测试替身没有为 {url} 预置响应")
        return sticky

    @staticmethod
    def _next(queue, label):
        if not queue:
            raise AssertionError(f"测试替身没有预置足够的响应（{label}）")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _backend(cfg, session, *, sync=False):
    """构造后端。

    Provider 默认是**同步**模式（qwen-image-2.0 实测同步返回）；
    多数测试针对**异步轮询**逻辑，故这里默认关掉 sync_mode，再单独测同步路径。
    """
    backend = apibackend.ApiBackend(cfg, session=session)
    backend.provider.sync_mode = sync
    return backend


def _succeeded(url="https://img/x.png"):
    return FakeResponse(200, {"output": {"task_status": "SUCCEEDED", "results": [{"url": url}]}})


def _write_config(tmp_path: Path, *, api_key="sk-test-not-a-real-key", mode="api") -> Path:
    path = tmp_path / "chibitoy.yaml"
    path.write_text(f"""
backend:
  mode: "{mode}"
  provider: "aliyun"
providers:
  aliyun:
    api_key: "{api_key}"
    base_url: "https://example.invalid"
    model: "wanx-style-repaint"
    timeout_seconds: 5
    poll_interval_seconds: 0
    poll_timeout_seconds: 2
    max_retries: 2
generation:
  width: 1088
  height: 1088
  output_format: "png"
  style_prompt: "chibi toy"
  negative_prompt: "photorealistic"
""", encoding="utf-8")
    return path


@pytest.fixture()
def photo(tmp_path) -> Path:
    from PIL import Image
    p = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), (200, 180, 170)).save(p)
    return p


# --- 1. 配置解析 -------------------------------------------------------------


def test_load_parses_config(tmp_path):
    cfg = apiconfig.load(_write_config(tmp_path))
    assert cfg.mode == "api" and cfg.provider == "aliyun"
    assert cfg.generation.width == 1088
    provider = cfg.active_provider()
    assert provider.model == "wanx-style-repaint"
    assert provider.poll_interval_seconds == 0.0      # 0 也不应被当成"未设置"
    assert provider.max_retries == 2


def test_load_rejects_unknown_mode(tmp_path):
    with pytest.raises(apiconfig.ConfigError, match="backend.mode"):
        apiconfig.load(_write_config(tmp_path, mode="nope"))


def test_load_missing_file_tells_user_how_to_fix(tmp_path):
    with pytest.raises(apiconfig.ConfigError) as excinfo:
        apiconfig.load(tmp_path / "absent.yaml")
    message = str(excinfo.value)
    assert "cp " in message and "chibitoy.example.yaml" in message


# --- 2. 缺 API Key 时给出可照做的提示 ----------------------------------------


def test_missing_key_raises_actionable_error(tmp_path):
    with pytest.raises(apiconfig.ConfigError) as excinfo:
        apiconfig.load(_write_config(tmp_path, api_key=""), require_key=True)
    message = str(excinfo.value)
    assert "api_key" in message and "aliyun" in message
    assert ".gitignore" in message                     # 告诉用户该文件不会入库


def test_missing_key_does_not_raise_without_require(tmp_path):
    cfg = apiconfig.load(_write_config(tmp_path, api_key=""))
    assert cfg.active_provider().has_key is False       # 允许只读配置，不强制


def test_api_backend_refuses_to_run_without_key(tmp_path):
    """**核心断言**：没密钥就明确报错，绝不伪造调用。"""
    cfg = apiconfig.load(_write_config(tmp_path, api_key=""))
    with pytest.raises(apiconfig.ConfigError, match="尚未配置 API Key"):
        apibackend.ApiBackend(cfg)


# --- 3. 密钥卫生：不进日志、不进描述、不进异常 --------------------------------


def test_describe_never_leaks_the_key(tmp_path):
    secret = "sk-super-secret-value-123"
    cfg = apiconfig.load(_write_config(tmp_path, api_key=secret))
    described = cfg.describe()
    assert secret not in described
    assert "已配置" in described


def test_error_messages_never_leak_the_key(tmp_path, photo):
    secret = "sk-super-secret-value-123"
    cfg = apiconfig.load(_write_config(tmp_path, api_key=secret))
    session = FakeSession(post=[FakeResponse(500, {"message": "boom"})] * 2)
    backend = _backend(cfg, session)
    request = backend.build_request(photo)
    with pytest.raises(apibackend.ApiError) as excinfo:
        backend.generate(request, output_dir=tmp_path / "out")
    assert secret not in str(excinfo.value)


# --- 4. 请求构造 -------------------------------------------------------------


def test_build_payload_inlines_the_photo_as_data_url(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    backend = _backend(cfg, FakeSession(), sync=True)
    payload = backend.provider.build_create_payload(backend.build_request(photo))
    assert payload["model"] == "wanx-style-repaint"
    image_url = payload["input"]["image_url"]
    assert image_url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(image_url.split(",", 1)[1])
    assert decoded == photo.read_bytes()
    assert payload["parameters"]["size"] == "1088*1088"


def test_build_payload_merges_negative_prompt_when_supported(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    backend = _backend(cfg, FakeSession(), sync=True)
    payload = backend.provider.build_create_payload(backend.build_request(photo, appearance="black hair"))
    prompt = payload["input"]["prompt"]
    assert "chibi toy" in prompt and "black hair" in prompt
    assert "photorealistic" in prompt                   # 负向词被并入


def test_provider_declares_it_has_no_controlnet_or_lora():
    """**不得假装云端有 ControlNet / LoRA。**"""
    caps = apibackend.AliyunWanxProvider.capabilities
    assert caps.controlnet is False and caps.lora is False
    assert "不保证" in caps.notes


# --- 5~6. 模拟成功、异步提交 + 轮询 ------------------------------------------


def test_generate_success_saves_png(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {"task_id": "t-1"}})],
                          query_response=_succeeded(),
                          download_response=FakeResponse(200, content=PNG_BYTES))
    backend = _backend(cfg, session)
    result = backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert result.prompt_id == "t-1"
    assert len(result.saved_paths) == 1
    assert result.saved_paths[0].read_bytes() == PNG_BYTES
    assert result.raw_history["provider"] == "aliyun"


def test_generate_polls_until_succeeded(tmp_path, photo):
    """先返回 RUNNING，再返回 SUCCEEDED —— 必须真的轮询，而不是只查一次。"""
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(
        post=[FakeResponse(200, {"output": {"task_id": "t-2"}})],
        get=[FakeResponse(200, {"output": {"task_status": "RUNNING"}}),
             _succeeded("https://img/y.png")],
        download_response=FakeResponse(200, content=PNG_BYTES))
    backend = _backend(cfg, session)
    result = backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert result.prompt_id == "t-2"
    assert sum(1 for c in session.calls if c["method"] == "GET") >= 2


def test_async_header_is_sent(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {"task_id": "t"}})],
                          query_response=_succeeded("u"),
                          download_response=FakeResponse(200, content=PNG_BYTES))
    _backend(cfg, session).generate(
        _backend(cfg, FakeSession(), sync=True).build_request(photo),
        output_dir=tmp_path / "out")
    post_call = next(c for c in session.calls if c["method"] == "POST")
    assert post_call["headers"].get("X-DashScope-Async") == "enable"


# --- 7. 限流 / 超时 / 鉴权 / 服务端错误 --------------------------------------


def test_auth_failure_is_not_retried(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(401, {"message": "bad key"})])
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiAuthError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert sum(1 for c in session.calls if c["method"] == "POST") == 1


def test_rate_limit_is_retried_then_reported(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))          # max_retries=2
    session = FakeSession(post=[FakeResponse(429, {"message": "slow down"}),
                                FakeResponse(429, {"message": "slow down"})])
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError, match="2 次尝试均失败"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert sum(1 for c in session.calls if c["method"] == "POST") == 2


def test_server_error_retried_then_raises(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(503, {"message": "down"}),
                                FakeResponse(503, {"message": "down"})])
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


def test_poll_timeout_raises_with_task_id(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))          # poll_timeout=2s
    session = FakeSession(post=[FakeResponse(200, {"output": {"task_id": "t-slow"}})] * 2,
                          query_response=FakeResponse(200, {"output": {"task_status": "RUNNING"}}))
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError, match="轮询超时"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


def test_http_timeout_is_wrapped(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[requests.Timeout("too slow"), requests.Timeout("too slow")])
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


def test_task_failed_status_raises(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {"task_id": "t-bad"}})] * 2,
                          get=[FakeResponse(200, {"output": {"task_status": "FAILED",
                                                             "message": "审核不通过"}})] * 2)
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError, match="任务失败"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


def test_non_json_response_is_reported(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, None, text="<html>oops</html>")] * 2)
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError, match="不是合法 JSON"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


def test_missing_task_id_is_reported(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {}})] * 2)
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError, match="task_id"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


# --- 8. 无效图片响应 ---------------------------------------------------------


def test_invalid_image_payload_is_rejected(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {"task_id": "t"}})] * 2,
                          query_response=_succeeded("u"),
                          download_response=FakeResponse(200, content=b"<html>not an image</html>"))
    backend = _backend(cfg, session)
    with pytest.raises(apibackend.ApiError, match="不是有效图片"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.png")), "无效图片不得落盘"


# --- 10. 后端选择 ------------------------------------------------------------


def test_build_backend_selects_api(tmp_path):
    cfg, backend = apibackend.build_backend(config_path=_write_config(tmp_path))
    assert isinstance(backend, apibackend.ApiBackend)
    assert cfg.mode == "api"


def test_build_backend_selects_comfyui(tmp_path):
    from chibitoy.client import ComfyUIClient
    cfg, backend = apibackend.build_backend(config_path=_write_config(tmp_path, mode="comfyui"))
    assert isinstance(backend, ComfyUIClient)


def test_build_backend_rejects_unknown(tmp_path):
    with pytest.raises(apiconfig.ConfigError, match="未知后端"):
        apibackend.build_backend("nonsense", config_path=_write_config(tmp_path))


def test_provider_missing_from_config_lists_known_ones(tmp_path):
    path = _write_config(tmp_path)
    path.write_text(path.read_text().replace('provider: "aliyun"', 'provider: "other"'),
                    encoding="utf-8")
    cfg = apiconfig.load(path)
    with pytest.raises(apiconfig.ConfigError, match="已有"):
        cfg.active_provider()


# --- 11. 单项失败不应污染批次 ------------------------------------------------


def test_retry_recovers_within_max_retries(tmp_path, photo):
    """第一次限流、第二次成功 —— 应当恢复而不是整批失败。"""
    cfg = apiconfig.load(_write_config(tmp_path))          # max_retries=2
    session = FakeSession(
        post=[FakeResponse(429, {"message": "slow"}),
              FakeResponse(200, {"output": {"task_id": "t-ok"}})],
        query_response=_succeeded("u"),
        download_response=FakeResponse(200, content=PNG_BYTES))
    backend = _backend(cfg, session)
    result = backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert result.raw_history["attempts"] == 2
    assert result.saved_paths[0].exists()


# --- 同步路径（qwen-image-2.0 实测形态） ------------------------------------


def test_sync_payload_sends_size_and_reference(tmp_path, photo):
    """**实测确认的形态**：同步多模态生图，必须带 size，否则只出 1024（短边不足 1080）。"""
    cfg = apiconfig.load(_write_config(tmp_path))
    backend = _backend(cfg, FakeSession(), sync=True)
    payload = backend.provider.build_sync_payload(backend.build_request(photo))
    assert payload["model"] == "wanx-style-repaint"
    content = payload["input"]["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert "chibi toy" in content[1]["text"]
    assert payload["parameters"]["size"] == "1088*1088"


def test_sync_generate_parses_verified_response(tmp_path, photo):
    """同步响应形态：output.choices[0].message.content[0].image。"""
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(
        post=[FakeResponse(200, {"output": {"choices": [
            {"message": {"content": [{"image": "https://img/sync.png"}]}}]}})],
        download_response=FakeResponse(200, content=PNG_BYTES))
    backend = _backend(cfg, session, sync=True)
    result = backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert result.saved_paths[0].read_bytes() == PNG_BYTES
    assert result.images[0]["url"] == "https://img/sync.png"


def test_sync_auth_error_not_retried(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(401, {"message": "bad key"})])
    backend = _backend(cfg, session, sync=True)
    with pytest.raises(apibackend.ApiAuthError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert sum(1 for c in session.calls if c["method"] == "POST") == 1


def test_sync_rate_limit_retried(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(429, {"message": "slow"}),
                                FakeResponse(200, {"output": {"choices": [
                                    {"message": {"content": [{"image": "u"}]}}]}})],
                          download_response=FakeResponse(200, content=PNG_BYTES))
    backend = _backend(cfg, session, sync=True)
    result = backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert result.raw_history["attempts"] == 2


def test_sync_response_without_image_is_reported(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {"choices": []}})] * 2)
    backend = _backend(cfg, session, sync=True)
    with pytest.raises(apibackend.ApiError, match="没有图片 URL"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")


def test_sync_payload_omits_size_when_unsupported(tmp_path, photo):
    """厂商不支持指定尺寸时不得塞 size 参数（能力差异要如实适配）。"""
    cfg = apiconfig.load(_write_config(tmp_path))
    backend = _backend(cfg, FakeSession(), sync=True)
    backend.provider.capabilities = apibackend.ProviderCapabilities(supports_size=False)
    payload = backend.provider.build_sync_payload(backend.build_request(photo))
    assert "parameters" not in payload


# --- hybrid_stability.py 的费用保护与参数校验 --------------------------------

def _stability_module():
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("hybrid_stability",
                                                  root / "scripts" / "hybrid_stability.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stability_refuses_runs_above_max_calls(tmp_path, monkeypatch, capsys):
    """**费用保护**：请求次数不得超过硬上限，否则直接拒绝执行。"""
    module = _stability_module()
    base = tmp_path / "base.png"
    from PIL import Image as _Img
    _Img.new("RGB", (32, 32)).save(base)
    monkeypatch.setattr(sys, "argv",
                        ["hybrid_stability.py", "--runs", "9", "--max-calls", "5",
                         "--confirm-cost", "--base", str(base), "--out", str(tmp_path / "o")])
    assert module.main() == 2
    assert "拒绝执行" in capsys.readouterr().err


def test_stability_requires_confirm_cost_and_makes_no_call(tmp_path, monkeypatch, capsys):
    """**核心断言**：没有 --confirm-cost 时绝不发起付费请求。"""
    module = _stability_module()
    from PIL import Image as _Img
    base = tmp_path / "base.png"
    _Img.new("RGB", (32, 32)).save(base)

    called = {"n": 0}

    class ExplodingBackend:
        def __init__(self, *a, **k):
            called["n"] += 1
            raise AssertionError("未确认费用时不得构造后端（会发起付费调用）")

    monkeypatch.setattr(apibackend, "ApiBackend", ExplodingBackend)
    monkeypatch.setattr(sys, "argv",
                        ["hybrid_stability.py", "--runs", "5",
                         "--base", str(base), "--out", str(tmp_path / "o")])
    assert module.main() == 0
    assert called["n"] == 0
    assert "费用保护" in capsys.readouterr().out


def test_stability_forces_single_attempt(tmp_path, monkeypatch):
    """**回归**：脚本承诺"失败不重试"，就必须把 max_retries 压到 1。

    否则 ApiBackend 内部默认重试 3 次 —— 一次失败会变成 3 次付费调用。
    """
    module = _stability_module()
    from PIL import Image as _Img
    base = tmp_path / "base.png"
    _Img.new("RGB", (32, 32)).save(base)
    cfg = apiconfig.load(_write_config(tmp_path))
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: cfg)
    captured = {}

    class CapturingBackend:
        def __init__(self, config, **kwargs):
            captured["max_retries"] = config.active_provider().max_retries
            self.cfg = config
            self.provider = type("P", (), {"spec": config.active_provider()})()
            raise apibackend.ApiError("停在构造后，避免真正调用")

    monkeypatch.setattr(apibackend, "ApiBackend", CapturingBackend)
    monkeypatch.setattr(sys, "argv",
                        ["hybrid_stability.py", "--runs", "1", "--confirm-cost",
                         "--base", str(base), "--out", str(tmp_path / "o")])
    try:
        module.main()
    except apibackend.ApiError:
        pass
    assert captured.get("max_retries") == 1


def test_stability_stats_uses_relative_range_not_cv(tmp_path):
    """统计口径必须与项目一致：(max-min)/mean，而不是标准差/均值。"""
    module = _stability_module()
    stats = module._stats([1.0, 2.0, 3.0])
    assert stats["n"] == 3 and stats["mean"] == 2.0
    assert stats["relative_range_percent"] == 100.0
    assert module._stats([]) == {}


def test_stability_records_failure_without_retry(tmp_path, monkeypatch):
    """API 失败时记录错误、不抛栈、不重试；报告仍能写出。"""
    module = _stability_module()
    from PIL import Image as _Img
    base = tmp_path / "base.png"
    _Img.new("RGB", (32, 32)).save(base)
    cfg = apiconfig.load(_write_config(tmp_path))
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: cfg)

    calls = {"n": 0}

    class FailingBackend:
        def __init__(self, config, **kwargs):
            self.cfg = config
            self.provider = type("P", (), {"spec": config.active_provider()})()
            self.capabilities = apibackend.AliyunWanxProvider.capabilities

        def build_request(self, path, **kwargs):
            return apibackend.GenerationRequest(image_path=path, prompt="x")

        def generate(self, request, *, output_dir):
            calls["n"] += 1
            raise apibackend.ApiError("模拟失败：不得重试")

    monkeypatch.setattr(apibackend, "ApiBackend", FailingBackend)
    monkeypatch.setattr(sys, "argv",
                        ["hybrid_stability.py", "--runs", "2", "--confirm-cost",
                         "--base", str(base), "--out", str(tmp_path / "o")])
    code = module.main()
    assert code == 1                      # 全失败 -> 非零退出
    assert calls["n"] == 2                # 两次运行各一次，**没有重试**
    report = json.loads((tmp_path / "o" / "report.json").read_text(encoding="utf-8"))
    assert report["success_count"] == 0 and report["calls_made"] == 2
    assert all("模拟失败" in r.get("error", "") for r in report["records"])


def test_stability_prompt_override_is_used_and_recorded(tmp_path, monkeypatch):
    """配饰移除消融实验靠 `--prompt` 覆盖，必须真的传进请求并记入报告。"""
    module = _stability_module()
    from PIL import Image as _Img
    base = tmp_path / "base.png"
    _Img.new("RGB", (32, 32)).save(base)
    cfg = apiconfig.load(_write_config(tmp_path))
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: cfg)
    seen = {}

    class CapturingBackend:
        def __init__(self, config, **kwargs):
            self.cfg = config
            self.provider = type("P", (), {"spec": config.active_provider()})()
            self.capabilities = apibackend.AliyunWanxProvider.capabilities

        def build_request(self, path, **kwargs):
            return apibackend.GenerationRequest(image_path=path, prompt="默认")

        def generate(self, request, *, output_dir):
            seen["prompt"] = request.prompt
            raise apibackend.ApiError("停在记录之后，不真正调用")

    monkeypatch.setattr(apibackend, "ApiBackend", CapturingBackend)
    custom = "REMOVE ALL HEADPHONES AND EARMUFFS, plain hair only"
    monkeypatch.setattr(sys, "argv",
                        ["hybrid_stability.py", "--runs", "1", "--confirm-cost",
                         "--prompt", custom, "--base", str(base), "--out", str(tmp_path / "o")])
    module.main()
    assert seen["prompt"] == custom
    report = json.loads((tmp_path / "o" / "report.json").read_text(encoding="utf-8"))
    assert report["config"]["prompt"] == custom
    assert report["config"]["prompt_overridden"] is True


# --- 12. `generate --backend api` 的费用保护 ---------------------------------
#
# 历史缺陷：`max_retries` 默认 3，且只有 hybrid_stability.py 手动压到 1。
# 于是 `chibitoy.generate --backend api` 在 5xx/限流/超时时会**静默发起 3 次付费调用**。
# 下面三条测试把"默认不重试"和"无确认不发请求"钉死。


def _write_minimal_config(tmp_path: Path, *, include_max_retries: bool) -> Path:
    """写一份最小配置；可选择性省略 max_retries 以检验**默认值**。"""
    retries = "    max_retries: 3\n" if include_max_retries else ""
    path = tmp_path / "cfg.yaml"
    path.write_text(f"""
backend:
  mode: "api"
  provider: "aliyun"
providers:
  aliyun:
    api_key: "test-key-not-real"
    base_url: "https://example.invalid"
    model: "wanx-style-repaint"
{retries}generation:
  width: 1088
  height: 1088
  style_prompt: "chibi toy"
""", encoding="utf-8")
    return path


def test_default_max_retries_is_single_attempt(tmp_path):
    """**回归**：配置里不写 max_retries 时，默认必须是 1 —— 不自动重试、不重复计费。"""
    assert apiconfig.ProviderConfig(name="x").max_retries == 1
    cfg = apiconfig.load(_write_minimal_config(tmp_path, include_max_retries=False))
    assert cfg.active_provider().max_retries == 1


def test_default_config_retries_is_single_attempt():
    """仓库自带的模板配置也必须默认不重试（防止模板把 3 又带回来）。"""
    example = Path(__file__).resolve().parents[1] / "config" / "chibitoy.example.yaml"
    cfg = apiconfig.load(example)
    assert cfg.active_provider().max_retries == 1, (
        "config/chibitoy.example.yaml 的 max_retries 必须是 1："
        "每次重试都是真实付费调用")


def test_generate_api_requires_confirm_cost(tmp_path, monkeypatch, capsys):
    """**核心断言**：`--backend api` 未给 --confirm-cost 时不得构造后端、不得发请求。"""
    import chibitoy.generate as gen

    cfg_path = _write_minimal_config(tmp_path, include_max_retries=False)
    real_load = apiconfig.load                       # 先抓住真函数，避免自递归
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: real_load(cfg_path))

    constructed = {"n": 0}

    class ExplodingBackend:
        def __init__(self, *a, **k):
            constructed["n"] += 1
            raise AssertionError("未确认费用时不得构造后端")

    monkeypatch.setattr(apibackend, "ApiBackend", ExplodingBackend)

    code = gen.main(["--backend", "api", "--input-image", str(tmp_path / "p.jpg"),
                     "--out", str(tmp_path / "o")])
    assert code == 3, ("未确认费用应返回 **非零** 退出码 3（未执行），"
                       "否则批量脚本会把空跑记成成功")
    assert constructed["n"] == 0, "不得构造 ApiBackend"
    out = capsys.readouterr().out
    assert "费用保护" in out and "--confirm-cost" in out
    assert "未发起任何请求" in out


def test_generate_api_confirm_cost_passes_gate(tmp_path, monkeypatch, capsys, photo):
    """给了 --confirm-cost 之后应继续走到构造后端（此处故意让其抛错以证明已越过门）。"""
    import chibitoy.generate as gen

    cfg_path = _write_minimal_config(tmp_path, include_max_retries=False)
    real_load = apiconfig.load                       # 先抓住真函数，避免自递归
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: real_load(cfg_path))

    reached = {"n": 0}

    class StoppingBackend:
        def __init__(self, config, **kwargs):
            reached["n"] += 1
            raise apiconfig.ConfigError("停在构造后，避免真正调用")

    monkeypatch.setattr(apibackend, "ApiBackend", StoppingBackend)

    code = gen.main(["--backend", "api", "--confirm-cost",
                     "--input-image", str(photo), "--out", str(tmp_path / "o")])
    assert reached["n"] == 1, "确认费用后应当构造后端"
    assert code == 2                     # ConfigError -> 退出码 2
    assert "费用保护" not in capsys.readouterr().out


def test_generate_api_max_retries_flag_is_applied(tmp_path, monkeypatch, photo, capsys):
    """`--max-retries` 必须真的写回配置，并在 >1 时提示费用放大。

    **回归**：该选项原先在 `ApiBackend` 构造**之后**才写回配置，导致完全不生效。
    """
    import chibitoy.generate as gen

    cfg_path = _write_minimal_config(tmp_path, include_max_retries=False)
    real_load = apiconfig.load                       # 先抓住真函数，避免自递归
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: real_load(cfg_path))

    seen = {}

    class CapturingBackend:
        """构造成功（以便费用提示打印），但在 generate 处中止，避免任何真实调用。"""

        class _Provider:
            def __init__(self, spec):
                self.spec = spec

        def __init__(self, config, **kwargs):
            seen["max_retries"] = config.active_provider().max_retries
            self.cfg = config
            self.provider = self._Provider(config.active_provider())
            self.capabilities = apibackend.AliyunWanxProvider.capabilities

        def build_request(self, path, **kwargs):
            return apibackend.GenerationRequest(image_path=path, prompt="x")

        def generate(self, request, *, output_dir):
            raise apibackend.ApiError("停在 generate，避免真实调用")

    monkeypatch.setattr(apibackend, "ApiBackend", CapturingBackend)
    code = gen.main(["--backend", "api", "--confirm-cost", "--max-retries", "4",
                     "--input-image", str(photo), "--out", str(tmp_path / "o")])

    assert seen["max_retries"] == 4, "--max-retries 必须在构造后端前写回配置"
    assert code == 1, "generate 抛 ApiError 应返回退出码 1"
    assert "4 次付费调用" in capsys.readouterr().out


def test_generate_api_rejects_zero_max_retries(tmp_path, monkeypatch, photo):
    """`--max-retries 0` 必须被拒绝（否则"零次尝试"语义不明）。"""
    import chibitoy.generate as gen

    cfg_path = _write_minimal_config(tmp_path, include_max_retries=False)
    real_load = apiconfig.load                       # 先抓住真函数，避免自递归
    monkeypatch.setattr(apiconfig, "load", lambda *a, **k: real_load(cfg_path))
    code = gen.main(["--backend", "api", "--confirm-cost", "--max-retries", "0",
                     "--input-image", str(photo), "--out", str(tmp_path / "o")])
    assert code == 2


# --- 13. 已提交（可能已计费）之后不得重试 ------------------------------------
#
# 历史缺陷：`_download()` / `_extract_image_url()` / 轮询超时抛的都是普通 ApiError，
# 被同一个重试循环捕获 → **同一张图被重复计费**。


def _post_count(session) -> int:
    return sum(1 for call in session.calls if call["method"] == "POST")


def test_download_failure_after_success_is_not_rebilled(tmp_path, photo):
    """生成已成功、下载失败：**不得**再次提交（否则重复付费）。"""
    cfg = apiconfig.load(_write_config(tmp_path))          # max_retries=2
    session = FakeSession(
        post=[FakeResponse(200, {"output": {"choices": [
            {"message": {"content": [{"image": "https://img/x.png"}]}}]}})] * 4,
        download_response=FakeResponse(500, content=b"boom"))
    backend = _backend(cfg, session, sync=True)

    with pytest.raises(apibackend.ApiError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")

    assert _post_count(session) == 1, "已计费的生成请求被重复提交了"


def test_non_image_response_after_success_is_not_rebilled(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(
        post=[FakeResponse(200, {"output": {"choices": [
            {"message": {"content": [{"image": "https://img/x.png"}]}}]}})] * 4,
        download_response=FakeResponse(200, content=b"<html>not an image</html>"))
    backend = _backend(cfg, session, sync=True)

    with pytest.raises(apibackend.ApiError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert _post_count(session) == 1


def test_missing_image_url_after_success_is_not_rebilled(tmp_path, photo):
    """200 但响应里没有图片 URL —— 已提交，不得重试。"""
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, {"output": {"choices": []}})] * 4)
    backend = _backend(cfg, session, sync=True)

    with pytest.raises(apibackend.ApiError, match="没有图片 URL"):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert _post_count(session) == 1


def test_bad_json_after_success_is_not_rebilled(tmp_path, photo):
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[FakeResponse(200, payload=None, text="<html>")] * 4)
    backend = _backend(cfg, session, sync=True)

    with pytest.raises(apibackend.ApiError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert _post_count(session) == 1


def test_sync_timeout_is_not_retried(tmp_path, photo):
    """同步调用超时：请求可能已被服务端接受并计费，故不得重发。"""
    cfg = apiconfig.load(_write_config(tmp_path))
    session = FakeSession(post=[requests.Timeout("slow")] * 4)
    backend = _backend(cfg, session, sync=True)

    with pytest.raises(apibackend.ApiError):
        backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert _post_count(session) == 1


def test_pre_submit_rate_limit_is_still_retried(tmp_path, photo):
    """修复不能过头：**提交前**的限流仍应重试（原本就要重试的场景）。"""
    cfg = apiconfig.load(_write_config(tmp_path))          # max_retries=2
    session = FakeSession(
        post=[FakeResponse(429, {"message": "slow"}),
              FakeResponse(200, {"output": {"choices": [
                  {"message": {"content": [{"image": "u"}]}}]}})],
        download_response=FakeResponse(200, content=PNG_BYTES))
    backend = _backend(cfg, session, sync=True)

    result = backend.generate(backend.build_request(photo), output_dir=tmp_path / "out")
    assert result.raw_history["attempts"] == 2
    assert _post_count(session) == 2


def test_post_submit_error_is_an_api_error_subclass():
    """必须是 ApiError 的子类，调用方既有的 `except ApiError` 仍然有效。"""
    assert issubclass(apibackend.ApiPostSubmitError, apibackend.ApiError)


# --- 14. 密钥不得出现在 repr / str ------------------------------------------

def test_provider_config_repr_hides_api_key():
    """`repr(cfg)`/`print(cfg)`/调试器/pytest 断言差异都不得泄露密钥。"""
    secret = "sk-REPR-MUST-NOT-LEAK-12345"
    provider = apiconfig.ProviderConfig(name="aliyun", api_key=secret)
    assert secret not in repr(provider)
    assert secret not in str(provider)

    cfg = apiconfig.ApiConfig(providers={"aliyun": provider})
    assert secret not in repr(cfg)
    assert secret not in str(cfg)
    # 但功能本身不能被破坏
    assert provider.has_key is True


# --- 15. 默认重试次数（源头） -----------------------------------------------

def test_apiconfig_default_max_retries_is_one():
    assert apiconfig.ProviderConfig(name="x").max_retries == 1
