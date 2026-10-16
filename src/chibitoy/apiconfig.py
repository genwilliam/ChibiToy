from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chibitoy import config as project_config

DEFAULT_CONFIG_PATH = project_config.CONFIG_DIR / "chibitoy.yaml"
EXAMPLE_CONFIG_PATH = project_config.CONFIG_DIR / "chibitoy.example.yaml"

API_MODES = ("api", "comfyui")


class ConfigError(RuntimeError):
    """配置缺失或非法"""


@dataclass
class ProviderConfig:
    name: str
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = 120.0
    poll_interval_seconds: float = 5.0
    poll_timeout_seconds: float = 300.0
    max_retries: int = 1

    @property
    def has_key(self) -> bool:
        return bool(self.api_key.strip())


@dataclass
class GenerationConfig:
    width: int = 1088
    height: int = 1088
    output_format: str = "png"
    style_prompt: str = ""
    negative_prompt: str = ""


@dataclass
class ApiConfig:
    mode: str = "api"
    provider: str = "aliyun"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    path: Path | None = None

    def active_provider(self) -> ProviderConfig:
        if self.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "（无）"
            raise ConfigError(
                f"配置里没有服务商 {self.provider!r}。已有：{known}。\n"
                f"请检查 {self.path or DEFAULT_CONFIG_PATH} 的 providers 段。"
            )
        return self.providers[self.provider]

    def describe(self) -> str:
        """安全摘要：**绝不含密钥内容**。"""
        provider = self.providers.get(self.provider)
        key_state = "未配置" if not provider or not provider.has_key else f"已配置（{len(provider.api_key)} 字符）"
        return (f"mode={self.mode} provider={self.provider} model="
                f"{provider.model if provider else 'n/a'} api_key={key_state}")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load(path: Path | str | None = None, *, require_key: bool = False) -> ApiConfig:
    """读取配置

    ``require_key=True`` 时，若当前服务商没填 api_key，会给出可直接照做的提示
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        def _shown(path: Path) -> str:
            try:
                return str(path.relative_to(project_config.PROJECT_ROOT))
            except ValueError:
                return str(path)

        raise ConfigError(
            f"找不到配置文件 {config_path}。\n"
            f"请先复制模板并填入 API Key：\n"
            f"    cp {_shown(EXAMPLE_CONFIG_PATH)} {_shown(config_path)}\n"
            f"然后在 providers.<服务商>.api_key 里填写你的密钥。"
        )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件 {config_path} 的顶层必须是映射（key: value）。")

    backend = raw.get("backend") or {}
    mode = str(backend.get("mode", "api")).strip().lower()
    if mode not in API_MODES:
        raise ConfigError(f"backend.mode 只能是 {API_MODES} 之一，当前是 {mode!r}。")

    providers: dict[str, ProviderConfig] = {}
    for name, spec in (raw.get("providers") or {}).items():
        spec = spec or {}
        providers[str(name)] = ProviderConfig(
            name=str(name),
            api_key=str(spec.get("api_key") or ""),
            base_url=str(spec.get("base_url") or "").rstrip("/"),
            model=str(spec.get("model") or ""),
            timeout_seconds=_as_float(spec.get("timeout_seconds"), 120.0),
            poll_interval_seconds=_as_float(spec.get("poll_interval_seconds"), 5.0),
            poll_timeout_seconds=_as_float(spec.get("poll_timeout_seconds"), 300.0),
            max_retries=_as_int(spec.get("max_retries"), 1),
        )

    gen_raw = raw.get("generation") or {}
    generation = GenerationConfig(
        width=_as_int(gen_raw.get("width"), 1088),
        height=_as_int(gen_raw.get("height"), 1088),
        output_format=str(gen_raw.get("output_format") or "png").lower(),
        style_prompt=str(gen_raw.get("style_prompt") or "").strip(),
        negative_prompt=str(gen_raw.get("negative_prompt") or "").strip(),
    )

    cfg = ApiConfig(mode=mode, provider=str(backend.get("provider") or "aliyun"),
                    providers=providers, generation=generation, path=config_path)

    if require_key and mode == "api":
        provider = cfg.active_provider()
        if not provider.has_key:
            raise ConfigError(
                f"服务商 {provider.name!r} 尚未配置 API Key。\n"
                f"请编辑 {config_path}，填写：\n"
                f"    providers:\n"
                f"      {provider.name}:\n"
                f"        api_key: \"你的密钥\"\n"
                f"（该文件已被 .gitignore 忽略，不会进入 Git。）"
            )
    return cfg
