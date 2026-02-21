from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


MIN_MLX_AUDIO_VERSION = (0, 3, 0)
MLX_AUDIO_DIST_NAME = 'mlx-audio'
MLX_AUDIO_MODEL_RELATIVE = Path('mlx_audio/tts/models/qwen3_tts')
INSTALL_HINT = (
    'Activate runtime/.venv and run: '
    'pip uninstall -y mlx-lm mlx-audio && '
    'pip install --upgrade --force-reinstall -r python_service/requirements-mlx.txt'
)


@dataclass(slots=True)
class RuntimeStatus:
    installed: bool
    runtimeVersion: str | None
    supportsQwen3Tts: bool
    minRequiredVersion: str
    compatible: bool
    reason: str


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in raw.split('.'):
        digits = ''.join(ch for ch in item if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def _is_at_least(version_raw: str, minimum: tuple[int, ...]) -> bool:
    current = _version_tuple(version_raw)
    if not current:
        return False
    padded_current = current + (0,) * max(0, len(minimum) - len(current))
    return padded_current[: len(minimum)] >= minimum


def _has_qwen3_model(dist) -> bool:
    files = dist.files or []
    for entry in files:
        rel = Path(str(entry))
        if rel == MLX_AUDIO_MODEL_RELATIVE or str(rel).startswith(f'{MLX_AUDIO_MODEL_RELATIVE}/'):
            return True

    # Fallback for distributions that do not expose file lists.
    base = Path(dist.locate_file(''))
    candidate = base / MLX_AUDIO_MODEL_RELATIVE
    return candidate.exists()


def detect_runtime_status() -> RuntimeStatus:
    min_version_str = '.'.join(str(part) for part in MIN_MLX_AUDIO_VERSION)
    try:
        dist = distribution(MLX_AUDIO_DIST_NAME)
    except PackageNotFoundError:
        return RuntimeStatus(
            installed=False,
            runtimeVersion=None,
            supportsQwen3Tts=False,
            minRequiredVersion=min_version_str,
            compatible=False,
            reason=f'{MLX_AUDIO_DIST_NAME} is not installed. {INSTALL_HINT}',
        )

    version_raw = dist.version or 'unknown'
    version_ok = _is_at_least(version_raw, MIN_MLX_AUDIO_VERSION)
    has_qwen3 = _has_qwen3_model(dist)

    if not version_ok:
        return RuntimeStatus(
            installed=True,
            runtimeVersion=version_raw,
            supportsQwen3Tts=has_qwen3,
            minRequiredVersion=min_version_str,
            compatible=False,
            reason=(
                f'{MLX_AUDIO_DIST_NAME} {version_raw} is too old for qwen3_tts. '
                f'Required >= {min_version_str}. {INSTALL_HINT}'
            ),
        )

    if not has_qwen3:
        return RuntimeStatus(
            installed=True,
            runtimeVersion=version_raw,
            supportsQwen3Tts=False,
            minRequiredVersion=min_version_str,
            compatible=False,
            reason=(
                f'{MLX_AUDIO_DIST_NAME} {version_raw} is installed but qwen3_tts model handlers are missing. '
                f'{INSTALL_HINT}'
            ),
        )

    return RuntimeStatus(
        installed=True,
        runtimeVersion=version_raw,
        supportsQwen3Tts=True,
        minRequiredVersion=min_version_str,
        compatible=True,
        reason='MLX runtime compatible for qwen3_tts.',
    )
