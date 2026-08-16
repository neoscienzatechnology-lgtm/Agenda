"""Configuração central do Pedigrafia Digital.

Todos os limiares metrológicos e de qualidade vivem aqui e são sobrescritíveis por
variável de ambiente (prefixo ``PEDIGRAFIA_``). Nenhum limiar deve ser codificado
literalmente em outro módulo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Constantes físicas exatas — NUNCA derivar estes valores de outro lugar.
# --------------------------------------------------------------------------------------

MM_PER_INCH = 25.4
"""Definição exata da polegada (acordo internacional de 1959)."""

PT_PER_INCH = 72.0
"""Ponto PostScript/PDF."""

MM_TO_PT = PT_PER_INCH / MM_PER_INCH
"""2.834645669291339 — único fator mm→pt do sistema."""

PT_TO_MM = MM_PER_INCH / PT_PER_INCH

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_WIDTH_PT = A4_WIDTH_MM * MM_TO_PT   # 595.2755905511812
A4_HEIGHT_PT = A4_HEIGHT_MM * MM_TO_PT  # 841.8897637795277

MARKER_SIZE_MM = 50.0
"""Aresta física do marcador fiducial. Referência metrológica primária."""


def _env(name: str, default: Any, cast: type) -> Any:
    raw = os.environ.get(f"PEDIGRAFIA_{name.upper()}")
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if cast is list:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return cast(raw)


@dataclass(frozen=True)
class QualityThresholds:
    """Limiares do *quality gate*. Score 0–100; ``min_total_score`` libera a análise."""

    min_total_score: float = 65.0
    # Gates rígidos (reprovam independentemente do score agregado)
    require_marker: bool = True
    require_four_corners: bool = True
    require_marker_inside_frame: bool = True
    require_at_least_one_foot: bool = True
    require_feet_inside_frame: bool = True

    # Resolução
    min_src_px_per_mm: float = 1.6
    good_src_px_per_mm: float = 3.5
    min_megapixels: float = 0.6
    good_megapixels: float = 3.0

    # Nitidez / movimento
    # Calibrados contra cenas sintéticas nítidas x desfocadas (ver docs/QUALITY.md):
    # nítida ~47, desfoque leve ~11, desfoque forte ~8, tremor ~26.
    min_focus_score: float = 20.0       # variância do laplaciano normalizada
    good_focus_score: float = 46.0
    max_motion_anisotropy: float = 0.42  # 0 = isotrópico, 1 = borrão direcional total
    good_motion_anisotropy: float = 0.12

    # Exposição
    max_clipped_high_frac: float = 0.06
    max_clipped_low_frac: float = 0.10
    # A luminância média NÃO reprova sozinha: uma plataforma branca bem iluminada
    # é legitimamente clara. O que reprova é o recorte (clipping) de informação.
    target_mean_luma: float = 150.0
    max_mean_luma_deviation: float = 105.0

    # Reflexos especulares dentro da área útil
    max_glare_frac: float = 0.045
    good_glare_frac: float = 0.008

    # Contraste pé/fundo (separabilidade)
    min_foot_bg_contrast: float = 12.0
    good_foot_bg_contrast: float = 45.0

    # Perspectiva
    max_tilt_deg: float = 38.0
    good_tilt_deg: float = 12.0
    max_marker_skew: float = 0.30        # desvio relativo dos lados do quadrilátero

    # Margem mínima (mm) entre os pés e a borda da imagem retificada
    min_border_margin_mm: float = 2.0

    # Pesos do score agregado
    weights: dict = field(default_factory=lambda: {
        "marker": 22.0,
        "resolution": 12.0,
        "focus": 14.0,
        "motion": 10.0,
        "exposure": 8.0,
        "glare": 8.0,
        "contrast": 10.0,
        "perspective": 10.0,
        "framing": 6.0,
    })


@dataclass(frozen=True)
class GeometryThresholds:
    """Limiares geométricos, todos em unidades físicas."""

    min_foot_area_mm2: float = 3500.0
    max_foot_area_mm2: float = 45000.0
    min_foot_length_mm: float = 90.0
    max_foot_length_mm: float = 400.0
    min_foot_elongation: float = 1.55       # razão eixo maior/menor da elipse ajustada
    max_foot_elongation: float = 4.6

    contour_resample_step_mm: float = 0.5   # passo do contorno de alta resolução
    smoothing_window_mm: float = 4.0        # janela da suavização preservando dimensão
    max_smoothing_length_drift_mm: float = 0.5
    simplify_max_error_mm: float = 0.30     # erro máximo dos nós editáveis
    simplify_target_nodes: int = 96
    simplify_min_nodes: int = 40
    simplify_max_nodes: int = 160

    toe_peak_min_prominence_mm: float = 1.4
    toe_peak_min_separation_mm: float = 5.5
    toe_region_t_start: float = 0.72        # busca de ápices dos pododáctilos

    forefoot_band: tuple = (0.55, 0.88)
    midfoot_band: tuple = (0.38, 0.62)
    heel_band: tuple = (0.02, 0.24)
    m1_band: tuple = (0.58, 0.82)
    m5_band: tuple = (0.52, 0.78)

    default_toe_cut_t: float = 0.78         # fallback do corte dos dedos (arch index)
    toe_valley_min_prominence_frac: float = 0.03


@dataclass(frozen=True)
class Settings:
    # --- Metrologia ---
    marker_size_mm: float = MARKER_SIZE_MM
    marker_dictionary: str = "DICT_4X4_50"
    marker_id: int = -1                     # -1 = aceita qualquer id do dicionário
    calibration_target: str = "auto"        # auto | single | board4 | caminho .json
    # Fora do casco convexo dos pontos de controle a homografia EXTRAPOLA, e o erro
    # cresce com a distância. Estes limiares transformam isso em aviso e bloqueio.
    max_extrapolation_warn_mm: float = 60.0
    # Extrapolar não é proibido — é menos exato, e o score reflete isso. Só bloqueia
    # em distâncias absurdas, onde a medida deixaria de ter qualquer significado.
    max_extrapolation_block_mm: float = 450.0
    rectified_px_per_mm: float = 6.0
    working_area_mm: float = 700.0          # janela física máxima retificada
    max_rectified_px: int = 4200            # teto de memória do raster retificado

    # --- Vista ---
    default_view: str = "below"             # podoscópio: câmera sob o vidro

    # --- Segmentação ---
    segmenter: str = "classical"
    seg_model_path: str = ""

    # --- Upload / sessão ---
    max_upload_bytes: int = 25 * 1024 * 1024
    allowed_mimes: tuple = ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif")
    session_ttl_seconds: int = 1800
    session_reaper_interval_seconds: int = 120
    storage_dir: str = ""                   # vazio => diretório temporário do SO

    # --- API ---
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")
    rate_limit_requests: int = 40
    rate_limit_window_seconds: int = 60
    debug_artifacts: bool = False           # grava estágios intermediários para /debug

    # --- PDF ---
    pdf_align_axis_to_page: bool = True
    pdf_margin_mm: float = 6.0
    pdf_author: str = "Pedigrafia Digital"

    quality: QualityThresholds = field(default_factory=QualityThresholds)
    geometry: GeometryThresholds = field(default_factory=GeometryThresholds)

    def as_public_dict(self) -> dict:
        d = asdict(self)
        d.pop("storage_dir", None)
        d.pop("seg_model_path", None)
        return d

    @property
    def storage_path(self) -> Path:
        if self.storage_dir:
            p = Path(self.storage_dir)
        else:
            import tempfile
            p = Path(tempfile.gettempdir()) / "pedigrafia-sessions"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        marker_size_mm=_env("marker_size_mm", MARKER_SIZE_MM, float),
        marker_dictionary=_env("marker_dictionary", "DICT_4X4_50", str),
        marker_id=_env("marker_id", -1, int),
        calibration_target=_env("target", "auto", str),
        max_extrapolation_warn_mm=_env("max_extrapolation_warn_mm", 60.0, float),
        max_extrapolation_block_mm=_env("max_extrapolation_block_mm", 450.0, float),
        rectified_px_per_mm=_env("rectified_px_per_mm", 6.0, float),
        working_area_mm=_env("working_area_mm", 700.0, float),
        max_rectified_px=_env("max_rectified_px", 4200, int),
        default_view=_env("default_view", "below", str),
        segmenter=_env("segmenter", "classical", str),
        seg_model_path=_env("seg_model", "", str),
        max_upload_bytes=_env("max_upload_bytes", 25 * 1024 * 1024, int),
        session_ttl_seconds=_env("session_ttl_seconds", 1800, int),
        storage_dir=_env("storage_dir", "", str),
        cors_origins=tuple(_env("cors_origins",
                                ["http://localhost:5173", "http://127.0.0.1:5173"], list)),
        rate_limit_requests=_env("rate_limit_requests", 40, int),
        rate_limit_window_seconds=_env("rate_limit_window_seconds", 60, int),
        debug_artifacts=_env("debug_artifacts", False, bool),
        pdf_align_axis_to_page=_env("pdf_align_axis_to_page", True, bool),
        pdf_margin_mm=_env("pdf_margin_mm", 6.0, float),
    )
