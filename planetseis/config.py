"""Single source of truth for pipeline hyperparameters.

Every experiment (training, evaluation, the web app) reads from this module so
that preprocessing can never drift between train time and serve time (PRD F-16).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
RUNS_DIR = PROJECT_ROOT / "runs"

# Continuous Apollo archive cache (~21 GB for four stations). Overridable via
# PLANETSEIS_ARCHIVE because the repo normally lives under OneDrive, and
# handing a cloud-sync client tens of gigabytes of incompressible float32
# causes exactly the RAM/IO stalls that have bitten this project before.
# Point it at a non-synced drive; everything else stays where it is.
ARCHIVE_CACHE = Path(os.environ.get("PLANETSEIS_ARCHIVE",
                                    str(DATA_CACHE / "archive")))

# Root of the extracted NASA Space Apps 2024 packet.
PACKET_ROOT = DATA_RAW / "space_apps_2024_seismic_detection"

SEED = 42


@dataclass
class PreprocConfig:
    # Apollo PSE long-period native rate. InSight (20 Hz) is downsampled to
    # this so both bodies share one input distribution and one model.
    target_rate_hz: float = 6.625
    # Below Nyquist (3.31 Hz); keeps the Martian 2.4 Hz resonance band and the
    # 0.5-1.0 Hz band where Apollo events carry most energy.
    band_hz: tuple = (0.5, 3.0)


@dataclass
class WindowConfig:
    n_samples: int = 8192          # ~1236 s at 6.625 Hz — lunar events ring for
    hop: int = 4096                # tens of minutes; long context separates
                                   # onset (rise from quiet) from coda. 50 % overlap.
    # A window is labeled positive only if the catalog pick falls inside the
    # central region — picks at the very edge are ambiguous (PRD F-12).
    edge_margin_frac: float = 0.05
    # Scoring tolerance when matching a detection to a catalog pick. The
    # lunar catalog picks are minute-quantized (all time_rel % 60 == 0) and
    # onsets are emergent (PRD F-7), so tolerance must exceed the 60 s label
    # resolution: +/-120 s, stated in the report.
    match_tolerance_sec: float = 120.0


# Ring-down duration after an event onset. Windows inside the coda are
# neither positive (onset not in window) nor negative (they are full of real
# event energy) — training on them as negatives is contradictory supervision.
# They are excluded from training and suppressed at detection time.
# Lunar events ring for tens of minutes (high scattering, no attenuation);
# the packet's Martian events last a few minutes.
CODA_SEC = {"lunar": 1800.0, "mars": 300.0, "mars_ext": 300.0,
            "lunar_dn": 1800.0}   # denoised twin of the lunar cache


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # negatives per positive in the sampled training set (full continuous
    # streams are used for evaluation, never this balanced subset)
    neg_pos_ratio: float = 4.0
    lambda_reg: float = 2.0        # weight of the arrival-offset loss
    augment: bool = True


@dataclass
class Config:
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT = Config()
