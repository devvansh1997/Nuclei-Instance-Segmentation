"""
utils/logger.py

Central logging setup for the MobileSAM-LoRA nuclei segmentation project.

Provides:
    setup_logger(cfg, run_name)  -- initialise console + optional file logger
    log_system_info(logger)      -- GPU, CUDA, PyTorch, Python versions
    log_config(logger, cfg)      -- pretty-print the full YAML config

All other modules should obtain their logger via:
    import logging
    logger = logging.getLogger(__name__)

This keeps log output consistent and avoids duplicate handlers across imports.
"""

import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

# colorama is installed in work_env — used for coloured console output
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _COLORAMA_AVAILABLE = True
except ImportError:
    _COLORAMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour map for console log levels
# ---------------------------------------------------------------------------
_LEVEL_COLOURS = {
    "DEBUG":    Fore.CYAN    if _COLORAMA_AVAILABLE else "",
    "INFO":     Fore.GREEN   if _COLORAMA_AVAILABLE else "",
    "WARNING":  Fore.YELLOW  if _COLORAMA_AVAILABLE else "",
    "ERROR":    Fore.RED     if _COLORAMA_AVAILABLE else "",
    "CRITICAL": Fore.MAGENTA if _COLORAMA_AVAILABLE else "",
}
_RESET = Style.RESET_ALL if _COLORAMA_AVAILABLE else ""


class _ColouredConsoleFormatter(logging.Formatter):
    """
    Console formatter that prepends a colour code based on log level.
    Format: [HH:MM:SS] [LEVEL]  message
    """
    _FMT = "[%(asctime)s] [{colour}%(levelname)-8s{reset}] %(message)s"
    _DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        fmt = self._FMT.format(colour=colour, reset=_RESET)
        formatter = logging.Formatter(fmt, datefmt=self._DATE)
        return formatter.format(record)


class _FileFormatter(logging.Formatter):
    """
    File formatter — more verbose than console; includes module and line number.
    Format: [YYYY-MM-DD HH:MM:SS] [LEVEL   ] [module:line] message
    """
    _FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logger(cfg: dict, run_name: str) -> logging.Logger:
    """
    Initialise the root logger for a training run.

    Sets up:
      - A coloured StreamHandler (console) at the configured log level
      - An optional FileHandler writing to <output.save_dir>/<run_name>/run.log

    Parameters
    ----------
    cfg      : Full config dict (loaded from YAML).
    run_name : Human-readable run identifier, e.g. "fold_0" or "debug_fold_0".
               Used as the log file subdirectory name.

    Returns
    -------
    logging.Logger
        The root logger (name = "nuclei_seg").
    """
    debug_cfg  = cfg.get("debug", {})
    output_cfg = cfg.get("output", {})

    log_level_str  = debug_cfg.get("log_level", "INFO").upper()
    log_to_file    = debug_cfg.get("log_to_file", True)
    log_level      = getattr(logging, log_level_str, logging.INFO)

    # Use a named root logger so child loggers (__name__) inherit it cleanly
    logger = logging.getLogger("nuclei_seg")
    logger.setLevel(logging.DEBUG)          # Logger itself accepts everything;
                                            # handlers filter by their own level
    logger.handlers.clear()                 # Avoid duplicate handlers on re-init

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(_ColouredConsoleFormatter())
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # File handler (optional)
    # ------------------------------------------------------------------
    if log_to_file:
        save_dir   = Path(output_cfg.get("save_dir", "outputs"))
        log_dir    = save_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path   = log_dir / "run.log"

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)    # File always captures everything
        file_handler.setFormatter(_FileFormatter())
        logger.addHandler(file_handler)

        logger.info("Log file  : %s", log_path.resolve())

    logger.info("Log level : %s", log_level_str)
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log hardware, OS, Python, and deep-learning framework versions.

    Called once at the start of each run so logs are self-contained and
    any environment mismatch (e.g. wrong CUDA version on HPC) is immediately
    visible.
    """
    logger.info("=" * 60)
    logger.info("SYSTEM INFO")
    logger.info("=" * 60)

    # Python & OS
    logger.info("Python    : %s", sys.version.replace("\n", " "))
    logger.info("Platform  : %s", platform.platform())

    # PyTorch
    try:
        import torch
        logger.info("PyTorch   : %s", torch.__version__)
        logger.info("CUDA avail: %s", torch.cuda.is_available())

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info("GPU count : %d", device_count)
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                logger.info(
                    "GPU[%d]    : %s  |  %.1f GB VRAM  |  CUDA cap %d.%d",
                    i, props.name, vram_gb, props.major, props.minor,
                )
            logger.info("CUDA ver  : %s", torch.version.cuda)
            logger.info("cuDNN ver : %s", str(torch.backends.cudnn.version()))
        else:
            logger.warning("No CUDA-capable GPU detected — training will be very slow.")

    except ImportError:
        logger.error("PyTorch is not installed.")

    # torchvision
    try:
        import torchvision
        logger.info("torchvision: %s", torchvision.__version__)
    except ImportError:
        logger.warning("torchvision not found.")

    # timm (MobileSAM dependency — version-sensitive)
    try:
        import timm
        logger.info("timm      : %s", timm.__version__)
    except ImportError:
        logger.warning("timm not found — MobileSAM will fail to load.")

    # NumPy (2.x has breaking changes — worth flagging)
    try:
        import numpy as np
        logger.info("NumPy     : %s", np.__version__)
        major = int(np.__version__.split(".")[0])
        if major >= 2:
            logger.debug(
                "NumPy 2.x detected — np.bool/np.int/np.float are removed. "
                "Ensure all dependencies are compatible."
            )
    except ImportError:
        logger.error("NumPy is not installed.")

    logger.info("=" * 60)


def log_config(logger: logging.Logger, cfg: dict) -> None:
    """
    Pretty-print the full configuration dictionary to the logger.

    Each key is logged at INFO level so it appears in the log file
    regardless of whether DEBUG mode is active.
    """
    try:
        import yaml
        cfg_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback: manual formatting if PyYAML somehow unavailable
        cfg_str = _dict_to_str(cfg)

    logger.info("=" * 60)
    logger.info("CONFIG")
    logger.info("=" * 60)
    for line in cfg_str.strip().splitlines():
        logger.info("  %s", line)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_str(d: dict, indent: int = 0) -> str:
    """Fallback recursive dict formatter (used if PyYAML is unavailable)."""
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_dict_to_str(v, indent + 1))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)
