"""Device compatibility helpers for SDAA / CUDA / CPU.

This module lets the MOSES code base run unchanged on three backends:

* ``sdaa``  -- Tecorigin SDAA accelerators (requires ``torch_sdaa``)
* ``cuda``  -- NVIDIA GPUs (requires a CUDA build of PyTorch)
* ``cpu``

Importing this module never requires ``torch_sdaa``: SDAA support is enabled
lazily, only when an SDAA device is requested or auto-detected.  On a CUDA
machine ``import torch_sdaa`` simply fails and the code falls back to CUDA.
"""

import importlib
import random
import time

import numpy as np
import torch


__all__ = [
    'sdaa_available', 'cuda_available', 'available_backends',
    'auto_device', 'resolve_device', 'build_device', 'seed_everything',
    'device_index', 'synchronize', 'warmup_device', 'DeviceTimer',
]

_SDAA_TRIED = False
_SDAA_OK = False


def _ensure_torch_sdaa():
    """Import ``torch_sdaa`` at most once; return True if usable."""
    global _SDAA_TRIED, _SDAA_OK
    if not _SDAA_TRIED:
        _SDAA_TRIED = True
        try:
            importlib.import_module('torch_sdaa')
            _SDAA_OK = hasattr(torch, 'sdaa')
        except Exception:
            _SDAA_OK = False
    return _SDAA_OK


def sdaa_available():
    return _ensure_torch_sdaa() and torch.sdaa.is_available()


def cuda_available():
    return torch.cuda.is_available()


def available_backends():
    backends = []
    if sdaa_available():
        backends.append('sdaa')
    if cuda_available():
        backends.append('cuda')
    backends.append('cpu')
    return backends


def auto_device():
    """Best accelerator in this process: ``sdaa`` > ``cuda`` > ``cpu``."""
    if sdaa_available():
        return 'sdaa'
    if cuda_available():
        return 'cuda'
    return 'cpu'


def _split_device(device):
    """Return ``(type, index_or_None)`` for a str/torch.device/None input."""
    if device is None:
        device = auto_device()
    if isinstance(device, torch.device):
        return device.type, device.index

    text = str(device).strip()
    if ':' in text:
        dev_type, _, dev_index = text.partition(':')
        return dev_type, int(dev_index) if dev_index != '' else None
    return text, None


def resolve_device(device=None):
    """Return a concrete device string, e.g. ``'sdaa:0'`` / ``'cuda:0'`` / ``'cpu'``.

    ``None`` (or ``'auto'``) selects the best available backend.  SDAA indices
    are normalised to 0 when omitted so downstream parsers always see ``:<n>``.
    """
    dev_type, dev_index = _split_device(device)

    if dev_type == 'sdaa':
        if not _ensure_torch_sdaa():
            raise RuntimeError(
                "SDAA device requested but 'torch_sdaa' is not importable")
        if not torch.sdaa.is_available():
            raise RuntimeError('SDAA device requested but no SDAA is available')
        return 'sdaa:{}'.format(0 if dev_index is None else dev_index)

    if dev_type == 'cuda':
        if not cuda_available():
            raise RuntimeError('CUDA device requested but CUDA is not available')
        return 'cuda:{}'.format(0 if dev_index is None else dev_index)

    if dev_type == 'cpu':
        return 'cpu'

    raise ValueError("Unknown device type: {!r}".format(device))


def device_index(device):
    """Return the integer index of a device string (0 for plain ``cpu``)."""
    _, index = _split_device(device)
    return 0 if index is None else index


def build_device(device=None):
    """Resolve ``device`` and select it for the current process.

    Returns a :class:`torch.device`.  For ``cuda``/``sdaa`` this also calls the
    matching ``set_device`` so later allocations default to the right card.
    """
    resolved = resolve_device(device)
    dev_type, dev_index = _split_device(resolved)

    if dev_type == 'sdaa':
        torch.sdaa.set_device(dev_index)
    elif dev_type == 'cuda':
        torch.cuda.set_device(dev_index)

    return torch.device(resolved)


def synchronize(device=None):
    """Block until all queued kernels on ``device`` have finished.

    No-op for ``cpu`` (and for backends that are not actually available), so
    callers can use it unconditionally around timing code.
    """
    dev_type, _ = _split_device(device)
    if dev_type == 'sdaa' and sdaa_available():
        torch.sdaa.synchronize()
    elif dev_type == 'cuda' and cuda_available():
        torch.cuda.synchronize()


def warmup_device(device=None, size=8):
    """Force lazy accelerator initialisation so it is not counted in timings.

    The first kernel launched on SDAA/CUDA pays a one-off cost (device context
    creation, LAPACK/kernel module loading). This runs a tiny op and
    synchronises, so later timings measure steady-state work only. It is best
    effort and never raises. Returns the resolved :class:`torch.device`.
    """
    dev = build_device(device)
    try:
        x = torch.ones(size, size, device=dev)
        y = x @ x
        synchronize(dev)
        del x, y
    except Exception:
        pass
    return dev


class DeviceTimer:
    """Context manager measuring wall time with device synchronisation.

    The accelerator is synchronised both before the clock starts and after it
    stops, so asynchronously queued kernels are fully accounted for. Combine
    with :func:`warmup_device` to exclude one-off backend start-up cost.
    """

    def __init__(self, device=None, name=''):
        self.device = device
        self.name = name
        self.elapsed = None
        self._start = None

    def __enter__(self):
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self._start
        return False


def seed_everything(seed):
    """Seed python/numpy and every available torch backend.

    Seeding is best-effort per backend so this is safe on CPU-only, CUDA-only
    and SDAA machines alike.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if cuda_available():
        torch.cuda.manual_seed_all(seed)
    if sdaa_available():
        torch.sdaa.manual_seed_all(seed)

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
