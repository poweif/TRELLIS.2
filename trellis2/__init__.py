import os
import torch

# ROCm runtime defaults for this fork's target (Strix Halo, gfx1151). Set here, before the
# submodule imports below, so every entry point gets them; flash_attn reads its variable at
# import time. setdefault keeps any value the user exported.
if torch.version.hip is not None:
    # Required for the ROCm Triton-based flash-attn backend (no HIP binary for gfx1151).
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    # MIOpen on gfx1151 fails to load the Winograd kernel assembly when benchmarking
    # new convolution shapes, raising miopenStatusUnknownError. Disable Winograd so
    # MIOpen only considers GEMM/Direct algorithms that work on this GPU.
    os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")

from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils
