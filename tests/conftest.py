"""Test configuration and fallback mocks for environment without torch."""

import sys
from unittest.mock import MagicMock

try:
    import torch
except ImportError:
    torch = MagicMock()
    torch.nn = MagicMock()
    torch.nn.Module = object
    torch.autograd = MagicMock()
    torch.autograd.Function = object
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.autograd"] = torch.autograd

try:
    import transformers
except ImportError:
    sys.modules["transformers"] = MagicMock()
