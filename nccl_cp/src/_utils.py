# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

def _make_device_tensor(
    values: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | int | None = None,
) -> torch.Tensor:
    """Materialize host metadata through pinned non-blocking H2D.

    The input dtype is preserved or inferred when ``dtype`` is omitted, and the
    current CUDA device is used when ``device`` is omitted.
    Internal caller contract: values are host metadata; use this only during
    preparation, never from a payload submission path.
    """

    if isinstance(values, torch.Tensor) and values.device.type != "cpu":
        raise ValueError("Metadata values must reside on CPU")
    host_tensor = torch.as_tensor(values, dtype=dtype, device="cpu")
    if device is None:
        device = torch.cuda.current_device()
    return host_tensor.pin_memory().to(device=device, non_blocking=True)


def wrap_to_list(x: Any, broadcast_to_length: int = 1) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x] * broadcast_to_length
