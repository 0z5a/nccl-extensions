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

"""Shared enum and native-object type names.

See _types.pyi for distinct static types, registered methods and storage
schemas. At runtime both native objects are torch.ScriptObject instances;
these aliases do not load the extension or wrap its objects. Their concrete
TorchBind classes are selected by the native Runtime/Plan factories.
"""

from typing import Literal, TypeAlias

import torch

GroupReduceOp: TypeAlias = Literal["sum", "avg", "lse"]
ZeroCtaRuntime: TypeAlias = torch.ScriptObject
ZeroCtaPlan: TypeAlias = torch.ScriptObject
