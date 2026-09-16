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

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeAlias

import torch
from torch.cuda import Event, Stream
from torch.distributed import Work

from ._utils import wrap_to_list


GeneralWorkItem: TypeAlias = Work | Stream | Event | None


@dataclass
class GeneralWork:
    work: GeneralWorkItem | Sequence[GeneralWorkItem] | "GeneralWork" = None

    def __post_init__(self):
        self.work_list: list[GeneralWorkItem] = wrap_to_list(self.work)

    def wait(self) -> None:
        for work in self.work_list:
            match work:
                case GeneralWork():  # recursively wait
                    work.wait()
                case Stream():
                    torch.cuda.current_stream().wait_stream(work)
                case Event():
                    torch.cuda.current_stream().wait_event(work)
                case Work():
                    # NOTE: WorkNCCL::wait only blocks the current stream
                    # on the NCCL stream by default
                    # unless in blocking mode, in which it will block CPU as well
                    work.wait()
                case None:
                    pass
                case _ if callable(getattr(work, "wait", None)):
                    # TorchBind custom classes returned by extension ops are
                    # represented as torch.ScriptObject rather than the
                    # Python ``torch.distributed.Work`` wrapper.  They still
                    # implement the same stream-ordering ``wait`` contract.
                    work.wait()
                case _:
                    raise TypeError(f"Unsupported type: {type(work)=}")


@dataclass
class WorkWithPostProcessFn:
    """Single-use work plus a deferred post-processing callback.

    Public *_async entries return this adapter. Their callers invoke
    wait_post_process() once, on the same current stream used at launch, before
    consuming results or reusing a native slot. It returns the supplied output
    tensor for CP operations; no output storage is allocated by this adapter.
    Keep work until that call and protect data/plan lifetimes through GPU
    completion. _work_done prevents repeated use; it is not a GPU completion
    indicator. CUDA waits establish stream dependencies without implying
    CPU-visible completion; CPU/Gloo completion may block the calling thread.
    The group_cast/group_reduce entries perform this step internally
    and continue returning None. No automatic destructor wait is performed.
    """

    work: GeneralWork | None = None
    post_process_fn: Callable = field(
        default_factory=lambda: lambda *args, **kwargs: None
    )
    async_op: bool = False

    def __post_init__(self):
        # the flag to note if
        # the optional work + post process fn are both done
        # to avoid repeatedly calling post process fn
        self._work_done = False

        # if sync mode, the given work needs to wait immediately
        if not self.async_op:
            self._wait_work()

    def wait_post_process(self, *args, **kwargs) -> Any:
        """Wait for the work to be done,
        then call the post process fn with the given args

        NOTE: this is a one-time API
        """
        if self._work_done:
            raise RuntimeError("Work has already been done.")

        self._wait_work()

        return self._apply_post_process(*args, **kwargs)

    def _wait_work(self) -> None:
        if self.work is not None:
            self.work.wait()
            self.work = None

    def _apply_post_process(self, *args, **kwargs) -> Any:
        ret = self.post_process_fn(*args, **kwargs)

        # Release callback-owned buffers after submitting post-processing.
        self.post_process_fn = lambda *args, **kwargs: None

        self._work_done = True

        return ret
