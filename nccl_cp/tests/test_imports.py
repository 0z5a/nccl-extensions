import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_import_dependencies():
    allowed = {
        "__future__",
        "os",
        "socket",
        "threading",
        "math",
        "importlib",
        "contextlib",
        "bisect",
        "argparse",
        "json",
        "sysconfig",
        "dataclasses",
        "itertools",
        "typing",
        "torch",
        "hashlib",
        "collections",
        "functools",
        "pathlib",
        "pickle",
        "nccl",
    }
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names), path
            if isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in allowed, path


def test_lazy_import_without_jit(tmp_path):
    import nccl.cp

    code = """
import sys
import nccl.cp
assert "create_handle" in nccl.cp.__all__
assert "torch" not in sys.modules
from nccl.cp import comm_meta, work, CpConfig, CpGroup, create_group
from nccl.cp import zero_cta as backend
import torch
assert not torch.cuda.is_initialized()
config = CpConfig(max_per_peer_slot=4, payload_shape=(32, 128), dtype=torch.bfloat16)
assert config.max_per_token_bytes == 8192
assert backend._get_extension.cache_info().currsize == 0
assert not hasattr(work, "EventOverlap")
assert comm_meta.ZeroCTACollectiveArg.__name__ == "ZeroCTACollectiveArg"
"""
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path / "jit"))
    script = tmp_path / "check_imports.py"
    script.write_text(code)
    subprocess.run(
        [sys.executable, str(ROOT / "tools/run.py"), "--package-dir",
         str(Path(nccl.cp.__file__).parent), str(script)],
        cwd=tmp_path, env=env, check=True,
    )
    assert not (tmp_path / "jit").exists()


def test_native_sources_are_bundled_with_runtime():
    from nccl.cp import zero_cta

    package_dir = Path(zero_cta.__file__).resolve().parent
    for name in ("zero_cta_collective.cpp", "zero_cta_kernels.cu", "zero_cta_kernels.h"):
        assert (package_dir / name).is_file()
        assert (package_dir / name).read_bytes() == (ROOT / "src" / name).read_bytes()
