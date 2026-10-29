"""Override only FLA's chunk module in profiling subprocesses.

Add this directory to PYTHONPATH and set GDN_CHUNK_SOURCE to a source snapshot.
The installed package and the vLLM checkout remain untouched.
"""

import importlib.abc
import importlib.util
import os
import sys


class ChunkSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm.third_party.flash_linear_attention.ops.chunk":
            return importlib.util.spec_from_file_location(
                fullname, os.environ["GDN_CHUNK_SOURCE"]
            )
        return None


if os.environ.get("GDN_CHUNK_SOURCE"):
    sys.meta_path.insert(0, ChunkSourceFinder())
