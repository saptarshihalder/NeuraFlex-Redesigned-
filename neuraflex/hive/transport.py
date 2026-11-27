"""Where the swarm meets. Any free blob store works as the parameter server.

    local:/path/to/shared/dir        (a shared folder, NFS, or a synced drive)
    hub:username/neuraflex-hive      (a Hugging Face model repo; free, versioned, resumable)
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile


class Transport:
    def put_bytes(self, path: str, data: bytes): ...
    def get_bytes(self, path: str) -> bytes | None: ...
    def put_file(self, path: str, local: str): ...
    def get_file(self, path: str, local: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete(self, path: str): ...


class LocalTransport(Transport):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _p(self, path):
        return os.path.join(self.root, path)

    def put_bytes(self, path, data):
        dst = self._p(path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst))
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)  # atomic: readers never see half a file

    def get_bytes(self, path):
        try:
            with open(self._p(path), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def put_file(self, path, local):
        dst = self._p(path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        shutil.copyfile(local, tmp)
        os.replace(tmp, dst)

    def get_file(self, path, local):
        src = self._p(path)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        shutil.copyfile(src, local)
        return True

    def list(self, prefix):
        """All files whose path starts with `prefix` (a directory or a filename prefix)."""
        base = self._p(prefix) if prefix.endswith("/") else os.path.dirname(self._p(prefix))
        out = []
        for d, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), self.root)
                if rel.startswith(prefix) and not f.endswith(".tmp") and not f.startswith("tmp"):
                    out.append(rel)
        return sorted(out)

    def delete(self, path):
        try:
            os.remove(self._p(path))
        except FileNotFoundError:
            pass


class HubTransport(Transport):
    """Hugging Face Hub repo as parameter server (needs `pip install huggingface_hub`
    and HF_TOKEN with write access). Every write is a commit, so keep sync windows
    long (tens of minutes), not seconds."""

    def __init__(self, repo_id: str, repo_type: str = "model"):
        from huggingface_hub import HfApi
        self.api, self.repo, self.type = HfApi(), repo_id, repo_type
        self.api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)

    def put_bytes(self, path, data):
        self.api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=path,
                             repo_id=self.repo, repo_type=self.type)

    def get_bytes(self, path):
        with tempfile.TemporaryDirectory() as d:
            return open(os.path.join(d, "f"), "rb").read() if self.get_file(path, os.path.join(d, "f")) else None

    def put_file(self, path, local):
        self.api.upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=self.repo, repo_type=self.type)

    def get_file(self, path, local):
        try:
            got = self.api.hf_hub_download(self.repo, path, repo_type=self.type, force_download=True)
        except Exception as e:  # EntryNotFoundError moved between huggingface_hub versions
            if "NotFound" in type(e).__name__:
                return False
            raise
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        shutil.copyfile(got, local)
        return True

    def list(self, prefix):
        return sorted(f for f in self.api.list_repo_files(self.repo, repo_type=self.type) if f.startswith(prefix))

    def delete(self, path):
        try:
            self.api.delete_file(path, repo_id=self.repo, repo_type=self.type)
        except Exception:
            pass


def make_transport(spec: str) -> Transport:
    kind, _, where = spec.partition(":")
    if kind == "local":
        return LocalTransport(where)
    if kind == "hub":
        return HubTransport(where)
    raise ValueError(f"unknown transport {spec!r} (use local:/dir or hub:user/repo)")
