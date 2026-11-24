# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for all backend implementations: disk and s3."""

from __future__ import annotations

import errno
import ctypes
import logging
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovannotators.backends._registry import BackendRegistry
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
)


# ===========================================================================
# DiskBackend
# ===========================================================================


class TestDiskBackend:
    def test_write_and_read(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        backend.write("test.bin", b"hello")
        assert backend.read("test.bin") == b"hello"

    def test_write_creates_parent_dirs(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        backend.write("a/b/c/test.bin", b"deep")
        assert backend.read("a/b/c/test.bin") == b"deep"

    def test_write_raises_if_exists_and_no_overwrite(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=False))
        backend.write("x.bin", b"first")
        with pytest.raises(BackendWriteError):
            backend.write("x.bin", b"second")

    def test_write_overwrites_when_enabled(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        backend.write("x.bin", b"first")
        backend.write("x.bin", b"second")
        assert backend.read("x.bin") == b"second"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_overwrite_rewrites_the_existing_file_in_place(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        private = tmp_path / "private.bin"
        private.write_bytes(b"old")
        private.chmod(0o640)
        before = private.stat()

        backend.write("private.bin", b"new")

        after = private.stat()
        assert private.read_bytes() == b"new"
        assert os.path.samestat(before, after)
        assert stat.S_IMODE(after.st_mode) == 0o640
        assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
        assert list(tmp_path.iterdir()) == [private]

    @pytest.mark.skipif(os.name != "posix", reason="POSIX write authorization")
    def test_overwrite_requires_destination_write_access(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root can open read-only files for writing")
        dest = tmp_path / "existing"
        dest.write_bytes(b"before")
        dest.chmod(0o444)
        with pytest.raises(PermissionError):
            os.close(os.open(dest, os.O_WRONLY))
        backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
        with pytest.raises(BackendWriteError):
            backend.write("existing", b"after")
        assert dest.read_bytes() == b"before"

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACLs")
    def test_overwrite_preserves_macos_deny_read_acl(self, tmp_path):
        import pwd

        if os.geteuid() == 0:
            pytest.skip("requires an unprivileged file owner")
        dest = tmp_path / "existing"
        dest.write_bytes(b"before")
        user = pwd.getpwuid(os.geteuid()).pw_name
        subprocess.run(["chmod", "+a", f"{user} deny read", str(dest)], check=True)
        backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
        backend.write("existing", b"after")
        with pytest.raises(PermissionError):
            dest.read_bytes()
        subprocess.run(["chmod", "-a#", "0", str(dest)], check=True)
        assert dest.read_bytes() == b"after"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows NTFS DACLs")
    def test_overwrite_preserves_a_windows_deny_read_dacl(self, tmp_path):
        dest = tmp_path / "existing"
        dest.write_bytes(b"before")
        subprocess.run(
            ["icacls", str(dest), "/deny", "*S-1-1-0:(RD)"],
            check=True,
            capture_output=True,
        )
        with pytest.raises(PermissionError):
            dest.read_bytes()
        backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
        backend.write("existing", b"after")
        with pytest.raises(PermissionError):
            dest.read_bytes()
        subprocess.run(
            ["icacls", str(dest), "/remove:d", "*S-1-1-0"],
            check=True,
            capture_output=True,
        )
        assert dest.read_bytes() == b"after"

    def test_windows_staging_directory_uses_private_inherited_access(
        self, tmp_path, monkeypatch
    ):
        from ovannotators._internal import filesystem

        def convert(sddl, revision, result, size):
            assert sddl == "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)"
            result._obj.value = 1234
            return 1

        def create(path, attributes):
            assert path == str(tmp_path / "staging")
            assert attributes._obj.descriptor == 1234
            assert not attributes._obj.inherit_handle
            return 0

        kernel = SimpleNamespace(
            CreateDirectoryW=MagicMock(side_effect=create), LocalFree=MagicMock()
        )
        security = SimpleNamespace(
            ConvertStringSecurityDescriptorToSecurityDescriptorW=MagicMock(
                side_effect=convert
            )
        )
        libraries = {"kernel32": kernel, "advapi32": security}
        monkeypatch.setattr(filesystem, "os", SimpleNamespace(name="nt"))
        monkeypatch.setattr(
            ctypes, "WinDLL", lambda name, **kwargs: libraries[name], raising=False
        )
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
        monkeypatch.setattr(ctypes, "WinError", PermissionError, raising=False)
        with pytest.raises(PermissionError):
            filesystem.make_private_directory(tmp_path / "staging")
        assert kernel.LocalFree.call_args.args[0].value == 1234

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux POSIX ACL representation"
    )
    @pytest.mark.parametrize("named_reader", [None, 12346])
    def test_overwrite_keeps_the_linux_access_acl(self, tmp_path, named_reader):
        def access_acl(uid):
            # Linux posix_acl_xattr_header v2 followed by tag, permissions, ID.
            undefined = 0xFFFFFFFF
            entries = [
                (1, 6, undefined),
                (2, 4, uid),
                (4, 0, undefined),
                (16, 4, undefined),
                (32, 0, undefined),
            ]
            return struct.pack("<I", 2) + b"".join(
                struct.pack("<HHI", *entry) for entry in entries
            )

        dest = tmp_path / "private.bin"
        dest.write_bytes(b"original")
        dest.chmod(0o640)
        try:
            os.setxattr(tmp_path, "system.posix_acl_default", access_acl(12345))
        except OSError as exc:
            if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip("filesystem does not support POSIX ACLs")
            raise
        name = "system.posix_acl_access"
        if named_reader is not None:
            os.setxattr(dest, name, access_acl(named_reader))
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))

        backend.write("private.bin", b"new")

        assert dest.read_bytes() == b"new"
        assert stat.S_IMODE(dest.stat().st_mode) == 0o640
        if named_reader is None:
            assert name not in os.listxattr(dest)
        else:
            assert os.getxattr(dest, name) == access_acl(named_reader)

    def test_create_if_absent_publishes_once_and_keeps_the_first_bytes(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))

        assert backend.write_if_absent("a/new.bin", b"data") is True
        assert backend.write_if_absent("a/new.bin", b"again") is False

        assert backend.read("a/new.bin") == b"data"
        assert [p.name for p in (tmp_path / "a").iterdir()] == ["new.bin"]

    def test_create_if_absent_falls_back_to_a_hard_link(self, tmp_path, monkeypatch):
        from ovannotators.backends import disk

        monkeypatch.setattr(disk, "_rename_without_overwrite", lambda *_: False)
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))

        assert backend.write_if_absent("a/new.bin", b"data") is True

        assert backend.read("a/new.bin") == b"data"
        assert [p.name for p in (tmp_path / "a").iterdir()] == ["new.bin"]

    def test_create_if_absent_reports_a_link_failure_other_than_missing_support(
        self, tmp_path, monkeypatch
    ):
        from ovannotators.backends import disk

        def denied(*_args, **_kwargs):
            raise PermissionError(errno.EACCES, "access denied")

        monkeypatch.setattr(disk, "_rename_without_overwrite", lambda *_: False)
        monkeypatch.setattr(disk.os, "link", denied)
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))

        with pytest.raises(BackendWriteError, match="access denied"):
            backend.write_if_absent("x.bin", b"data")

        assert list(tmp_path.iterdir()) == []

    def test_create_if_absent_keeps_a_concurrent_writers_file(
        self, tmp_path, monkeypatch
    ):
        from ovannotators.backends import disk

        def lose_the_race(_source, destination):
            Path(destination).write_bytes(b"other writer")
            raise FileExistsError(errno.EEXIST, "exists", str(destination))

        monkeypatch.setattr(disk, "_rename_without_overwrite", lambda *_: False)
        monkeypatch.setattr(disk.os, "link", lose_the_race)
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))

        assert backend.write_if_absent("x.bin", b"mine") is False

        assert (tmp_path / "x.bin").read_bytes() == b"other writer"
        assert [p.name for p in tmp_path.iterdir()] == ["x.bin"]

    def test_failed_temporary_cleanup_does_not_fail_a_committed_write(
        self, tmp_path, monkeypatch, caplog
    ):
        from ovannotators.backends import disk

        real_unlink = Path.unlink

        def busy_temporary(self, *args, **kwargs):
            if self.name.endswith(".tmp"):
                raise PermissionError(errno.EACCES, "busy")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(disk, "_rename_without_overwrite", lambda *_: False)
        monkeypatch.setattr(Path, "unlink", busy_temporary)
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))

        with caplog.at_level(logging.WARNING, logger="ovannotators.backends.disk"):
            assert backend.write_if_absent("x.bin", b"committed") is True

        assert (tmp_path / "x.bin").read_bytes() == b"committed"
        assert "Could not remove temporary file" in caplog.text
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "x.bin"]
        assert len(leftovers) == 1 and leftovers[0].startswith(".ovannotators-")
        assert backend.list() == ["x.bin"]

    def test_failed_create_if_absent_leaves_nothing_and_can_be_retried(
        self, tmp_path, monkeypatch
    ):
        from ovannotators.backends import disk

        real_publish = disk._publish_without_overwrite

        def disk_full(*_args):
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(disk, "_publish_without_overwrite", disk_full)
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        with pytest.raises(BackendWriteError, match="no space"):
            backend.write_if_absent("x.bin", b"data")
        assert list(tmp_path.iterdir()) == []

        monkeypatch.setattr(disk, "_publish_without_overwrite", real_publish)
        assert backend.write_if_absent("x.bin", b"data") is True
        assert (tmp_path / "x.bin").read_bytes() == b"data"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX setgid directories")
    def test_create_if_absent_inherits_a_setgid_directory_group(self, tmp_path):
        other = [gid for gid in os.getgroups() if gid != os.getegid()]
        if not other:
            pytest.skip("requires a supplementary group")
        shared = tmp_path / "shared"
        shared.mkdir()
        try:
            os.chown(shared, -1, other[0])
            shared.chmod(0o2775)
        except PermissionError:
            pytest.skip("cannot assign the supplementary group")
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))

        assert backend.write_if_absent("shared/frame.bin", b"data") is True

        assert (shared / "frame.bin").stat().st_gid == other[0]

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux POSIX ACL representation"
    )
    def test_create_if_absent_inherits_the_directory_default_acl(self, tmp_path):
        undefined = 0xFFFFFFFF
        entries = [
            (1, 6, undefined),
            (2, 4, 12345),
            (4, 4, undefined),
            (16, 4, undefined),
            (32, 4, undefined),
        ]
        default_acl = struct.pack("<I", 2) + b"".join(
            struct.pack("<HHI", *entry) for entry in entries
        )
        try:
            os.setxattr(tmp_path, "system.posix_acl_default", default_acl)
        except OSError as exc:
            if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip("filesystem does not support POSIX ACLs")
            raise
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))

        assert backend.write_if_absent("frame.bin", b"data") is True

        acl = os.getxattr(tmp_path / "frame.bin", "system.posix_acl_access")
        inherited = [
            struct.unpack_from("<HHI", acl, offset) for offset in range(4, len(acl), 8)
        ]
        assert any(tag == 2 and uid == 12345 for tag, _, uid in inherited)

    @pytest.mark.parametrize("winning_bytes", [b"mine", b"other producer"])
    def test_immutable_write_matches_a_competitor_without_replacing_it(
        self, tmp_path, monkeypatch, winning_bytes
    ):
        from ovannotators.backends import disk
        from ovannotators.profile_writers._artifacts import write_matching_or_new

        real_rename = disk._rename_without_overwrite
        backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))

        def commit_competitor(staged, destination):
            assert backend.list() == []
            destination.write_bytes(winning_bytes)
            return real_rename(staged, destination)

        monkeypatch.setattr(disk, "_rename_without_overwrite", commit_competitor)
        if winning_bytes == b"mine":
            write_matching_or_new(backend, "frame", b"mine")
        else:
            with pytest.raises(ValueError, match="different bytes"):
                write_matching_or_new(backend, "frame", b"mine")
        assert backend.read("frame") == winning_bytes
        assert backend.list() == ["frame"]

    def test_immutable_write_requires_an_atomic_filesystem(self, tmp_path, monkeypatch):
        from ovannotators.backends import disk

        def unsupported(*args):
            raise OSError(errno.ENOTSUP, "hard links unavailable")

        monkeypatch.setattr(disk, "_rename_without_overwrite", lambda *_: False)
        monkeypatch.setattr(disk.os, "link", unsupported)
        backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
        with pytest.raises(BackendWriteError, match="atomic create-if-absent"):
            backend.write_if_absent("_ovannotators/_SUCCESS", b"complete")
        assert backend.list() == []

    def test_read_nonexistent_raises(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        with pytest.raises(BackendReadError):
            backend.read("nofile.bin")

    def test_exists(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        assert not backend.exists("foo.txt")
        backend.write("foo.txt", b"bar")
        assert backend.exists("foo.txt")

    def test_list(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        backend.write("dir/a.txt", b"a")
        backend.write("dir/b.txt", b"b")
        backend.write("other.txt", b"c")
        result = backend.list("dir")
        assert sorted(result) == ["dir/a.txt", "dir/b.txt"]

    def test_list_empty_dir(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        assert backend.list("nonexistent") == []

    def test_makedirs(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        backend.makedirs("some/deep/dir")
        assert (tmp_path / "some" / "deep" / "dir").is_dir()

    def test_absolute_paths_inside_absolute_root(self, tmp_path):
        root = tmp_path / "output"
        backend = DiskBackend(DiskBackendConfig(root_dir=root))
        directory = root / "nested"
        backend.makedirs(str(directory))
        assert directory.is_dir()

        path = directory / "data.bin"
        backend.write(str(path), b"inside")
        assert backend.exists(str(path))
        assert backend.read(str(path)) == b"inside"
        assert backend.list(str(directory)) == ["nested/data.bin"]

        array_path = directory / "values.npy"
        backend.write_array(str(array_path), np.array([1, 2, 3]))
        np.testing.assert_array_equal(np.load(array_path), [1, 2, 3])

    @pytest.mark.parametrize("path", ["../escape.bin", "/tmp/escape.bin"])
    def test_write_rejects_paths_outside_root(self, tmp_path, path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        with pytest.raises(
            BackendWriteError, match="must remain within backend root_dir"
        ):
            backend.write(path, b"escape")

    @pytest.mark.parametrize("encoding", ["npy", "png", "exr"])
    @pytest.mark.parametrize("name", ["../escape", "/tmp/escape"])
    def test_write_array_rejects_paths_outside_root(self, tmp_path, name, encoding):
        """The shared-encoder ``write_array`` path still goes through ``_abs``."""
        root = tmp_path / "root"
        root.mkdir()
        backend = DiskBackend(DiskBackendConfig(root_dir=root))
        array = np.zeros((4, 4, 3), dtype=np.uint8 if encoding == "png" else np.float32)
        with pytest.raises(
            BackendWriteError, match="must remain within backend root_dir"
        ):
            backend.write_array(f"{name}.{encoding}", array, encoding=encoding)
        assert not (tmp_path / f"escape.{encoding}").exists()

    def test_write_array_rejects_symlink_escape(self, tmp_path):
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        self._symlink_or_skip(root / "linkdir", outside, is_dir=True)
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))
        with pytest.raises(BackendWriteError):
            backend.write_array("linkdir/new.npy", np.zeros(3), encoding="npy")
        assert not (outside / "new.npy").exists()

    @pytest.mark.parametrize("path", ["../escape.bin", "/tmp/escape.bin"])
    def test_read_and_list_reject_paths_outside_root(self, tmp_path, path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        with pytest.raises(
            BackendReadError, match="must remain within backend root_dir"
        ):
            backend.read(path)
        with pytest.raises(
            BackendReadError, match="must remain within backend root_dir"
        ):
            backend.list(path)

    @pytest.mark.parametrize("path", ["../escape", "/tmp/escape"])
    def test_exists_returns_false_and_makedirs_rejects_paths_outside_root(
        self, tmp_path, path
    ):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        assert not backend.exists(path)
        with pytest.raises(
            BackendWriteError, match="must remain within backend root_dir"
        ):
            backend.makedirs(path)

    def test_absolute_paths_outside_root_are_rejected_portably(self, tmp_path):
        # Built from tmp_path so the paths are valid absolute paths on every OS.
        root = tmp_path / "output"
        sibling = tmp_path / "output_evil"  # shares the root's name prefix
        outside = tmp_path / "elsewhere"
        for directory in (root, sibling, outside):
            directory.mkdir()
        (outside / "secret.bin").write_bytes(b"secret")
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))

        inside = root / "ok.bin"
        backend.write(str(inside), b"ok")
        assert backend.read(str(inside)) == b"ok"
        assert backend.exists(str(inside))

        for target in (outside / "secret.bin", sibling / "new.bin"):
            assert not backend.exists(str(target))
            with pytest.raises(BackendReadError):
                backend.read(str(target))
            with pytest.raises(BackendWriteError):
                backend.write(str(target), b"escape")
        with pytest.raises(BackendReadError):
            backend.list(str(outside))
        with pytest.raises(BackendWriteError):
            backend.makedirs(str(outside / "newdir"))
        assert not (sibling / "new.bin").exists()
        assert not (outside / "newdir").exists()
        assert (outside / "secret.bin").read_bytes() == b"secret"

    def test_relative_root_dir_is_pinned_at_construction(self, tmp_path, monkeypatch):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        monkeypatch.chdir(first)
        backend = DiskBackend(DiskBackendConfig(root_dir="out"))
        monkeypatch.chdir(second)
        backend.write("a.bin", b"a")
        assert (first / "out" / "a.bin").read_bytes() == b"a"
        assert not (second / "out").exists()

    def _symlink_or_skip(self, link, target, *, is_dir):
        try:
            link.symlink_to(target, target_is_directory=is_dir)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available on this platform/user")

    def test_symlink_to_outside_directory_is_rejected(self, tmp_path):
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        (outside / "secret.bin").write_bytes(b"secret")
        self._symlink_or_skip(root / "linkdir", outside, is_dir=True)
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))

        with pytest.raises(BackendReadError):
            backend.read("linkdir/secret.bin")
        with pytest.raises(BackendReadError):
            backend.list("linkdir")
        with pytest.raises(BackendWriteError):
            backend.write("linkdir/new.bin", b"escape")
        with pytest.raises(BackendWriteError):
            backend.makedirs("linkdir/sub")
        assert not backend.exists("linkdir/secret.bin")
        assert not (outside / "new.bin").exists()
        assert not (outside / "sub").exists()

    def test_symlink_to_outside_file_is_rejected(self, tmp_path):
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        secret = outside / "secret.bin"
        secret.write_bytes(b"secret")
        self._symlink_or_skip(root / "linkfile", secret, is_dir=False)
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))

        with pytest.raises(BackendReadError):
            backend.read("linkfile")
        with pytest.raises(BackendWriteError):
            backend.write("linkfile", b"overwrite")
        assert not backend.exists("linkfile")
        assert secret.read_bytes() == b"secret"

    def test_list_omits_symlinks_to_outside_files(self, tmp_path):
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        secret = outside / "secret.bin"
        secret.write_bytes(b"secret")
        (root / "real.bin").write_bytes(b"real")
        self._symlink_or_skip(root / "link_out", secret, is_dir=False)
        self._symlink_or_skip(root / "link_in", root / "real.bin", is_dir=False)
        backend = DiskBackend(DiskBackendConfig(root_dir=root))

        assert backend.list() == ["link_in", "real.bin"]
        with pytest.raises(BackendReadError):
            backend.read("link_out")
        assert backend.read("link_in") == b"real"

    def test_list_skips_broken_and_looping_symlinks(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "real.bin").write_bytes(b"real")
        self._symlink_or_skip(root / "broken", root / "missing.bin", is_dir=False)
        self._symlink_or_skip(root / "loop_a", root / "loop_b", is_dir=False)
        self._symlink_or_skip(root / "loop_b", root / "loop_a", is_dir=False)
        backend = DiskBackend(DiskBackendConfig(root_dir=root))

        assert backend.list() == ["real.bin"]

    def test_symlink_loop_is_rejected_without_unhandled_errors(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        self._symlink_or_skip(root / "loop_a", root / "loop_b", is_dir=True)
        self._symlink_or_skip(root / "loop_b", root / "loop_a", is_dir=True)
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))

        # Path.resolve() raises RuntimeError (3.10/3.11) or OSError (3.12) for
        # a loop, and may not raise at all on newer Pythons. Whatever it does,
        # callers see only the documented backend errors, never a raw
        # RuntimeError or OSError from resolution.
        with pytest.raises(BackendReadError):
            backend.read("loop_a/file.bin")
        with pytest.raises(BackendWriteError):
            backend.write("loop_a/file.bin", b"x")
        with pytest.raises(BackendWriteError):
            backend.makedirs("loop_a/sub")
        assert not backend.exists("loop_a/file.bin")
        try:
            assert backend.list("loop_a") == []
        except BackendReadError:
            pass

    def test_abs_converts_resolve_errors_to_value_error(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        for error in (RuntimeError("Symlink loop"), OSError(40, "Too many links")):
            with patch("pathlib.Path.resolve", side_effect=error):
                with pytest.raises(ValueError, match="cannot be resolved"):
                    backend._abs("a.bin")

    def test_symlink_inside_root_is_allowed(self, tmp_path):
        root = tmp_path / "root"
        (root / "real").mkdir(parents=True)
        (root / "real" / "data.bin").write_bytes(b"data")
        self._symlink_or_skip(root / "linkdir", root / "real", is_dir=True)
        self._symlink_or_skip(
            root / "linkfile", root / "real" / "data.bin", is_dir=False
        )
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))

        assert backend.read("linkdir/data.bin") == b"data"
        assert backend.read("linkfile") == b"data"
        assert backend.exists("linkdir/data.bin")
        backend.write("linkdir/new.bin", b"new")
        assert (root / "real" / "new.bin").read_bytes() == b"new"
        assert backend.list("linkdir") == ["real/data.bin", "real/new.bin"]

    def test_write_array_npy(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        arr = np.array([1.0, 2.0, 3.0])
        backend.write_array("data.npy", arr, encoding="npy")
        loaded = np.load(tmp_path / "data.npy")
        np.testing.assert_array_equal(loaded, arr)

    def test_write_array_png(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        backend.write_array("img.png", img, encoding="png")
        data = backend.read("img.png")
        assert data[:4] == b"\x89PNG"

    def test_write_array_jpg(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        backend.write_array("img.jpg", img, encoding="jpg")
        data = backend.read("img.jpg")
        assert data[:2] == b"\xff\xd8"

    def test_write_array_unsupported_encoding(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        arr = np.zeros((10, 10))
        with pytest.raises(BackendWriteError, match="Unsupported encoding"):
            backend.write_array("x.xyz", arr, encoding="xyz")

    def test_write_array_exr_import_error(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        with patch.dict("sys.modules", {"OpenEXR": None, "Imath": None}):
            with pytest.raises(BackendWriteError, match="install OpenEXR"):
                backend.write_array("x.exr", arr, encoding="exr")

    def test_write_oserror(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        with patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")):
            with pytest.raises(BackendWriteError, match="disk full"):
                backend.write("test.bin", b"data")

    def test_write_array_png_uses_shared_codec(self, tmp_path):
        """PNG encoding is delegated to the shared image codec."""
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        codec = MagicMock()
        codec.encode.return_value = b"png-bytes"
        with patch("ovannotators._internal.codecs.get_image_codec", return_value=codec):
            backend.write_array("img.png", arr, encoding="png")
        codec.encode.assert_called_once()
        assert codec.encode.call_args.kwargs == {"format": "png"}
        assert (tmp_path / "img.png").read_bytes() == b"png-bytes"

    def test_write_array_exr_success(self, tmp_path):
        """EXR path writes bytes from the shared EXR encoder helper."""
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes",
            return_value=b"fake_exr",
        ) as mock_encode:
            backend.write_array("depth.exr", arr, encoding="exr", exr_compression="piz")
        mock_encode.assert_called_once_with(arr, compression="piz")
        assert (tmp_path / "depth.exr").read_bytes() == b"fake_exr"

    def test_exr_compression_rejected_for_npy(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path))
        with pytest.raises(ValueError, match="requires encoding='exr'"):
            backend.write_array("data.npy", np.ones((2, 2)), exr_compression="piz")

    def test_write_array_exr_compression_rejects_paths_outside_root(self, tmp_path):
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        link = root / "linkdir"
        try:
            link.symlink_to(outside, target_is_directory=True)
            have_symlink = True
        except (OSError, NotImplementedError):
            have_symlink = False
        backend = DiskBackend(DiskBackendConfig(root_dir=root, overwrite=True))
        arr = np.zeros((2, 2, 3), dtype=np.float32)
        targets = ["../escape.exr", str(outside / "abs.exr")]
        if have_symlink:
            targets.append("linkdir/link.exr")
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes",
            return_value=b"fake_exr",
        ):
            for target in targets:
                with pytest.raises(
                    BackendWriteError, match="must remain within backend root_dir"
                ):
                    backend.write_array(
                        target, arr, encoding="exr", exr_compression="piz"
                    )
        assert not (tmp_path / "escape.exr").exists()
        assert list(outside.iterdir()) == []

    def test_makedirs_oserror(self, tmp_path):
        backend = DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))
        with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
            with pytest.raises(BackendWriteError):
                backend.makedirs("sub/dir")


# ===========================================================================
# S3Backend (mocked boto3)
# ===========================================================================


class TestS3Backend:
    @pytest.fixture
    def mock_boto3(self):
        with patch("ovannotators.backends.s3._BOTO3_AVAILABLE", True):
            with patch("ovannotators.backends.s3.boto3") as mock_b3:
                mock_client = MagicMock()
                mock_b3.client.return_value = mock_client
                mock_b3.s3.transfer.TransferConfig.return_value = MagicMock()
                yield mock_client

    def test_import_error_when_boto3_unavailable(self):
        with patch("ovannotators.backends.s3._BOTO3_AVAILABLE", False):
            from ovannotators.backends.s3 import S3Backend, S3BackendConfig

            with pytest.raises(ImportError, match="boto3"):
                S3Backend(S3BackendConfig(bucket="test"))

    def test_write(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="data"))
        backend.write("file.bin", b"content")
        mock_boto3.put_object.assert_called_once_with(
            Bucket="mybucket", Key="data/file.bin", Body=b"content"
        )

    def test_immutable_write_uses_server_side_precondition(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        botocore = pytest.importorskip("botocore.exceptions")
        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="data"))
        assert backend.write_if_absent("file.bin", b"first") is True
        mock_boto3.put_object.assert_called_once_with(
            Bucket="mybucket", Key="data/file.bin", Body=b"first", IfNoneMatch="*"
        )
        mock_boto3.put_object.side_effect = botocore.ClientError(
            {"ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject"
        )
        assert backend.write_if_absent("file.bin", b"second") is False

    @pytest.mark.parametrize("conflict", [False, True])
    def test_immutable_multipart_precondition_and_abort(
        self, mock_boto3, monkeypatch, conflict
    ):
        from ovannotators.backends import s3

        botocore = pytest.importorskip("botocore.exceptions")
        monkeypatch.setattr(s3, "_MAX_SINGLE_PUT_BYTES", 4)
        backend = s3.S3Backend(s3.S3BackendConfig(bucket="test"))
        backend._transfer_config.multipart_chunksize = 4
        mock_boto3.create_multipart_upload.return_value = {"UploadId": "upload"}
        mock_boto3.upload_part.side_effect = [{"ETag": "first"}, {"ETag": "last"}]
        if conflict:
            mock_boto3.complete_multipart_upload.side_effect = botocore.ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 412}}, "CompleteMultipartUpload"
            )
        assert backend.write_if_absent("clip.mp4", b"123456") is not conflict
        assert [
            call.kwargs["Body"] for call in mock_boto3.upload_part.call_args_list
        ] == [b"1234", b"56"]
        mock_boto3.complete_multipart_upload.assert_called_once_with(
            Bucket="test",
            Key="clip.mp4",
            UploadId="upload",
            IfNoneMatch="*",
            MultipartUpload={
                "Parts": [
                    {"PartNumber": 1, "ETag": "first"},
                    {"PartNumber": 2, "ETag": "last"},
                ]
            },
        )
        assert mock_boto3.abort_multipart_upload.call_count == int(conflict)

    def test_write_large_payload_uses_multipart_upload(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(
            S3BackendConfig(bucket="mybucket", prefix="data", multipart_threshold_mb=1)
        )
        payload = b"x" * (1024 * 1024)
        backend.write("video.mp4", payload)
        mock_boto3.put_object.assert_not_called()
        mock_boto3.upload_fileobj.assert_called_once()
        fileobj, bucket, key = mock_boto3.upload_fileobj.call_args.args
        assert (bucket, key) == ("mybucket", "data/video.mp4")
        assert fileobj.read() == payload
        assert (
            mock_boto3.upload_fileobj.call_args.kwargs["Config"]
            is backend._transfer_config
        )

    def test_multipart_write_array_applies_exr_compression(self, mock_boto3):
        """EXR compression is applied at encode time, so multipart matches single-put."""
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(
            S3BackendConfig(bucket="mybucket", multipart_threshold_mb=1)
        )
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        encoded = b"e" * (1024 * 1024)
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes", return_value=encoded
        ) as mock_encode:
            backend.write_array("img.exr", arr, encoding="exr", exr_compression="piz")
        mock_encode.assert_called_once_with(arr, compression="piz")
        mock_boto3.put_object.assert_not_called()
        fileobj = mock_boto3.upload_fileobj.call_args.args[0]
        assert fileobj.read() == encoded

    @pytest.mark.parametrize(
        ("threshold_mb", "chunksize_mb"),
        [(1, 5), (8, 8), (64, 64), (100, 64), (5120, 64)],
    )
    def test_multipart_chunksize_is_bounded(self, threshold_mb, chunksize_mb):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        with patch("ovannotators.backends.s3._BOTO3_AVAILABLE", True):
            with patch("ovannotators.backends.s3.boto3") as mock_b3:
                backend = S3Backend(
                    S3BackendConfig(bucket="b", multipart_threshold_mb=threshold_mb)
                )
                kwargs = mock_b3.s3.transfer.TransferConfig.call_args.kwargs
                assert kwargs["multipart_chunksize"] == chunksize_mb * 1024**2
                assert kwargs["multipart_threshold"] == threshold_mb * 1024**2
                assert backend._multipart_threshold == threshold_mb * 1024**2

    @pytest.mark.parametrize("threshold", [0, -1])
    def test_non_positive_multipart_threshold_rejected(self, mock_boto3, threshold):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig
        from ovannotators.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="multipart_threshold_mb"):
            S3Backend(
                S3BackendConfig(bucket="mybucket", multipart_threshold_mb=threshold)
            )

    def test_write_below_threshold_uses_put_object(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(
            S3BackendConfig(bucket="mybucket", multipart_threshold_mb=1)
        )
        backend.write("small.bin", b"x" * (1024 * 1024 - 1))
        mock_boto3.upload_fileobj.assert_not_called()
        mock_boto3.put_object.assert_called_once()

    def test_multipart_threshold_capped_at_single_put_limit(self):
        with patch("ovannotators.backends.s3._BOTO3_AVAILABLE", True):
            with patch("ovannotators.backends.s3.boto3") as mock_b3:
                client = MagicMock()
                mock_b3.client.return_value = client
                from ovannotators.backends.s3 import S3Backend, S3BackendConfig

                backend = S3Backend(
                    S3BackendConfig(bucket="b", multipart_threshold_mb=8 * 1024)
                )
                limit = 5 * 1024**3
                assert backend._multipart_threshold == limit
                kwargs = mock_b3.s3.transfer.TransferConfig.call_args.kwargs
                assert kwargs["multipart_threshold"] == limit
                # Routing keeps the 5 GiB cap; part size stays bounded.
                assert kwargs["multipart_chunksize"] == 64 * 1024**2
                # A 6 GiB payload would exceed the single-PUT limit; it must
                # take the multipart path even though it is below 8 GiB.
                payload = MagicMock(spec=bytes)
                payload.__len__.return_value = 6 * 1024**3
                with patch("ovannotators.backends.s3.io.BytesIO"):
                    backend.write("big.bin", payload)
                client.put_object.assert_not_called()
                client.upload_fileobj.assert_called_once()

    def test_write_error_raises(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        mock_boto3.put_object.side_effect = Exception("network error")
        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        with pytest.raises(BackendWriteError):
            backend.write("file.bin", b"data")

    def test_read(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        mock_body = MagicMock()
        mock_body.read.return_value = b"filedata"
        mock_boto3.get_object.return_value = {"Body": mock_body}
        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="pfx"))
        result = backend.read("test.txt")
        assert result == b"filedata"
        mock_boto3.get_object.assert_called_once_with(
            Bucket="mybucket", Key="pfx/test.txt"
        )

    def test_read_error_raises(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        mock_boto3.get_object.side_effect = Exception("not found")
        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        with pytest.raises(BackendReadError):
            backend.read("missing.bin")

    def test_list(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "pfx/a.txt"}, {"Key": "pfx/b.txt"}]}
        ]
        mock_boto3.get_paginator.return_value = paginator
        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="pfx"))
        result = backend.list("")
        assert sorted(result) == ["a.txt", "b.txt"]

    def test_list_error_raises(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        paginator = MagicMock()
        paginator.paginate.side_effect = Exception("access denied")
        mock_boto3.get_paginator.return_value = paginator
        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        with pytest.raises(BackendReadError):
            backend.list("")

    def test_exists_true(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        mock_boto3.head_object.return_value = {}
        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        assert backend.exists("file.bin") is True

    def test_exists_false(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        mock_boto3.head_object.side_effect = Exception("404")
        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        assert backend.exists("nofile.bin") is False

    def test_makedirs_is_noop(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        backend.makedirs("some/path")  # should not raise

    def test_write_array_npy(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        arr = np.array([1, 2, 3])
        backend.write_array("data.npy", arr, encoding="npy")
        mock_boto3.put_object.assert_called_once()
        call_kwargs = mock_boto3.put_object.call_args[1]
        assert call_kwargs["Key"] == "data.npy"

    def test_write_array_unsupported_encoding(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        with pytest.raises(BackendWriteError, match="Unsupported encoding"):
            backend.write_array("x.xyz", np.zeros(10), encoding="xyz")

    def test_write_array_png_uses_shared_codec(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        codec = MagicMock()
        codec.encode.return_value = b"png-bytes"
        with patch("ovannotators._internal.codecs.get_image_codec", return_value=codec):
            backend.write_array("img.png", arr, encoding="png")
        codec.encode.assert_called_once()
        assert codec.encode.call_args.kwargs == {"format": "png"}
        assert mock_boto3.put_object.call_args[1]["Body"] == b"png-bytes"

    def test_write_array_exr(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes", return_value=b"exr"
        ) as mock_encode:
            backend.write_array("img.exr", arr, encoding="exr", exr_compression="piz")
        mock_encode.assert_called_once_with(arr, compression="piz")
        call_kwargs = mock_boto3.put_object.call_args[1]
        assert call_kwargs["Body"] == b"exr"

    def test_exr_compression_rejected_for_non_exr(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        with pytest.raises(ValueError, match="requires encoding='exr'"):
            backend.write_array("a.npy", np.zeros((2, 2)), exr_compression="piz")
        mock_boto3.put_object.assert_not_called()

    def test_write_array_exr_import_error(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket"))
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes",
            side_effect=ImportError,
        ):
            with pytest.raises(BackendWriteError, match="install OpenEXR"):
                backend.write_array("img.exr", arr, encoding="exr")

    def test_key_with_no_prefix(self, mock_boto3):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix=""))
        backend.write("path.txt", b"x")
        call_kwargs = mock_boto3.put_object.call_args[1]
        assert call_kwargs["Key"] == "path.txt"

    @pytest.mark.parametrize(
        "path",
        [
            "../x",
            "a/../../x",
            "a/..",
            "/abs/x",
            "C:/outside/x",
            "\\\\server\\share\\x",
            "a//b",
            "",
            "./",
        ],
    )
    def test_paths_outside_prefix_are_rejected_without_requests(self, mock_boto3, path):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="data"))
        with pytest.raises(BackendWriteError) as write_error:
            backend.write(path, b"x")
        assert isinstance(write_error.value.__cause__, ValueError)
        with pytest.raises(BackendReadError):
            backend.read(path)
        assert backend.exists(path) is False
        mock_boto3.put_object.assert_not_called()
        mock_boto3.upload_fileobj.assert_not_called()
        mock_boto3.get_object.assert_not_called()
        mock_boto3.head_object.assert_not_called()

    @pytest.mark.parametrize("prefix", ["../other", "/abs", "a//b"])
    def test_list_prefix_outside_prefix_is_rejected(self, mock_boto3, prefix):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="data"))
        with pytest.raises(BackendReadError):
            backend.list(prefix)
        mock_boto3.get_paginator.assert_not_called()

    @pytest.mark.parametrize(
        ("path", "key"),
        [
            ("a/b/c.png", "data/a/b/c.png"),
            ("./a/./b.png", "data/a/b.png"),
            ("..hidden/x", "data/..hidden/x"),
        ],
    )
    def test_nested_paths_keep_their_key(self, mock_boto3, path, key):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="data/"))
        backend.write(path, b"x")
        assert mock_boto3.put_object.call_args.kwargs["Key"] == key

    @pytest.mark.parametrize("configured", ["data", "/data", "data/", "/data/"])
    def test_list_strips_the_normalized_prefix(self, mock_boto3, configured):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [{"Key": "data/a/b.png"}]}]
        mock_boto3.get_paginator.return_value = paginator
        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix=configured))
        assert backend.list() == ["a/b.png"]
        assert paginator.paginate.call_args.kwargs["Prefix"] == "data/"

    @pytest.mark.parametrize(
        ("prefix", "key_prefix"), [("", "data/"), ("ds-1/", "data/ds-1/")]
    )
    def test_list_keeps_root_and_trailing_slash(self, mock_boto3, prefix, key_prefix):
        from ovannotators.backends.s3 import S3Backend, S3BackendConfig

        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_boto3.get_paginator.return_value = paginator
        backend = S3Backend(S3BackendConfig(bucket="mybucket", prefix="data"))
        assert backend.list(prefix) == []
        assert paginator.paginate.call_args.kwargs["Prefix"] == key_prefix

    def test_config_region_and_endpoint(self):
        with patch("ovannotators.backends.s3._BOTO3_AVAILABLE", True):
            with patch("ovannotators.backends.s3.boto3") as mock_b3:
                mock_b3.client.return_value = MagicMock()
                mock_b3.s3.transfer.TransferConfig.return_value = MagicMock()
                from ovannotators.backends.s3 import S3Backend, S3BackendConfig

                S3Backend(
                    S3BackendConfig(
                        bucket="mybucket",
                        region="us-east-1",
                        endpoint_url="http://localhost:9000",
                    )
                )
                call_kwargs = mock_b3.client.call_args[1]
                assert call_kwargs["region_name"] == "us-east-1"
                assert call_kwargs["endpoint_url"] == "http://localhost:9000"


class TestBackendRegistry:
    def test_list_names_contains_builtin(self) -> None:
        assert "disk" in BackendRegistry.list_names()

    def test_get_disk(self) -> None:
        assert BackendRegistry.get("disk") is DiskBackend

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown backend"):
            BackendRegistry.get("nonexistent_backend_xyz")

    def test_register_decorator(self) -> None:
        @BackendRegistry.register
        class DummyBackend:
            name = "test_dummy_backend_9999"

            def write(self, path, data):
                pass

            def write_array(self, path, array, encoding="npy"):
                pass

            def read(self, path):
                return b""

            def list(self, prefix=""):
                return []

            def exists(self, path):
                return False

            def makedirs(self, path):
                pass

        try:
            assert BackendRegistry.get("test_dummy_backend_9999") is DummyBackend
        finally:
            BackendRegistry._registry.pop("test_dummy_backend_9999", None)

    def test_builtins_registered_under_public_name(self) -> None:
        from ovannotators.backends.ovstorage import OvstorageBackend
        from ovannotators.backends.s3 import S3Backend

        for backend_cls in (DiskBackend, S3Backend, OvstorageBackend):
            assert BackendRegistry.get(backend_cls.name) is backend_cls

    def test_register_without_name_raises(self) -> None:
        class NamelessBackend:
            pass

        with pytest.raises(ValueError, match="must define a non-empty"):
            BackendRegistry.register(NamelessBackend)

    def test_register_duplicate_name_raises(self) -> None:
        class OtherDisk:
            name = "disk"

        with pytest.raises(ValueError, match="already registered"):
            BackendRegistry.register(OtherDisk)

    def test_create_instantiates_backend(self, tmp_path) -> None:
        backend = BackendRegistry.create(
            "disk", DiskBackendConfig(root_dir=str(tmp_path))
        )
        assert isinstance(backend, DiskBackend)
