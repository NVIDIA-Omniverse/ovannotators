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

"""Tests for all backend implementations: disk, s3, ovstorage."""

from __future__ import annotations

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
        # Trigger OSError in the try block by using a path with invalid chars
        # or by mocking
        with patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")):
            with pytest.raises(BackendWriteError):
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
            backend.write_array("depth.exr", arr, encoding="exr")
        mock_encode.assert_called_once()
        assert (tmp_path / "depth.exr").read_bytes() == b"fake_exr"

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
        ):
            backend.write_array("img.exr", arr, encoding="exr")
        call_kwargs = mock_boto3.put_object.call_args[1]
        assert call_kwargs["Body"] == b"exr"

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


# ===========================================================================
# OvstorageBackend (mocked httpx)
# ===========================================================================


class TestOvstorageBackend:
    @pytest.fixture
    def mock_httpx(self):
        with patch("ovannotators.backends.ovstorage._HTTPX_AVAILABLE", True):
            with patch("ovannotators.backends.ovstorage.httpx") as mock_h:
                mock_client = MagicMock()
                mock_h.Client.return_value = mock_client
                yield mock_client

    def test_import_error_when_httpx_unavailable(self):
        with patch("ovannotators.backends.ovstorage._HTTPX_AVAILABLE", False):
            from ovannotators.backends.ovstorage import (
                OvstorageBackend,
                OvstorageBackendConfig,
            )

            with pytest.raises(ImportError, match="httpx"):
                OvstorageBackend(OvstorageBackendConfig(base_url="http://x"))

    def test_write(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.put.return_value = mock_response

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local", prefix="v1")
        )
        backend.write("file.bin", b"content")
        mock_httpx.put.assert_called_once_with("/v1/file.bin", content=b"content")

    def test_write_error_raises(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_httpx.put.side_effect = Exception("timeout")
        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        with pytest.raises(BackendWriteError):
            backend.write("x.bin", b"data")

    def test_read(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_response = MagicMock()
        mock_response.content = b"file_data"
        mock_response.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_response

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        assert backend.read("myfile.bin") == b"file_data"

    def test_read_error_raises(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_httpx.get.side_effect = Exception("404")
        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        with pytest.raises(BackendReadError):
            backend.read("missing.bin")

    def test_list(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = ["a.txt", "b.txt"]
        mock_response.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_response

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        result = backend.list("prefix")
        assert result == ["a.txt", "b.txt"]

    def test_list_error_raises(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_httpx.get.side_effect = Exception("error")
        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        with pytest.raises(BackendReadError):
            backend.list("")

    def test_exists_true(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.head.return_value = mock_response

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        assert backend.exists("file.txt") is True

    def test_exists_false(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_httpx.head.return_value = mock_response

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        assert backend.exists("missing.txt") is False

    def test_exists_exception_returns_false(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_httpx.head.side_effect = Exception("timeout")
        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        assert backend.exists("x.txt") is False

    def test_makedirs_noop(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        backend.makedirs("path")  # should not raise

    def test_write_array_npy(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.put.return_value = mock_response

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        backend.write_array("data.npy", np.array([1, 2, 3]), encoding="npy")
        mock_httpx.put.assert_called_once()

    def test_write_array_unsupported_encoding(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        with pytest.raises(BackendWriteError, match="Unsupported encoding"):
            backend.write_array("x.bin", np.zeros(10), encoding="abc")

    def test_write_array_jpg_uses_shared_codec(self, mock_httpx):
        """jpg encoding routes the array through the shared codec and PUTs bytes."""
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        codec = MagicMock()
        codec.encode.return_value = b"jpg-bytes"
        with patch("ovannotators._internal.codecs.get_image_codec", return_value=codec):
            backend.write_array("img.jpg", arr, encoding="jpg")

        codec.encode.assert_called_once()
        assert codec.encode.call_args.kwargs == {"format": "jpg"}
        # ...and the resulting bytes are uploaded via HTTP PUT.
        mock_httpx.put.assert_called_once()
        assert mock_httpx.put.call_args[0][0].endswith("img.jpg")
        assert mock_httpx.put.call_args.kwargs["content"] == b"jpg-bytes"

    def test_write_array_exr(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes", return_value=b"exr"
        ):
            backend.write_array("img.exr", arr, encoding="exr")
        put_kwargs = mock_httpx.put.call_args[1]
        assert put_kwargs["content"] == b"exr"

    def test_write_array_exr_import_error(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        backend = OvstorageBackend(
            OvstorageBackendConfig(base_url="http://store.local")
        )
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        with patch(
            "ovannotators._internal.codecs._encode_exr_bytes",
            side_effect=ImportError,
        ):
            with pytest.raises(BackendWriteError, match="install OpenEXR"):
                backend.write_array("img.exr", arr, encoding="exr")

    def test_api_key_from_env(self, mock_httpx):
        from ovannotators.backends.ovstorage import (
            OvstorageBackend,
            OvstorageBackendConfig,
        )

        with patch.dict("os.environ", {"STORAGE_API_KEY": "secret123"}):
            import ovannotators.backends.ovstorage as mod

            with patch.object(mod, "_HTTPX_AVAILABLE", True):
                with patch.object(mod, "httpx") as mock_h:
                    mock_h.Client.return_value = MagicMock()
                    OvstorageBackend(
                        OvstorageBackendConfig(base_url="http://x", api_key=None)
                    )
                    call_kwargs = mock_h.Client.call_args[1]
                    assert "Bearer secret123" in str(call_kwargs.get("headers", {}))


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
