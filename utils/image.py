import base64
import logging
import os
from pathlib import Path
from typing import Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImageData(BaseModel):
    file_path: str = None
    base64_data: str = None
    uri_data: str = None
    binary_data: bytes = None
    mime_type: str = None


class ImageInterface:
    """Unified image class that handles conversion between all formats."""

    def __init__(
        self,
        file_path: str = None,
        base64_data: str = None,
        uri_data: str = None,
        binary_data: bytes = None,
    ):
        """
        Initialize Image from any format:
        - str: file path, data URI, or base64 string
        - bytes: binary image data
        - Path: file path
        """
        self.data = ImageData()
        if file_path:
            self._load_from_file(file_path)
        elif base64_data:
            self._load_from_base64(base64_data)
        elif uri_data:
            self._load_from_uri(uri_data)
        elif binary_data:
            self._load_from_binary(binary_data)
        else:
            raise ValueError("No image data provided")

    @property
    def file_path(self) -> str:
        return self.data.file_path

    @property
    def base64_data(self) -> str:
        if self.data.base64_data:
            return self.data.base64_data
        else:
            base64_data = base64.b64encode(self.data.binary_data).decode("utf-8")
            self.data.base64_data = base64_data
            return base64_data

    @property
    def binary_data(self) -> bytes:
        return self.data.binary_data

    @property
    def mime_type(self) -> str:
        if self.data.mime_type:
            return self.data.mime_type
        else:
            mime_type = self._detect_mime_type()
            self.data.mime_type = mime_type
            return mime_type

    @property
    def uri_data(self) -> str:
        if self.data.uri_data:
            return self.data.uri_data
        else:
            uri_data = f"data:{self.mime_type};base64,{self.base64_data}"
            self.data.uri_data = uri_data
            return uri_data

    @property
    def extension(self) -> str:
        return self.mime_type.split("/")[1]

    def _load_from_file(self, file_path: str):
        """Load from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        self.data.file_path = file_path
        self.data.binary_data = open(file_path, "rb").read()

    def _load_from_uri(self, data_uri: str):
        """Load from data URI (data:image/png;base64,...)."""
        if not data_uri.startswith("data:"):
            raise ValueError("Invalid data URI format")

        self.data.uri_data = data_uri

        # Extract mime type and base64 data
        header, base64_data = data_uri.split(",", 1)
        mime_part = header.split(":")[1].split(";")[0]

        self.data.mime_type = mime_part
        self.data.binary_data = base64.b64decode(base64_data)

    def _load_from_base64(self, base64_data: str):
        """Load from base64 string."""
        self.data.base64_data = base64_data
        self.data.binary_data = base64.b64decode(base64_data)
        self.data.mime_type = self._detect_mime_type()

    def _load_from_binary(self, binary_data: bytes):
        """Load from binary data."""
        self.data.binary_data = binary_data
        self.data.mime_type = self._detect_mime_type()

    def _detect_mime_type(self) -> str:
        """Detect MIME type from binary data."""
        data = self.data.binary_data
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        elif data.startswith(b"BM"):
            return "image/bmp"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            return "image/webp"
        else:
            logger.warning("Unknown image format, defaulting to PNG")
            return "image/png"

    def save(self, file_path: Union[str, Path]):
        """Save to file."""
        if not file_path.endswith(f".{self.extension}"):
            logger.warning(f"File extension mismatch: {file_path} != {self.extension}")
        self.data.file_path = file_path

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(self.data.binary_data)

    def __repr__(self):
        return f"Image(mime_type='{self.data.mime_type}', size={len(self.data.binary_data)} bytes)"
