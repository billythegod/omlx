# SPDX-License-Identifier: Apache-2.0
"""Tests for the MarkItDown integration."""

from __future__ import annotations

import base64
import sys
import types
import warnings
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.api.markitdown import (
    MARKITDOWN_EMPTY_PDF_MESSAGE,
    MARKITDOWN_MODEL_ID,
    MarkItDownFile,
    MarkItDownRequestError,
    convert_file_to_markdown,
    parse_file_part,
    preprocess_markitdown_file_parts,
)
from omlx.api.openai_models import Message
from omlx.server import ServerState, app
from omlx.settings import GlobalSettings


def _data_uri(payload: bytes = b"doc", mime_type: str = "application/pdf") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _file_part(
    filename: str = "sample.pdf",
    data: str | None = None,
    mime_type: str = "application/pdf",
) -> dict:
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "mime_type": mime_type,
            "file_data": data or _data_uri(mime_type=mime_type),
        },
    }


class _EmptyPool:
    def get_status(self) -> dict:
        return {
            "final_ceiling": 0,
            "current_model_memory": 0,
            "model_count": 0,
            "loaded_count": 0,
            "models": [],
        }


def test_openai_models_includes_markitdown_when_enabled():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "MarkItDown" in ids


def test_openai_models_hides_markitdown_when_disabled():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_enabled = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert MARKITDOWN_MODEL_ID not in ids


def test_openai_models_hides_markitdown_when_not_exposed():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_expose_model = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert MARKITDOWN_MODEL_ID not in ids


def test_markitdown_chat_completion_converts_file(monkeypatch):
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()

    def fake_convert(file: MarkItDownFile) -> str:
        assert file.filename == "sample.pdf"
        return "# Converted"

    monkeypatch.setattr(
        "omlx.api.markitdown.convert_file_to_markdown", fake_convert
    )

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [{"role": "user", "content": [_file_part()]}],
            },
        )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "## Attached file: sample.pdf" in content
    assert "# Converted" in content


def test_markitdown_chat_completion_uses_latest_user_turn(monkeypatch):
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()

    def fake_convert(file: MarkItDownFile) -> str:
        return f"# Converted {file.filename}"

    monkeypatch.setattr(
        "omlx.api.markitdown.convert_file_to_markdown", fake_convert
    )

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [
                    {"role": "user", "content": [_file_part("file1.pdf")]},
                    {
                        "role": "assistant",
                        "content": (
                            "## Attached file: file1.pdf\n\n# Converted file1.pdf"
                        ),
                    },
                    {"role": "user", "content": [_file_part("file2.pdf")]},
                ],
            },
        )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "file2.pdf" in content
    assert "file1.pdf" not in content


def test_markitdown_chat_completion_disabled_returns_404():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_enabled = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 404
    assert "markitdown" in response.json()["error"]["message"].lower()


def test_markitdown_chat_completion_hidden_model_returns_404():
    state = ServerState()
    state.engine_pool = _EmptyPool()
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_expose_model = False

    with patch("omlx.server._server_state", state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MARKITDOWN_MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 404
    assert "markitdown" in response.json()["error"]["message"].lower()


def test_preprocess_file_parts_for_llm(monkeypatch):
    def fake_convert(file: MarkItDownFile) -> str:
        return "Converted text"

    monkeypatch.setattr(
        "omlx.api.markitdown.convert_file_to_markdown", fake_convert
    )
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "Summarize this."},
                _file_part("paper.pdf"),
            ],
        )
    ]

    processed = preprocess_markitdown_file_parts(
        messages,
        global_settings=GlobalSettings(),
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    assert parts[0].type == "text"
    assert parts[1].type == "text"
    assert "## Attached file: paper.pdf" in (parts[1].text or "")
    assert "Converted text" in (parts[1].text or "")


def test_preprocess_file_parts_works_when_model_not_exposed(monkeypatch):
    def fake_convert(file: MarkItDownFile) -> str:
        return "Converted text"

    monkeypatch.setattr(
        "omlx.api.markitdown.convert_file_to_markdown", fake_convert
    )
    settings = GlobalSettings()
    settings.integrations.markitdown_expose_model = False

    processed = preprocess_markitdown_file_parts(
        [Message(role="user", content=[_file_part("paper.pdf")])],
        global_settings=settings,
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    assert parts[0].type == "text"
    assert "Converted text" in (parts[0].text or "")


def test_text_and_markdown_file_parts_are_inlined_without_converter(monkeypatch):
    called = False

    def fake_convert(file: MarkItDownFile) -> str:
        nonlocal called
        called = True
        raise AssertionError("plain text attachments should not use MarkItDown")

    monkeypatch.setattr(
        "omlx.api.markitdown.convert_file_to_markdown", fake_convert
    )

    processed = preprocess_markitdown_file_parts(
        [
            Message(
                role="user",
                content=[
                    _file_part(
                        "notes.txt",
                        data=_data_uri(b"Plain notes", mime_type="text/plain"),
                        mime_type="text/plain",
                    ),
                    _file_part(
                        "guide.md",
                        data=_data_uri(b"# Guide", mime_type="text/markdown"),
                        mime_type="text/markdown",
                    ),
                ],
            )
        ],
        global_settings=GlobalSettings(),
    )

    parts = processed[0].content
    assert isinstance(parts, list)
    rendered = "\n".join(part.text or "" for part in parts)
    assert "## Attached file: notes.txt" in rendered
    assert "Plain notes" in rendered
    assert "## Attached file: guide.md" in rendered
    assert "# Guide" in rendered
    assert called is False


def test_preprocess_file_parts_does_not_create_mixed_content_warning(monkeypatch):
    def fake_convert(file: MarkItDownFile) -> str:
        return "Converted text"

    monkeypatch.setattr(
        "omlx.api.markitdown.convert_file_to_markdown", fake_convert
    )
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "Summarize this."},
                _file_part("paper.pdf"),
            ],
        )
    ]

    processed = preprocess_markitdown_file_parts(
        messages,
        global_settings=GlobalSettings(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        processed[0].model_dump()

    assert not [
        warning
        for warning in caught
        if "Pydantic serializer warnings" in str(warning.message)
    ]


def test_preprocess_file_parts_rejects_when_disabled():
    settings = GlobalSettings()
    settings.integrations.markitdown_enabled = False

    with pytest.raises(MarkItDownRequestError, match="disabled"):
        preprocess_markitdown_file_parts(
            [Message(role="user", content=[_file_part()])],
            global_settings=settings,
        )


def test_xlsx_is_rejected_without_pandas_dependency():
    part = {
        "type": "file",
        "file": {
            "filename": "sheet.xlsx",
            "mime_type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            "file_data": _data_uri(),
        },
    }

    with pytest.raises(MarkItDownRequestError, match="Spreadsheet"):
        parse_file_part(part, max_file_size_mb=25)


def test_openai_file_data_without_filename_infers_from_data_uri():
    parsed = parse_file_part(
        {
            "type": "file",
            "file": {
                "file_data": _data_uri(
                    b"Plain notes",
                    mime_type="text/plain",
                ),
            },
        },
        max_file_size_mb=25,
    )

    assert parsed.filename == "attachment.txt"
    assert parsed.mime_type == "text/plain"
    assert parsed.data == b"Plain notes"


def test_file_id_is_rejected():
    with pytest.raises(MarkItDownRequestError, match="file_id"):
        parse_file_part(
            {"type": "file", "file": {"file_id": "file_123", "filename": "x.pdf"}},
            max_file_size_mb=25,
        )


def test_empty_pdf_conversion_logs_warning(monkeypatch, caplog):
    @dataclass(frozen=True)
    class FakeStreamInfo:
        extension: str | None = None
        mimetype: str | None = None
        filename: str | None = None

    class FakeResult:
        markdown = ""

    class FakeConverter:
        def convert_stream(self, stream, stream_info=None):
            return FakeResult()

    fake_markitdown = types.ModuleType("markitdown")
    fake_markitdown.StreamInfo = FakeStreamInfo
    monkeypatch.setitem(sys.modules, "markitdown", fake_markitdown)
    monkeypatch.setattr("omlx.api.markitdown._converter", FakeConverter())

    caplog.set_level("WARNING")
    with pytest.raises(MarkItDownRequestError) as exc_info:
        convert_file_to_markdown(
            MarkItDownFile(
                filename="scan.pdf",
                mime_type="application/pdf",
                data=b"%PDF",
            )
        )

    assert exc_info.value.detail == MARKITDOWN_EMPTY_PDF_MESSAGE
    assert "no extractable text" in caplog.text.lower()
