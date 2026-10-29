"""Phase 19: the optional HTTP API (`glypharis.server`) on a tiny bundle built from untrained
models, through a real socket on a free local port."""

from __future__ import annotations

import http.client
import io
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from det_fixtures import rect_polygon
from test_pipeline import _page, tiny_detector, tiny_recognizer

from glypharis import __version__
from glypharis.export.bundle import write_bundle
from glypharis.pipeline import GlypharisOCR
from glypharis.server import main, make_server

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "det_ck.pt")
    return write_bundle(root / "det_ck.pt", det.spec, rec, root / "tiny-0.0.1", "0.0.1-test")


@pytest.fixture(scope="module")
def ocr(bundle) -> GlypharisOCR:
    return GlypharisOCR.from_pretrained(bundle, cache=False)


@pytest.fixture(scope="module")
def page_png() -> bytes:
    buf = io.BytesIO()
    _page([rect_polygon(150, 80, 200, 20)]).save(buf, format="PNG")
    return buf.getvalue()


@contextmanager
def serving(ocr: Any, **kw: Any) -> Iterator[http.client.HTTPConnection]:
    server = make_server(ocr, "127.0.0.1", 0, **kw)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
    try:
        yield conn
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def call(
    conn: http.client.HTTPConnection, method: str, path: str, body: bytes | None = None
) -> tuple[int, dict[str, Any]]:
    conn.request(method, path, body=body)
    resp = conn.getresponse()
    assert resp.getheader("Content-Type", "").startswith("application/json")
    return resp.status, json.loads(resp.read().decode("utf-8"))


def test_health_reports_the_versions(ocr):
    with serving(ocr) as conn:
        status, body = call(conn, "GET", "/health")
    assert status == 200
    assert body == {"status": "ok", "model_version": "0.0.1-test", "glypharis_version": __version__}


def test_ocr_returns_the_documented_json_and_matches_predict(ocr, page_png, tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(page_png)
    expected = ocr.predict(path).to_dict()
    with serving(ocr) as conn:
        status, body = call(conn, "POST", "/ocr", page_png)
        # each answer closes the connection; http.client reconnects for the next request
        status2, body2 = call(conn, "POST", "/ocr", page_png)
    assert status == status2 == 200
    assert set(body) == set(expected)
    assert body["model_version"] == "0.0.1-test"
    assert (body["width"], body["height"]) == (400, 300)
    assert body["text"] == body2["text"] == expected["text"]
    assert [ln["text"] for ln in body["lines"]] == [ln["text"] for ln in expected["lines"]]


@pytest.mark.parametrize("data", [b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, b""])
def test_unreadable_bodies_are_400(ocr, data):
    with serving(ocr) as conn:
        status, body = call(conn, "POST", "/ocr", data)
    assert status == 400 and body["error"] == "invalid_image"


def test_size_limits_are_413(bundle, page_png):
    small = GlypharisOCR.from_pretrained(bundle, cache=False)
    with serving(small, max_bytes=len(page_png) - 1) as conn:
        status, body = call(conn, "POST", "/ocr", page_png)
    assert status == 413 and body["error"] == "request_too_large"
    small.max_pixels = 100 * 100  # the page is 400 x 300
    with serving(small) as conn:
        status, body = call(conn, "POST", "/ocr", page_png)
    assert status == 413 and body["error"] == "image_too_large"


def test_missing_length_unknown_paths_and_methods(ocr):
    with serving(ocr) as conn:
        conn.putrequest("POST", "/ocr")
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 411 and json.loads(resp.read())["error"] == "length_required"
    with serving(ocr) as conn:
        assert call(conn, "GET", "/nope")[0] == 404
        assert call(conn, "GET", "/ocr") == (
            405,
            {"error": "method_not_allowed", "message": "POST the image file to /ocr"},
        )
        assert call(conn, "PUT", "/ocr", b"x")[0] == 405
    with serving(ocr) as conn:
        assert call(conn, "POST", "/elsewhere", b"x")[0] == 404


def test_unexpected_errors_are_500_without_details(page_png, caplog):
    class Broken:
        model_version = "broken"

        def predict(self, image: bytes) -> None:
            raise RuntimeError("secret internals")

    with serving(Broken()) as conn:
        status, body = call(conn, "POST", "/ocr", page_png)
    assert status == 500 and body["error"] == "internal_error"
    assert "secret" not in json.dumps(body)
    assert "secret internals" in caplog.text


def test_main_refuses_a_missing_model(tmp_path, capsys):
    assert main(["--model", str(tmp_path / "nothing-here")]) == 2
    assert "error:" in capsys.readouterr().err


def test_an_idle_client_does_not_block_the_next_one(ocr, monkeypatch):
    import socket

    import glypharis.server as srv_mod

    monkeypatch.setattr(srv_mod, "SOCKET_TIMEOUT", 1.0)
    with serving(ocr) as conn:
        idle = socket.create_connection(("127.0.0.1", conn.port))  # connects, never sends
        try:
            status, body = call(conn, "GET", "/health")  # served once the idle one times out
        finally:
            idle.close()
    assert status == 200 and body["status"] == "ok"
    resp_headers = None
    with serving(ocr) as conn:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp_headers = dict(resp.getheaders())
        resp.read()
    assert resp_headers.get("Connection") == "close"
