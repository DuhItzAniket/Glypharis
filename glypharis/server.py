"""Optional HTTP API: ``glypharis-serve`` (Phase 19).

    glypharis-serve [--model glypharis-0.1.0] [--host 127.0.0.1] [--port 8080]
    curl --data-binary @page.jpg http://127.0.0.1:8080/ocr

An integration boundary, not a production web service: one process, one request at a time (the
model is loaded once and is not shared between threads), the standard library's HTTP server, no
authentication and no TLS. Each connection carries one request and is closed after the answer,
and a client silent for 30 seconds is dropped, so one idle or slow client cannot hold the server.
It listens on 127.0.0.1 unless told otherwise; put it behind a proper server or proxy before
exposing it to a network.

Endpoints:

- ``POST /ocr``: the request body is the image file itself (JPEG, PNG, TIFF, BMP or WebP; any
  content type). The answer is `OCRResult.to_dict` as JSON, which carries ``model_version`` and
  ``glypharis_version``.
- ``GET /health``: ``{"status": "ok", "model_version": ..., "glypharis_version": ...}``.

Errors are JSON ``{"error": <code>, "message": ...}`` (`glypharis.errors`): 400 ``invalid_image``
for a body that is not a readable image, 411 ``length_required`` without a Content-Length, 413
``request_too_large`` for a body above ``--max-bytes`` and ``image_too_large`` for more pixels than
``--max-pixels``, 404 and 405 for unknown paths and methods, and 500 ``internal_error`` (logged
with its traceback, never sent) for anything else.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from glypharis import __version__
from glypharis.errors import GlypharisError, ImageTooLargeError

DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # a 600 dpi A4 scan as PNG is about 10 MB
SOCKET_TIMEOUT = 30.0  # seconds a client may stay silent mid-request
log = logging.getLogger(__name__)


def make_handler(ocr: Any, max_bytes: int = DEFAULT_MAX_BYTES) -> type[BaseHTTPRequestHandler]:
    """A request handler class bound to one loaded `GlypharisOCR`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"glypharis/{__version__}"
        protocol_version = "HTTP/1.1"
        # One request per connection, and a client that stalls is dropped: the server handles one
        # connection at a time, so an idle keep-alive or a slow upload would block everyone else.
        timeout = SOCKET_TIMEOUT

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": code, "message": message})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "model_version": ocr.model_version,
                        "glypharis_version": __version__,
                    },
                )
            elif self.path == "/ocr":
                self._error(405, "method_not_allowed", "POST the image file to /ocr")
            else:
                self._error(404, "not_found", f"no endpoint {self.path}; use POST /ocr")

        def do_POST(self) -> None:
            if self.path != "/ocr":
                self._error(404, "not_found", f"no endpoint {self.path}; use POST /ocr")
                return
            length = self.headers.get("Content-Length")
            if length is None or not length.strip().isdigit():
                self._error(411, "length_required", "send the image with a Content-Length header")
                return
            n = int(length)
            if n > max_bytes:
                self._error(
                    413, "request_too_large", f"the body has {n} bytes; the limit is {max_bytes}"
                )
                return
            body = self.rfile.read(n)
            try:
                result = ocr.predict(body)
            except ImageTooLargeError as exc:
                self._send(413, exc.to_dict())
            except GlypharisError as exc:
                self._send(400, exc.to_dict())
            except Exception:
                log.exception("unexpected error on POST /ocr")
                self._error(500, "internal_error", "the server failed to read this image")
            else:
                self._send(200, result.to_dict())

        def do_PUT(self) -> None:
            self._error(405, "method_not_allowed", "POST the image file to /ocr")

        do_DELETE = do_PATCH = do_PUT

        def log_message(self, format: str, *args: Any) -> None:
            log.info("%s %s", self.address_string(), format % args)

    return Handler


def make_server(
    ocr: Any, host: str = "127.0.0.1", port: int = 8080, max_bytes: int = DEFAULT_MAX_BYTES
) -> HTTPServer:
    """A single-threaded server for ``ocr``; ``port=0`` picks a free port (see ``server_port``)."""
    return HTTPServer((host, port), make_handler(ocr, max_bytes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GLYPHARIS HTTP API (one process).")
    parser.add_argument("--model", default="glypharis-0.1.0", help="bundle name or directory")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-pixels", type=int, default=None, help="default: the library's")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from glypharis.pipeline.ocr import GlypharisOCR

    try:
        ocr = GlypharisOCR.from_pretrained(args.model, backend=args.backend, threads=args.threads)
    except GlypharisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.max_pixels is not None:
        ocr.max_pixels = args.max_pixels
    server = make_server(ocr, args.host, args.port, args.max_bytes)
    log.info(
        "GLYPHARIS %s (model %s) on http://%s:%d/ocr",
        __version__,
        ocr.model_version,
        args.host,
        server.server_port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
