"""Optional HTTP API: ``glypharis-serve`` (Phase 19).

    glypharis-serve [--model glypharis-v1] [--no-download] [--host 127.0.0.1] [--port 8080]
    curl --data-binary @page.jpg http://127.0.0.1:8080/ocr

The model is loaded with `GlypharisOCR.from_pretrained` before the server starts. A registered name
(`glypharis.hub`) that no models folder holds is downloaded and verified first
(`glypharis.download_model`), unless ``--no-download`` is given or ``GLYPHARIS_OFFLINE`` is set; a
path or any other name is never downloaded, and nothing is downloaded once the server is running.

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

A request refused before its body was read gets its answer at once; the server then discards what
the client is still sending, up to ``DISCARD_MAX_BYTES`` and ``DISCARD_SECONDS``, so that closing
the connection does not destroy the answer. A client still sending past those limits may see the
connection reset before it reads the answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from glypharis import __version__
from glypharis.errors import GlypharisError, ImageTooLargeError

DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # a 600 dpi A4 scan as PNG is about 10 MB
SOCKET_TIMEOUT = 30.0  # seconds a client may stay silent mid-request
DISCARD_MAX_BYTES = 64 * 1024 * 1024  # of a refused request's body, read and thrown away
DISCARD_SECONDS = 5.0  # the longest a refused request may keep this one-at-a-time server busy
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

        def _refuse(self, status: int, code: str, message: str) -> None:
            """Answer a request whose body was not read, and close without losing the answer."""
            self._error(status, code, message)
            self._discard_body()

        def _discard_body(self) -> None:
            # Closing a socket with unread data resets the connection, and the reset can destroy
            # an answer the client has not read yet (seen on Windows). So finish our side first and
            # take what the client is still sending, within limits: this server has one thread.
            if "Content-Length" not in self.headers and "Transfer-Encoding" not in self.headers:
                return  # no body was announced
            try:
                left = min(int(self.headers["Content-Length"]), DISCARD_MAX_BYTES)
            except (TypeError, ValueError):  # chunked or unreadable: until the client stops
                left = DISCARD_MAX_BYTES
            deadline = time.monotonic() + DISCARD_SECONDS
            try:
                self.connection.shutdown(socket.SHUT_WR)
                while left > 0 and (remaining := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(remaining)
                    chunk = self.rfile.read1(min(left, 1 << 16))
                    if not chunk:
                        break
                    left -= len(chunk)
            except OSError:  # timed out or the client went away: nothing more to do
                pass

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
                self._refuse(405, "method_not_allowed", "POST the image file to /ocr")
            else:
                self._refuse(404, "not_found", f"no endpoint {self.path}; use POST /ocr")

        def do_POST(self) -> None:
            if self.path != "/ocr":
                self._refuse(404, "not_found", f"no endpoint {self.path}; use POST /ocr")
                return
            length = self.headers.get("Content-Length")
            if length is None or not length.strip().isdigit():
                self._refuse(411, "length_required", "send the image with a Content-Length header")
                return
            n = int(length)
            if n > max_bytes:
                self._refuse(
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
            self._refuse(405, "method_not_allowed", "POST the image file to /ocr")

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
    parser.add_argument("--model", default="glypharis-v1", help="bundle name or directory")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="never fetch a registered model that is not installed",
    )
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-pixels", type=int, default=None, help="default: the library's")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from glypharis.hub import can_fetch, download_model, needs_download
    from glypharis.pipeline.ocr import GlypharisOCR

    try:
        model: str | Path = args.model
        if not args.no_download and needs_download(args.model):
            if can_fetch(args.model):
                log.info("model %s is not installed; downloading it", args.model)
            model = download_model(args.model)
        ocr = GlypharisOCR.from_pretrained(model, backend=args.backend, threads=args.threads)
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
