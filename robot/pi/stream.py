import threading
from http import server
from socketserver import ThreadingMixIn

import cv2


HTML_PAGE = b"""\
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shadow Vision Stream</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101820;
      --panel: #17232d;
      --accent: #6ee7b7;
      --text: #eef6f8;
    }
    body {
      margin: 0;
      font-family: sans-serif;
      background: linear-gradient(135deg, #0d1419, #13232f);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(96vw, 1100px);
      padding: 20px;
      background: rgba(23, 35, 45, 0.92);
      border: 1px solid rgba(110, 231, 183, 0.2);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.4rem;
    }
    p {
      margin: 0 0 16px;
      opacity: 0.82;
    }
    img {
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: #000;
    }
    code {
      color: var(--accent);
    }
  </style>
</head>
<body>
  <main>
    <h1>Shadow Vision Stream</h1>
    <p>Abra esta pagina no notebook para acompanhar a webcam e o debug em tempo real.</p>
    <img src="/stream.mjpg" alt="Stream da camera">
    <p>Endpoint direto: <code>/stream.mjpg</code></p>
  </main>
</body>
</html>
"""


class _ThreadedHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamServer:
    def __init__(self, host="0.0.0.0", port=8080, jpeg_quality=80):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._frame_lock = threading.Lock()
        self._frame_bytes = None
        self._server = None
        self._thread = None

    def update_frame(self, frame_bgr):
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            return
        with self._frame_lock:
            self._frame_bytes = encoded.tobytes()

    def get_frame(self):
        with self._frame_lock:
            return self._frame_bytes

    def start(self):
        owner = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.end_headers()
                    self.wfile.write(HTML_PAGE)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            frame = owner.get_frame()
                            if frame is None:
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame)))
                            self.end_headers()
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        self._server = _ThreadedHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None
