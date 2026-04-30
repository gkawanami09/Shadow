import threading
from http import server
from socketserver import ThreadingMixIn

import cv2


PAGINA_HTML = b"""\
<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Shadow Stream</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #0f1418;
      color: #f4f7f8;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(96vw, 1120px);
      padding: 18px;
      border-radius: 14px;
      background: #18242d;
      border: 1px solid #304654;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.3rem;
    }
    p {
      margin: 0 0 14px;
      opacity: 0.86;
    }
    img {
      width: 100%;
      border-radius: 10px;
      background: #000;
      border: 1px solid #2b3f4c;
    }
    code {
      color: #74e0b3;
    }
  </style>
</head>
<body>
  <main>
    <h1>Shadow - Stream de Debug</h1>
    <p>Abra esta pagina no notebook para acompanhar a visao em tempo real.</p>
    <img src=\"/stream.mjpg\" alt=\"Stream da camera\">
    <p>Endpoint direto: <code>/stream.mjpg</code></p>
  </main>
</body>
</html>
"""


class _ServidorHTTPComThreads(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ServidorStream:
    def __init__(self, host="0.0.0.0", port=8080, qualidade_jpeg=80):
        self.host = host
        self.port = int(port)
        self.qualidade_jpeg = int(qualidade_jpeg)
        self._trava_quadro = threading.Lock()
        self._condicao_quadro = threading.Condition(self._trava_quadro)
        self._quadro_jpeg = None
        self._sequencia_quadro = 0
        self._servidor = None
        self._thread = None

    def atualizar_quadro(self, quadro_bgr):
        sucesso, codificado = cv2.imencode(
            ".jpg",
            quadro_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.qualidade_jpeg)],
        )
        if not sucesso:
            return

        with self._condicao_quadro:
            self._quadro_jpeg = codificado.tobytes()
            self._sequencia_quadro += 1
            self._condicao_quadro.notify_all()

    def _esperar_quadro(self, ultima_sequencia, timeout=1.0):
        with self._condicao_quadro:
            if self._sequencia_quadro == ultima_sequencia:
                self._condicao_quadro.wait(timeout=timeout)
            return self._quadro_jpeg, self._sequencia_quadro

    def iniciar(self):
        dono = self

        class Manipulador(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.end_headers()
                    self.wfile.write(PAGINA_HTML)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        ultima_sequencia = -1
                        while True:
                            quadro, sequencia = dono._esperar_quadro(ultima_sequencia, timeout=1.0)
                            if quadro is None or sequencia == ultima_sequencia:
                                continue
                            ultima_sequencia = sequencia
                            self.wfile.write(b"--frame\\r\\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(quadro)))
                            self.end_headers()
                            self.wfile.write(quadro)
                            self.wfile.write(b"\\r\\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return

                self.send_error(404)

            def log_message(self, formato, *argumentos):
                return

        self._servidor = _ServidorHTTPComThreads((self.host, self.port), Manipulador)
        self._thread = threading.Thread(target=self._servidor.serve_forever, daemon=True)
        self._thread.start()

    def parar(self):
        if self._servidor is not None:
            self._servidor.shutdown()
            self._servidor.server_close()
            self._servidor = None
        self._thread = None
