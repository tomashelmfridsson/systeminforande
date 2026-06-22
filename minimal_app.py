import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
DEPLOY_REVISION_FILE = "deploy_revision.txt"


def load_deploy_revision() -> str:
    try:
        with open(DEPLOY_REVISION_FILE, encoding="utf-8") as revision_file:
            return revision_file.read().strip() or "unknown"
    except FileNotFoundError:
        return "local"


DEPLOY_REVISION = load_deploy_revision()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in {"/health", "/ready"}:
            self.write_json({"status": "ok", "revision": DEPLOY_REVISION})
            return

        if self.path == "/config":
            self.write_json(
                {
                    "app": "minimal-systeminforande-test",
                    "revision": DEPLOY_REVISION,
                    "purpose": "Hugging Face runtime isolation test",
                }
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            f"""<!doctype html>
<html lang="sv">
  <head>
    <meta charset="utf-8">
    <title>Minimal HF test</title>
  </head>
  <body>
    <h1>Minimal Hugging Face runtime test</h1>
    <p>Revision: <code data-deploy-revision="{DEPLOY_REVISION}">{DEPLOY_REVISION}</code></p>
    <p>Om du ser detta svarar den nya containern utan Gradio.</p>
  </body>
</html>
""".encode("utf-8")
        )

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}", flush=True)

    def write_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


print("HF_TOKEN present:", bool(os.environ.get("HF_TOKEN")), flush=True)
print("Deploy revision:", DEPLOY_REVISION, flush=True)
print(f"* Running on local URL:  http://{HOST}:{PORT}", flush=True)

server = ThreadingHTTPServer((HOST, PORT), Handler)
server.serve_forever()
