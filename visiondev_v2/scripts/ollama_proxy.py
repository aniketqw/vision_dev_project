"""Local adapter: OpenAI /v1/chat/completions + /health  ->  Ollama native /api/chat.

Why: the VisionDev v3 pipeline speaks OpenAI-style to a vLLM server and polls /health.
The A6000's Ollama has no /health and its /v1 route ignores num_ctx, so qwen2.5vl:7b
fails to load at the default context. This proxy sets num_ctx per request.

  python ollama_proxy.py --listen 8000 --upstream http://localhost:11435 --num-ctx 8192
(upstream is an SSH tunnel: ssh -N -L 11435:localhost:11434 a6000)
Plain-text and image requests only; no streaming, no tool calls (the pipeline uses neither).
"""
import argparse
import json
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.*)$", re.S)


def to_ollama(messages):
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        texts, images = [], []
        for part in c:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                mt = DATA_URL.match(url)
                if not mt:
                    raise ValueError("only data: image URLs are supported")
                images.append(mt.group(1))
        msg = {"role": m["role"], "content": "\n".join(texts)}
        if images:
            msg["images"] = images
        out.append(msg)
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            with urllib.request.urlopen(ARGS.upstream + "/api/tags", timeout=5) as r:
                ok = r.status == 200
        except Exception as e:
            return self._send(503, {"status": "upstream down", "error": str(e)})
        if self.path.startswith("/health"):
            return self._send(200, {"status": "ok"})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": ARGS.model, "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if body.get("stream"):
            return self._send(400, {"error": "streaming not supported by proxy"})
        try:
            req = {"model": ARGS.model, "stream": False, "messages": to_ollama(body["messages"]),
                   "options": {"num_ctx": ARGS.num_ctx, "temperature": body.get("temperature", 0.1),
                               "num_predict": body.get("max_tokens", 1500)}}
            r = urllib.request.Request(ARGS.upstream + "/api/chat", json.dumps(req).encode(),
                                       {"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=ARGS.timeout) as resp:
                d = json.loads(resp.read())
        except Exception as e:
            return self._send(502, {"error": {"message": f"upstream failure: {e}", "type": "proxy_error"}})
        p, c = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        self._send(200, {
            "id": "chatcmpl-proxy", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", ARGS.model),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": d["message"]["content"]},
                         "finish_reason": "length" if d.get("done_reason") == "length" else "stop"}],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})

    def log_message(self, fmt, *a):
        print("proxy:", fmt % a, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8000)
    ap.add_argument("--upstream", default="http://localhost:11435")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=600)
    ARGS = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", ARGS.listen), H).serve_forever()
