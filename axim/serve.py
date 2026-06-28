"""OpenAI-compatible API server for .axim models — port of scripts/api_server.py.

Exposes:
  - build_app(model, tok, cfg, device) -> FastAPI: the 5 OpenAI-compatible routes
    with verbose per-request logging and SSE streaming, verbatim from
    scripts/api_server.py.
  - serve(axim_path, host, port, device): load a model via axim.model.load_model,
    wrap its tokenizer in axim.tokenizer.Tokenizer, build the app, run uvicorn.
    This is the entry point the `axim serve` CLI (Task 7) calls.
"""

import json
import time
import socket

import torch

from .model import load_model
from .tokenizer import Tokenizer
from .generate import sample


# ---------- verbose logging ----------
# Per-request summary logs go to the server console (stdout), flushed immediately
# so you can watch what the API is doing while driving it from the webUI. One block
# per request: endpoint, sampling params, prompt token count, then completion
# length / elapsed / tok-per-sec / stop reason when it finishes.

_REQ_COUNTER = {"n": 0}

def _next_req_id():
    _REQ_COUNTER["n"] += 1
    return _REQ_COUNTER["n"]

def _log(rid, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] #{rid:>4} {msg}", flush=True)

def _fmt_params(temp, top_k, max_tok, rep_pen, stop, stream, greedy):
    parts = [f"stream={'on' if stream else 'off'}",
            f"max_tokens={max_tok}",
            f"temp={temp}",
            f"top_k={top_k}",
            f"rep_pen={rep_pen}"]
    if stop:
        parts.append(f"stop={stop}")
    if greedy:
        parts.append("greedy")
    return "  ".join(parts)


def build_app(model, tok, cfg, device):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uuid
    import time

    app = FastAPI(title="axim api", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{
            "id": cfg.get("model_type", "axim"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "axim",
        }]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        rid = _next_req_id()
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tok = body.get("max_tokens", 256)
        temp = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        greedy = temp == 0
        rep_pen = body.get("repetition_penalty", 1.0)
        top_k_req = body.get("top_k", None)
        stop = body.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]

        prompt_ids = tok.encode(prompt)
        if len(prompt_ids) == 0:
            prompt_ids = [tok.bos]
        else:
            # Prepend BOS if not already present
            if prompt_ids[0] != tok.bos:
                prompt_ids = [tok.bos] + prompt_ids

        k = (None if greedy else 50) if top_k_req is None else top_k_req
        t = 0.0 if greedy else temp

        _log(rid, f"→ POST /v1/completions  prompt={len(prompt_ids)}tok  "
                  f"{_fmt_params(t, k, max_tok, rep_pen, stop, stream, greedy)}")

        if stream:
            async def _stream():
                status = {}
                t0 = time.perf_counter()
                rid_s = f"cmpl-{uuid.uuid4().hex}"
                cr = int(time.time())
                yield f"data: {json.dumps({'id': rid_s, 'object': 'text_completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'text': '', 'finish_reason': None}]})}\n\n"

                count = 0
                for token in sample(model, tok, prompt_ids, max_tok, t, k, device,
                                   repetition_penalty=rep_pen, stop_strings=stop, status=status):
                    count += 1
                    txt = tok.decode([token])
                    yield f"data: {json.dumps({'id': rid_s, 'object': 'text_completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'text': txt, 'finish_reason': None}]})}\n\n"

                elapsed = time.perf_counter() - t0
                tps = count / elapsed if elapsed > 0 else 0.0
                _log(rid, f"← done  {count}tok in {elapsed:.2f}s ({tps:.1f} tok/s)  "
                          f"finish={status.get('reason','stop')} ({status.get('cause','')})")
                yield f"data: {json.dumps({'id': rid_s, 'object': 'text_completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'text': '', 'finish_reason': status.get('reason', 'stop')}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        else:
            status = {}
            t0 = time.perf_counter()
            toks = []
            for token in sample(model, tok, prompt_ids, max_tok, t, k, device,
                                repetition_penalty=rep_pen, stop_strings=stop, status=status):
                toks.append(token)
            elapsed = time.perf_counter() - t0
            tps = len(toks) / elapsed if elapsed > 0 else 0.0
            _log(rid, f"← done  {len(toks)}tok in {elapsed:.2f}s ({tps:.1f} tok/s)  "
                      f"finish={status.get('reason','stop')} ({status.get('cause','')})")

            return JSONResponse({
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": cfg.get("model_type", "axim"),
                "choices": [{"index": 0, "text": tok.decode(toks), "finish_reason": status.get("reason", "stop")}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(toks), "total_tokens": len(prompt_ids) + len(toks)},
            })

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        rid = _next_req_id()
        body = await request.json()
        msgs = body.get("messages", [])
        max_tok = body.get("max_tokens", 256)
        temp = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        greedy = temp == 0
        rep_pen = body.get("repetition_penalty", 1.0)
        top_k_req = body.get("top_k", None)
        stop = body.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]

        prompt = tok.render_chat(msgs, max_len=cfg["max_position_embeddings"])
        prompt.append(tok.encode_special("<|assistant_start|>"))

        k = (None if greedy else 50) if top_k_req is None else top_k_req
        t = 0.0 if greedy else temp

        _log(rid, f"→ POST /v1/chat/completions  prompt={len(prompt)}tok  msgs={len(msgs)}  "
                  f"{_fmt_params(t, k, max_tok, rep_pen, stop, stream, greedy)}")

        if stream:
            async def _stream():
                status = {}
                t0 = time.perf_counter()
                rid_s = f"chatcmpl-{uuid.uuid4().hex}"
                cr = int(time.time())
                yield f"data: {json.dumps({'id': rid_s, 'object': 'chat.completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

                count = 0
                for token in sample(model, tok, prompt, max_tok, t, k, device,
                                   repetition_penalty=rep_pen, stop_strings=stop, status=status):
                    count += 1
                    txt = tok.decode([token])
                    yield f"data: {json.dumps({'id': rid_s, 'object': 'chat.completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'delta': {'content': txt}, 'finish_reason': None}]})}\n\n"

                elapsed = time.perf_counter() - t0
                tps = count / elapsed if elapsed > 0 else 0.0
                _log(rid, f"← done  {count}tok in {elapsed:.2f}s ({tps:.1f} tok/s)  "
                          f"finish={status.get('reason','stop')} ({status.get('cause','')})")
                yield f"data: {json.dumps({'id': rid_s, 'object': 'chat.completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': status.get('reason', 'stop')}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        else:
            status = {}
            t0 = time.perf_counter()
            toks = []
            for token in sample(model, tok, prompt, max_tok, t, k, device,
                                repetition_penalty=rep_pen, stop_strings=stop, status=status):
                toks.append(token)
            elapsed = time.perf_counter() - t0
            tps = len(toks) / elapsed if elapsed > 0 else 0.0
            _log(rid, f"← done  {len(toks)}tok in {elapsed:.2f}s ({tps:.1f} tok/s)  "
                      f"finish={status.get('reason','stop')} ({status.get('cause','')})")

            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": cfg.get("model_type", "axim"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(toks)}, "finish_reason": status.get("reason", "stop")}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(toks), "total_tokens": len(prompt) + len(toks)},
            })

    @app.post("/v1/debug/template")
    async def debug_template(request: Request):
        """Render the chat template for a set of messages WITHOUT running the model.
        Returns the token-id list, decoded text, and count — lets the webUI show
        devs exactly what tokens the chat template produces. No forward pass."""
        rid = _next_req_id()
        body = await request.json()
        msgs = body.get("messages", [])
        ids = tok.render_chat(msgs, max_len=cfg["max_position_embeddings"])
        ids = ids + [tok.encode_special("<|assistant_start|>")]
        _log(rid, f"→ POST /v1/debug/template  rendered {len(ids)}tok from {len(msgs)} msgs (no forward pass)")
        return JSONResponse({
            "token_ids": ids,
            "decoded": tok.decode(ids),
            "count": len(ids),
        })

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _find_free_port(start=8000, end=9000):
    """Find an available TCP port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError(f"no free port found in range {start}-{end}")


def serve(axim_path, host="0.0.0.0", port=None, device="cpu"):
    import uvicorn
    dev = torch.device(device)
    model, enc, cfg = load_model(axim_path, dev)
    tok = Tokenizer(enc)

    total = sum(p.numel() for p in model.parameters())
    print(f"loaded {total:,} params ({total/1e9:.2f}B)")
    print(f"device: {dev}")
    if port is None:
        port = _find_free_port()

    app = build_app(model, tok, cfg, dev)
    print(f"server: http://{host}:{port}")
    print("endpoints:")
    print("  POST /v1/chat/completions   (OpenAI-style chat, stream=true for SSE)")
    print("  POST /v1/completions        (raw prompt completion, stream=true for SSE)")
    print("  POST /v1/debug/template     (render chat template, no forward pass)")
    print("  GET  /v1/models  /health")
    print("verbose per-request logs below (one line per request ->/<-):")
    uvicorn.run(app, host=host, port=port)