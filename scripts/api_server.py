"""
OpenAI-compatible API server for .axim models.

Usage:
    python api_server.py --axim model.axim --device cuda
    python api_server.py --axim model.axim --port 8080
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from axim.core import load_axim


def _load_model(axim_path, device):
    print(f"loading {axim_path}")
    data = load_axim(axim_path)
    
    # need nanochat for model class
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))
    from nanochat.gpt import GPT, GPTConfig
    
    cfg = data["config"]
    config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=cfg["max_position_embeddings"],
        window_pattern=cfg["window_pattern"],
    )
    
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(data["weights"], strict=True, assign=True)
    model = model.to(device)
    model.eval()
    
    return model, data["tokenizer"], cfg


class _TokenizerWrap:
    def __init__(self, enc):
        self.enc = enc
        self.bos = enc.encode_single_token("<|bos|>")
    
    def encode_special(self, s):
        return self.enc.encode_single_token(s)
    
    def encode(self, text):
        return self.enc.encode_ordinary(text)
    
    def decode(self, ids):
        return self.enc.decode(ids)
    
    def render_chat(self, messages, max_len=2048):
        import copy
        ids = []
        msgs = copy.deepcopy(messages)
        
        if msgs[0]["role"] == "system":
            msgs[1]["content"] = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
            msgs = msgs[1:]
        
        us = self.encode_special("<|user_start|>")
        ue = self.encode_special("<|user_end|>")
        ast = self.encode_special("<|assistant_start|>")
        aen = self.encode_special("<|assistant_end|>")
        
        ids.append(self.bos)
        for i, msg in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == want
            if msg["role"] == "user":
                ids += [us] + self.encode(msg["content"]) + [ue]
            else:
                ids += [ast] + self.encode(msg["content"]) + [aen]
        
        return ids[:max_len]


@torch.inference_mode()
def _generate(model, tok, prompt_ids, max_tok, temp, top_k, device):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    for _ in range(max_tok):
        logits = model(ids)
        logits = logits[:, -1, :]
        
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        if temp <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits / temp, dim=-1), 1)
        
        ids = torch.cat((ids, nxt), dim=1)
        t = nxt.item()
        
        if t == tok.encode_special("<|assistant_end|>") or t == tok.bos:
            break
        if ids.size(1) >= model.config.sequence_len:
            break
        
        yield t


def _make_app(model, tok, cfg, device):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    import uuid
    import time
    
    app = FastAPI(title="axim api", version="1.0")
    
    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{
            "id": cfg.get("model_type", "axim"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "axim",
        }]}
    
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        msgs = body.get("messages", [])
        max_tok = body.get("max_tokens", 256)
        temp = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        greedy = temp == 0
        
        prompt = tok.render_chat(msgs, max_len=cfg["max_position_embeddings"])
        prompt.append(tok.encode_special("<|assistant_start|>"))
        
        if stream:
            async def _stream():
                rid = f"chatcmpl-{uuid.uuid4().hex}"
                cr = int(time.time())
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                
                t = 0.0 if greedy else temp
                k = None if greedy else 50
                for token in _generate(model, tok, prompt, max_tok, t, k, device):
                    txt = tok.decode([token])
                    yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'delta': {'content': txt}, 'finish_reason': None}]})}\n\n"
                
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': cr, 'model': cfg.get('model_type', 'axim'), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(_stream(), media_type="text/event-stream")
        
        else:
            toks = []
            t = 0.0 if greedy else temp
            k = None if greedy else 50
            for token in _generate(model, tok, prompt, max_tok, t, k, device):
                toks.append(token)
            
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": cfg.get("model_type", "axim"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(toks)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(toks), "total_tokens": len(prompt) + len(toks)},
            })
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axim", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model, tokenizer, cfg = _load_model(args.axim, device)
    tok = _TokenizerWrap(tokenizer)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"loaded {total:,} params ({total/1e9:.2f}B)")
    print(f"device: {device}")
    
    app = _make_app(model, tok, cfg, device)
    
    import uvicorn
    print(f"server: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
