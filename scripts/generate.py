"""
Generate text from an .axim file.

Usage:
    python generate.py model.axim "The capital of France is"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from axim.core import load_axim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "nanochat"))
from nanochat.gpt import GPT, GPTConfig


def generate(model, tokenizer, prompt_text, max_tokens=50, temperature=0.7, device="cpu"):
    """Generate text continuation from prompt."""
    # Build prompt tokens with conversation format
    bos = tokenizer.encode_single_token("<|bos|>")
    user_start = tokenizer.encode_single_token("<|user_start|>")
    user_end = tokenizer.encode_single_token("<|user_end|>")
    assistant_start = tokenizer.encode_single_token("<|assistant_start|>")
    assistant_end = tokenizer.encode_single_token("<|assistant_end|>")
    
    # Simple user message format
    prompt_ids = [bos, user_start]
    prompt_ids.extend(tokenizer.encode_ordinary(prompt_text))
    prompt_ids.extend([user_end, assistant_start])
    
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)
            logits = logits[:, -1, :]
            
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            
            token = next_id.item()
            
            # Stop on special tokens
            if token == assistant_end or token == bos:
                break
            
            generated.append(token)
            ids = torch.cat((ids, next_id), dim=1)
            
            # Prevent exceeding sequence length
            if ids.size(1) >= model.config.sequence_len:
                break
    
    return tokenizer.decode(generated)


def main():
    if len(sys.argv) < 2:
        print("usage: python generate.py <model.axim> [prompt]")
        print('example: python generate.py model.axim "The capital of France is"')
        return
    
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    
    print(f"loading {path}...")
    data = load_axim(path)
    cfg = data["config"]
    
    print(f"model: {cfg.get('model_type', '?')}")
    print(f"params: {sum(v.numel() for v in data['weights'].values()):,}")
    print()
    
    # Build model
    config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_kv_head=cfg["num_key_value_heads"],
        n_embd=cfg["hidden_size"],
        sequence_len=cfg["max_position_embeddings"],
        window_pattern=cfg["window_pattern"],
    )
    
    device = torch.device("cpu")
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(data["weights"], strict=True, assign=True)
    model = model.to(device)
    model.eval()
    
    tokenizer = data["tokenizer"]
    
    print(f"prompt: {prompt!r}")
    print("generating...")
    print()
    
    result = generate(model, tokenizer, prompt, max_tokens=30, temperature=0.7, device=device)
    
    print(f"{prompt}{result}")


if __name__ == "__main__":
    main()
