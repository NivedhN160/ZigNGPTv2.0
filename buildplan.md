## ZigNGPTv2.0 — Build Plan

new repo, old model, same personality, new brain underneath. Here's the order, with what to build at each step and how to know it's working before moving on.

---

### 0. Repo structure (set this up first)
```
zigngpt/
├── src/
│   ├── engine_v1/        ← your existing trigram code, untouched
│   ├── engine_v2/         ← everything new goes here
│   │   ├── tensor.zig
│   │   ├── autograd.zig
│   │   ├── nn.zig          (Linear, embedding, layernorm, optimizer)
│   │   ├── tokenizer.zig
│   │   ├── transformer.zig
│   │   └── train.zig
│   ├── cli.zig            ← existing chat interface, untouched for now
│   └── main.zig
```
Keep `engine_v1` working and callable the whole time. You're adding, not replacing, until the very end.

---

### 1. Tensor struct
Build in `tensor.zig`:
- `Tensor` struct: flat `[]f32` + shape (`[2]usize` for 2D to start)
- `matmul`, `add`, `transpose`, `sum`
- **Test:** multiply two 2x2 matrices you've solved by hand on paper, assert exact match. Do this for every op before moving on.

---

### 2. Autograd engine — scalars first
In `autograd.zig`:
- A `Value` node: holds `data: f32`, `grad: f32`, parent pointers, and a backward function
- Implement ops one at a time, each with its backward rule immediately:
  - `add` (grad passes through unchanged)
  - `mul` (grad = other parent's value)
  - `relu` (grad = 1 if input > 0, else 0)
- `backward()`: topological sort the graph, walk it in reverse, accumulate gradients
- **Build a numerical gradient checker now, not later:** perturb input by epsilon, compare numerical slope to your analytical grad, assert close. Run this after every op you add from now on.

Get this working on plain numbers before touching tensors.

---

### 3. Extend autograd to tensors
- Same `Value`/backward pattern, but operating on your `Tensor` type
- Add `matmul` and `softmax` backward rules — these are the two most likely to have subtle bugs, so gradient-check them extra carefully
- Use an arena allocator per forward+backward pass; free the arena after each training step

---

### 4. Basic NN layers + optimizer
In `nn.zig`:
- `Linear` layer (weights + bias via your matmul/add)
- Embedding lookup (token id → row in a weight matrix)
- Layer norm
- SGD optimizer first, then Adam (needs momentum + variance buffers per parameter)
- **Test:** train this tiny stack on XOR. If loss doesn't go down, stop and debug here — don't proceed with a broken foundation.

---

### 5. Tokenizer
In `tokenizer.zig`:
- v1: character-level (just map bytes to ids) — get the full pipeline working end to end first
- v2 later: basic BPE (merge most frequent byte pairs iteratively) — good from-scratch exercise, do it after the model trains successfully on char-level

---

### 6. Transformer
In `transformer.zig`, built only from your own primitives:
1. Token + positional embeddings
2. Self-attention head: Q/K/V via Linear, scaled dot-product via matmul + softmax, causal mask (upper triangular)
3. Multi-head attention: parallel heads, concatenated
4. Feed-forward block: Linear → GELU → Linear
5. Transformer block: attention + feed-forward + residual connections + layer norm (pre-norm)
6. Output: Linear → vocab size → softmax

Start tiny: 2 layers, 2 heads, embedding dim 32-64. Get *any* coherent output before scaling up.

---

### 7. Training loop
In `train.zig`:
- Cross-entropy loss on next-token prediction
- Loop: forward → loss → `backward()` → optimizer step
- Train on the same Bhagavad Gita + Sherlock Holmes corpus NGPTv1 already uses (so you can do direct v1-vs-v2 comparisons later)

---

### 8. Wire it into the existing CLI 
- Swap the model NGPT calls under the hood — keep `/sarcastic`, `/haiku`, `/simulate` exactly as they are
- Add a new command: `/engine v1` vs `/engine v2` to switch backends live, for side-by-side demos
- Optional: `/attention` command that prints which previous tokens the model weighted most for its last prediction

---

### 9. Make it fast (open-ended, after it works)
- Profile — time will be dominated by matmul
- Vectorize matmul using Zig's `@Vector` SIMD types
- Thread pool for batch processing
- Reduce allocation overhead with pooled/arena strategies

---

### Document as you go
Keep a running bug log: wrong gradients caught by your numerical checker, Zig memory bugs, before/after performance numbers from vectorizing. This log — plus a v1-vs-v2 sarcasm-mode comparison on the same prompt — is your strongest portfolio artifact from this whole project.

Don't open `transformer.zig` until XOR trains correctly on your own autograd engine — that's the checkpoint that tells you the foundation is solid.