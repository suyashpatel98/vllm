# What breaks when serving Inkling 2 Mini on an RTX 4090 (SM89)

Task ticket [#3](https://github.com/suyashpatel98/vllm/issues/3) (map [#1](https://github.com/suyashpatel98/vllm/issues/1)).
Run on 2026-11-25 on a RunPod secure RTX 4090 (driver 595.112.07, CUDA 13.2) with precompiled vLLM `7b7af30196`
(`cu132`), `torch 2.14.0+cu132`, checkpoint `thinkingmachines/Inkling-2-Mini`.

Each attempt is `try.sh` (in this folder): `vllm serve` with `--enforce-eager --max-model-len 4096
--max-num-batched-tokens 1024 --max-num-seqs 4 --gpu-memory-utilization 0.92` plus the extra flags listed, then
one chat request. Bypasses are the commits on this branch, applied one at a time with `git cherry-pick`.

| # | Code state | Extra flags | Result | Log |
|---|---|---|---|---|
| 1 | min capability 90 → 89 | | Weights load (18.24 GiB, 15 s). **Encoder profiling crashes**: `CUDA error (vllm-flash-attn-src/hopper/flash_fwd_launch_template.h:172): no kernel image is available` (FA3 in `vision.py`) | `serve_a1-gate.log` |
| 2 | + vision FA2 | | Encoder profile passes. **Text profile run crashes in the first layer**: `ptxas … Instruction 'griddepcontrol' requires .target sm_90 or higher` compiling `_gate_sink_kernel` (`ops/attention.py:321`, via `attention.py:527`) | `serve_a2-fa2.log` |
| 3 | + PDL removed | | **Crashes at the first conv/mix**: `RuntimeError: launch, csrc/inkling2/conv_mix.cu:978, no kernel image is available` (`ops/conv_mix.py:124`, from `loop.py:279`) | `serve_a3-pdl.log` |
| 4 | + Triton conv/mix fallback | | **Whole profile run passes** (Humming NVFP4 + MXFP8 experts, `rel_attention`, router, short conv, AttnRes); first run spends ~2 min in Triton JIT. **Fails the KV check**: 4096 tokens need 2.13 GiB of KV, 1.39 GiB available (max ≈ 1,472) | `serve_a4-convmix.log` |
| 5 | same | `--gpu-memory-utilization 0.97` | Same check: 1.64 GiB available (max ≈ 1,760) | `serve_a5-mem.log` |
| 6 | same | `0.97`, `--max-model-len 1536` | **Serves.** Text and image prompts answer coherently (below) | `serve_a6-len1536.log` |
| 7 | same | `0.97`, `1536`, `--max-num-batched-tokens 2048` | **Serves.** The 2,048-token profile run takes the TMA router GEMM without error | `serve_a7-tma.log` |

## Smoke output (attempt 6, temperature 0, `probe.py`)

No `--reasoning-parser` was set, so the model's reasoning appears inline before the answer.

- *"What is the capital of France? Answer in one sentence."* → reasoning, then "The capital of France is Paris." (82 tokens, 6.2 s)
- *"Write a Python function that returns the n-th Fibonacci number iteratively."* → a correct iterative
  `fibonacci(n)` with traced checks for n = 0..4 (hit the 600-token cap, 44.9 s, about 13 tok/s)
- A 448×448 PNG of a red square (upper left) and a blue circle (lower right) → "The image shows a red square in
  the upper-left area and a blue circle in the lower-right area, on a white background." (94 tokens, 11.5 s)

## TMA router GEMM check (`tma.py`)

`k_group_partials` called directly against `x.float() @ w.float().T` (128 × 2304 router weight):

| Tokens | Path | Max abs error |
|---|---|---|
| 1024 | non-TMA, 3 K-groups | 2.9e-4 |
| 2048 | TMA, 4 stages | 3.7e-4 |
| 4096 | TMA, 3 stages | 3.4e-4 |
| 8192 | TMA, 3 stages | 4.2e-4 |

The compiled kernels use 36 KiB (4 stages, BN 64) and 64 KiB (3 stages, BN 128) of shared memory. Triton's
descriptor-to-pointer rewrite on SM89 does not keep the Hopper multi-buffering, so both fit under 99 KiB.

## Memory (attempt 6)

- Weights plus non-torch: 18.35 GiB. Profile-run activation peak: 0.26 GiB (0.46 GiB at 2,048-token batches).
- **Warmup peak 1.54 GiB**, and the extensible KV cache keeps 1.87 GiB aside for transient activations. After
  weights, this is the largest consumer, much larger than the profile peak.
- KV cache 1.64 GiB = 1,765 tokens at `max-model-len 1536` (840 blocks of about 2 MiB).
- vLLM's estimate for one 4,096-token request is 2.13 GiB. The
  [inventory](https://github.com/suyashpatel98/vllm/blob/research/inkling2-arch-inventory/docs/research/inkling2-arch-inventory.md)
  estimated about 0.3 GiB (32 KiB per token plus 170 MiB per request), so the real per-token cost is several
  times higher. The memory-fit work has to explain this.
