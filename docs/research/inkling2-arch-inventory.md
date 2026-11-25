# Inkling 2: inventory of architecture-specific code paths (for SM89)

Research for [#5](https://github.com/suyashpatel98/vllm/issues/5) (map [#1](https://github.com/suyashpatel98/vllm/issues/1)).
Gathered 2026-11-25 from this repo at `7b7af30196`, plus primary sources for the toolchain: Triton `main`
(`third_party/nvidia/backend/compiler.py`, `third_party/nvidia/language/cuda/gdc.py`,
`TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp`), LLVM `main` (`NVPTXIntrinsics.td`, `NVPTXInstrInfo.td`),
the CUDA Programming Guide compute-capability tables, the Humming README and the HF `config.json` of
`thinkingmachines/Inkling-2-Mini` (revision `c9d028302c`). Line numbers refer to `7b7af30196`.

Scope: `Inkling2LoopForConditionalGeneration` (Inkling 2 Mini), TP=1, text + image, no DSpark, BF16 KV cache,
on an RTX 4090 (SM89, 24 GB).

## TL;DR

- **On SM89 the model takes the Hopper path by default.** Every dispatch is "SM100 family → Blackwell kernels,
  SM12x family → SM12x kernels, otherwise Hopper". Nothing tests for SM89 specifically, so it lands on the
  Hopper branches: Humming experts, the Triton relative attention (`attention_sm90.py`), cuBLAS linears.
- **Five things break, in the order a run would hit them:**
  1. **Start-up refusal.** `Inkling2QuantConfig.get_min_capability()` returns 90 (`quant_config.py:45-46`), and
     `VllmConfig` raises at `config/vllm.py:998-1006`.
  2. **PDL in nearly every Triton kernel.** About 14 kernels on the TP1 path call `tl.extra.cuda.gdc_wait()`
     or `gdc_launch_dependents()` with no guard. Triton lowers these to `griddepcontrol`, which LLVM selects
     only for SM90+. The first launch fails to compile.
  3. **The fused conv/mix CUDA op.** The loop executor calls `torch.ops._C.inkling2_conv_mix` in every layer,
     including the profile run. It is built only for `9.0a;10.0f;10.3f` and uses `cp.async.bulk`/`mbarrier`
     transactions, which are SM90 instructions. A Triton fallback already exists (the SM12x branch).
  4. **Vision attention asks for FA3.** `vision.py:94` picks `fa_version=3` on anything that is not SM12x,
     and FA3 is Hopper-only. FA2 is the fallback, as on SM12x.
  5. **The TMA router GEMM.** It runs for batches of 2,048 or more tokens (`linear.py:217-238`). Triton
     rewrites its descriptors to plain pointers below SM90, so it compiles, but its Hopper stage counts need
     well over 99 KiB of shared memory. The non-TMA kernel, with SM12x's 2-stage cap, is the fallback.
- **No item needs a new kernel.** Each has a fallback in-tree: the SM12x branches (SM12x and SM89 both have
  99 KiB of shared memory per block and no tcgen05), the Hopper Triton attention, and the Humming experts.
  The PDL fix is mechanical: vLLM's other Triton kernels already guard `gdc_wait` with a
  `launch_pdl: tl.constexpr` set from `current_platform.is_arch_support_pdl()`.
- **The expert GEMMs stay on Humming (BF16 activations, NVFP4/MXFP8 weights).** The Humming README lists
  BF16-activation GEMMs with any FP weight type up to 8 bits on SM80+, and `HummingConfig.get_min_capability()`
  is 75. Tuning on SM89 is for [#4](https://github.com/suyashpatel98/vllm/issues/4) to settle.
- **The csrc/inkling2 kernels are absent for SM89 in every build**, precompiled or from source (see section 3).
- **Memory: plausible but tight.** Weights are about 19.5 GB (18.2 GiB) because the experts must stay packed:
  BF16 experts alone would be 48.9 GB. KV and conv state cost about 32 KiB per token at the margin plus about
  170 MiB per request, so a 32K-token request needs about 1.2 GiB. That leaves roughly 2-3 GiB at
  `--gpu-memory-utilization 0.95` for activations, CUDA graphs and the vision encoder. Section 5 has the
  details; measure it on the pod.

## 1. Which paths a TP1, text + image, no-DSpark run exercises

Inkling 2 Mini is `model_type: inkling2_loop` with `loop_config {core_start 6, core_end 18, num_passes 3}`
(HF `config.json`). `Inkling2ForConditionalGeneration.__init__` builds an `Inkling2LoopModel`
(`model.py:569-573`), whose `forward` sends every layer visit through `_run_layer` (`loop.py:202-319`). At
TP1, `allreduce_workspace` stays `None` (it is only created under `world_size > 1`, `model.py:274`), so
`plan = [(False, False)] * 2` (`loop.py:336`). Every sublayer therefore takes the `conv_mix` branch
(`loop.py:279`, `loop.py:312`), never `all_reduce_mix`.

Per layer visit (48 visits: 6 prelude + 3 × 12 core + 6 coda), the code runs:

- **Attention.** `Inkling2Attention.forward` (`attention.py:370`) runs `linear` (cuBLAS on SM89), the
  `attention_prologue` Triton kernels (conv K/V, clip q/r, relative logits), then decode `rel_decode_attention`
  and/or prefill `rel_attention` (Triton, `attention_sm90.py`), `gate_sink`, and the output `linear`.
- **Hidden-stream conv and mixing.** `Inkling2ShortConv.conv_mix` → `ops/conv_mix.conv_mix`, either the csrc
  op or the SM12x Triton fallback.
- **Loop messages (passes 2-3).** `attn_res`, `pair_rms_norm`, `torch.bmm` and `blend_rms_norm`
  (`loop.py:113-133`).
- **MoE.** `Inkling2MoE.forward` → `select_experts` (`k_group_partials` + `_select_kernel`),
  `moe_align_block_size`, Humming `w13`, `swiglu`, Humming `w2`, then `combine` (`moe.py:354-397`,
  `moe_sm90.py`).
- **Vision (image requests only).** `Inkling2VisionEncoder`: Triton patch/rotary/norm kernels, `nn.Linear`,
  and `flash_attn_varlen_func`.

## 2. Inventory table

"Path?" says whether a TP1, text + image, no-DSpark run on SM89 reaches the code. "Today on SM89" is what this
tree does at `7b7af30196`.

### 2a. Host-side gates (capability checks, imports)

| Module / function | Path? | Today on SM89 | Existing fallback | Verdict |
|---|---|---|---|---|
| `Inkling2QuantConfig.get_min_capability` → 90 (`quant_config.py:45-46`), enforced in `VllmConfig._get_quantization_config` (`config/vllm.py:998-1006`) | yes, startup | **ValueError**: "Minimum capability: 90. Current capability: 89." | none needed: nothing in the Hopper path is SM90-only once rows 2b-1 to 2b-5 are fixed. Humming BF16 needs SM80 | change the number |
| `use_sm100_kernels()` = `is_device_capability_family(100)` (`ops/__init__.py:6-13`) | yes | False: every caller takes its non-Blackwell branch. Callers: `attention.py:307` (FA4 warmup skipped), `attention.py:618` (prefill → `rel_attention`), `ops/attention.py:1233` (decode → `rel_attention`), `linear.py:34` (`F.linear`, not FlashInfer `mm_bf16`), `moe.py:150,160,189,194,205,277,385` (Humming), `trtllm_bmm.py:144` (plan only), `vision.py:86` (no tml_fa4), `nvfp4.py:21-23` (software e2m1, off-path) | Hopper branches | OK as is |
| `is_device_capability_family(120)` sites | mixed | False everywhere, so SM89 gets the Hopper choice. `conv_mix.py:92` → csrc op (**breaks**, 2b-2). `vision.py:94` → FA3 (**breaks**, 2b-3). `linear.py:216-247` → Hopper stage counts (**smem risk**, 2b-4). `vision.py:48` → `nn.Linear` (OK). `attention.py:295-298` → KV dtype `auto`/`bfloat16` only (OK; fp8 KV is SM12x-only). `ll_gemm_sm120.py:255,361` → low-M GEMM off, falls back to torch (OK). `moe.py:151` → Humming (OK). `speculator_config.py:18` (DSpark, off-path). `config/vllm.py:1600-1607` (fp16 KV on SM12x, off-path) | SM12x branches are the right model for SM89: 99 KiB/block shared memory, no tcgen05 | flip the three marked gates |
| Raw `torch.cuda.get_device_capability(device)[0] in (9, 10)` (`ops/allreduce.py:86`) | no (TP>1) | not reached: `Inkling2AllReduceWorkspace.create` runs only when TP>1 (`model.py:274-279`); it also needs the csrc op (`allreduce.py:75`) | n/a | out of scope |
| FlashInfer imports: `linear.py:37` (`mm_bf16`), `trtllm_bmm.py:42-44`, `allreduce.py:30,79`, `seq_parallel.py:45`, `dspark.py:62,85` | no | all inside functions behind `use_sm100_kernels()`, TP>1 or DSpark. `linear.py:10` imports only `vllm.utils.flashinfer.is_flashinfer_autotuning`, which is safe | n/a | OK |
| tml_fa4 / CuTe DSL imports: `ops/attention.py:121`, `vision.py:88`, `inkling/nvidia/ops/fa4_rel_attention.py:39` | no | function-local, Blackwell only. The module-level import `inkling/nvidia/ops/fa4_rel_attention` pulls in only helpers (`bucket_max_seqlen_q`, `_num_warps_bucket`) | n/a | OK |
| Gluon `_rel_verify_kernel` (tcgen05, TMA, warp-specialized; `ops/attention.py:856`) | no | defined at import (gluon imports are arch-neutral), launched only when `use_sm100_kernels()` (`ops/attention.py:1233`) | `rel_attention` | OK |
| Attention metadata: `Inkling2AttentionBackend(FlashAttentionBackend)` (`attention.py:107-110`) | yes | generic: `supports_compute_capability` is `>= 8.0` (`v1/attention/backends/flash_attn.py:551-552`) | n/a | OK |
| Inkling2 bits outside the model dir: `config/vllm.py:122-123` (breakable-CUDA-graph architectures, arch-neutral on CUDA, `:170-173`), `config/model.py:837-838,1493` (tokenizer mode, quant override), `config/speculative.py:1260` (DSpark), `config/lora.py:156`, `model_loader/weight_utils.py:253-254` (`hf_quant_config.json`), `transformers_utils/config.py:239-241`, `v1/worker/gpu/cudagraph_utils.py:967` (comment only) | yes | none is architecture-specific | n/a | OK |

### 2b. Kernels and ops on the TP1 path

| # | Module / function | Today on SM89 | Existing fallback | Verdict |
|---|---|---|---|---|
| 2b-1 | **PDL in the Triton kernels.** Unconditional `tl.extra.cuda.gdc_wait()` / `gdc_launch_dependents()` in: `attn_res._attn_res_kernel` / `_attn_res_row_kernel` (`ops/attn_res.py:77,191`); `sconv._conv` / `_sconv_kernel` (`ops/sconv.py:251,338`); `_attn_prologue_kernel` (`:557,559`); `_qr_rel_kernel` (`:703`); `linear._k_group_kernel` / `_k_group_sum_kernel` / `_k_group_tma_kernel` (`ops/linear.py:134,153,181,183`); `attention._gate_sink_kernel` / `_rel_decode_combine_kernel` (`ops/attention.py:288,469`); `moe._combine_kernel` (`ops/moe.py:1072-1074`). Launch sites pass `launch_pdl=True` (e.g. `attn_res.py:294`, `sconv.py:461,983,1015`, `linear.py:237,271,289`, `attention.py:336,1298`, `moe.py:1145`) | **Compile failure on first launch.** Triton lowers `gdc_wait` to `NVVM::GriddepcontrolOp` with no capability check (`BarrierOpToLLVM.cpp:56-68`). LLVM defines `griddepcontrol.wait` / `.launch_dependents` only under `Predicates = [SM90]` (`NVPTXIntrinsics.td:6146-6151`). Triton's own docs say "Programmatic dependent launch is supported on SM90 (Hopper) and beyond" (`gdc.py`), and its tutorial gates it on capability >= 9 | Semantics: `gdc_wait` only waits for the previous grid, and without PDL launches are already stream-ordered, so dropping it is correct. Pattern in-tree: a `launch_pdl: tl.constexpr` kernel argument plus `launch_pdl=current_platform.is_arch_support_pdl()` (`platforms/cuda.py:801-807`; e.g. `layers/mamba/ops/causal_conv1d.py:65,764`, `kernels/triton/fused_pre_fc_norms.py:102,160`) | **mechanical new code**: one constexpr guard, about 14 kernels, about 12 launch sites. Whether `launch_pdl=True` alone is harmless on SM89 is unverified; guard it too |
| 2b-2 | **`conv_mix` → `torch.ops._C.inkling2_conv_mix`** (`ops/conv_mix.py:120-125`), called from `loop.py:279,312` on every layer visit and in the profile run (`sconv.py:178-180`) | **Fails at the first forward.** The kernel is built only for `9.0a;10.0f;10.3f` (section 3) and uses `cp.async.bulk…mbarrier::complete_tx`, `cp.async.bulk.prefetch.L2` and `mbarrier.try_wait.parity` (`csrc/inkling2/conv_mix.cu:300-330`); LLVM gates `cp.async.bulk` on SM90 (`NVPTXInstrInfo.td:240-247`) | **SM12x branch** `conv_mix.py:92-119`: Triton `short_conv` (writes the conv-state cache, residual update) + `attn_res` with the same arguments. Its comment says "Native conv/mix kernels require Hopper or SM10x" | gate change (`family(120)` → "not SM90/SM10x"), plus 2b-1 for its kernels |
| 2b-3 | **Vision attention** `flash_attn_varlen_func(..., fa_version=3)` (`vision.py:91-95`) | **Fails.** `_is_fa3_supported` → "FA3 is only supported on devices with compute capability 9.x" (`vllm_flash_attn/flash_attn_interface.py:62-69`) | FA2 (SM80+, `flash_attn_interface.py:55-59`), already used on SM12x. Head dim 1024/16 = 64 | one-line change |
| 2b-4 | **Router GEMM** `k_group_partials` (`ops/linear.py:202-273`), from `select_experts` (`moe_sm90.py:86`); TMA branch for 2,048+ tokens (`linear.py:217-238`) | Compiles: Triton runs `add_rewrite_tensor_descriptor_to_pointer` when `capability // 10 < 9` (`compiler.py:296-297`). Shared memory: per stage x `128×128` + w `BN×128` bf16 = 48 KiB (BN 64) or 64 KiB (BN 128), at 4 or 3 stages, which is far above SM89's **99 KiB per block** (CUDA Guide Table 34). **Expected `OutOfResources` on prefill chunks of 2,048+ tokens.** `tl.range(..., warp_specialize=True)` on SM8x goes through `add_hopper_warpspec` (`compiler.py:331-340`); the effect there is unverified. Below 2,048 tokens the non-TMA kernel's configs are 96 KiB or less (fits, barely) | non-TMA `_k_group_kernel` with the SM12x cap `num_stages = min(num_stages, 2)` (`linear.py:246-247`); the comment at `:221` states the same 99 KiB limit | gate change (skip TMA below SM90; apply the small-smem stage cap to SM89), plus 2b-1. Workaround meanwhile: `--max-num-batched-tokens < 2048` |
| 2b-5 | **Relative attention** `rel_attention` / `_rel_attention_kernel` (`ops/attention_sm90.py:60-343`), for prefill (`attention.py:618-634`) and decode (`ops/attention.py:1250-1268`) | Plain Triton (`tl.dot` → `mma.sync`); no PDL, no TMA, no fp8 unless the KV cache is fp8. Config: BLOCK_M up to 128, BLOCK_N 64, 8 warps, 2 stages (`:338-343`), the same config that runs on SM12x (docstring `:3`), whose 99 KiB/block limit equals SM89's | it **is** the fallback (the Hopper/SM12x stand-in for tml_fa4 and tcgen05) | expected to work; verify on pod |
| 2b-6 | `attention_prologue` (`ops/sconv.py:832`) and `short_conv` (`:388`) | PDL only (2b-1); otherwise generic Triton, no dot or TMA | — | fixed by 2b-1 |
| 2b-7 | `attn_res` (`ops/attn_res.py:244`), `gate_sink` (`ops/attention.py:302`), `_rel_decode_combine_kernel` (`:442`), `combine` (`ops/moe.py:1095`) | PDL only | — | fixed by 2b-1 |
| 2b-8 | `rms_norm` (`ops/norm.py:47`), `pair_rms_norm` / `blend_rms_norm` (`ops/loop.py:42,97`), `compute_log_scaling_tau` (`ops/attention.py:356`), `_select_kernel` / `_swiglu_kernel` (`moe_sm90.py:34,112`), vision Triton kernels (`ops/vision.py`) | Generic: no PDL, TMA or fp8. Their inline asm (`tanh.approx.f32`, `ex2.approx`, `rcp.approx`, `rsqrt.approx`, `cvt.rn.bf16.f32`) is all legal before SM90 | — | OK |
| 2b-9 | **Expert GEMMs** `HummingExperts` (`moe_sm90.py:160-316`) via `vllm.utils.humming.humming_forward`; NVFP4 layers 0-19, MXFP8 layers 20-23; BF16 activations | Not tested here. Humming README: "every NVIDIA architecture from SM75 onward", BF16 activations "SM80+" with "arbitrary signed FP (kBits ≤ 8)" weights; `HummingConfig.get_min_capability()` = 75 (`quantization/humming.py:172-173`). Tuning comes from `get_heuristics_config` (`moe_sm90.py:222-226`); whether its SM89 picks fit 99 KiB is unknown | it **is** the fallback (Hopper's) | expected to work; details in [#4](https://github.com/suyashpatel98/vllm/issues/4) |
| 2b-10 | `moe_align_block_size` (vLLM `_moe_C`), dense `F.linear` / `torch.mm` / `torch.bmm` (cuBLAS), `Inkling2LogitsProcessor` → base `_apply_head` (`model.py:84-103`) | generic, built for all CUDA archs | — | OK |
| 2b-11 | Vision linears `_linear` → `nn.Linear` (`vision.py:46-49`); encoder CUDA graphs (`model.py:617-720`) | generic (cuBLAS) | `_SM120Linear` (`linear_k_groups`) exists if a padding-invariant GEMM is needed for graph replay | OK; watch the encoder CUDA-graph replay test |

### 2c. Hopper+/Blackwell features present in the tree but off the SM89 path

| Feature | Where | Why not reached at TP1 / no DSpark |
|---|---|---|
| tcgen05 / TMEM / gluon warp specialization | `ops/attention.py:548-1102` (`_rel_verify_kernel`) | `use_sm100_kernels()` |
| tml_fa4 (CuTe DSL FA4) | `ops/attention.py:84-249`, `vision.py:86-89` | `use_sm100_kernels()` |
| TRT-LLM-gen expert GEMMs (FlashInfer JIT, `trtllm_bmm.cu`) | `ops/trtllm_bmm.py:41-70`, `moe.py:160-165` | `use_sm100_kernels()` |
| `tl.dot_scaled` + `cvt.rn.satfinite.e2m1x2` (FP4 MMA) | `ops/moe_sm120.py:18-77`, `ops/nvfp4.py:67-87,133-163` | SM12x only (`moe.py:205,248`); the e2m1 cvt is SM100f/110f/120f only in LLVM (`NVPTXInstrInfo.td:254-258`) |
| fp8 `tl.float8e4nv` stores (NVFP4 scales, MXFP8 values) | `ops/nvfp4.py:103-128,204`, `ops/mxfp8.py:30`, `ops/activation.py:210-278`, `ops/moe.py:866-1056` (`route`) | the packed-activation expert path (`moe.py:277`: SM100/SM12x). On SM89 these compile (Triton adds `fp8e4nv` for capability >= 89, `compiler.py:249-253`), which matters only for unit tests |
| fp8 KV dequant `cvt.rn.f16x2.e4m3x2` | `ops/attention_sm90.py:38-57` | fp8 KV is allowed on SM12x only (`attention.py:295-298`). LLVM permits this cvt on SM89 with PTX 8.1 (`NVPTXInstrInfo.td:249-252`), so it is a possible memory lever later |
| Low-M BF16 GEMMs (`ll_gemm_sm120.py`) | `qkv_bmm`, `sm12x_bf16_ll_gemm` | gated on `family(120)` (`:255,361`) |
| NVLink multicast all-reduce, sequence parallel, `allreduce_conv_mix.cu`, `seq_parallel.cu` | `ops/allreduce.py`, `ops/seq_parallel.py`, `csrc/inkling2/` | TP>1 only (`model.py:274`) |
| DSpark draft (FlashInfer RMSNorm / fused all-reduce) | `dspark.py`, `ops/dspark.py`, `speculator*.py` | no `--speculative-config` |
| LoRA grouped GEMM | `lora.py`, `ops/grouped_gemm.py` | no LoRA |

## 3. Are the csrc/inkling2 kernels in a wheel for SM89?

**No.** `CMakeLists.txt:1206-1228` builds `allreduce_conv_mix.cu`, `conv_mix.cu` and `seq_parallel.cu` only
for `cuda_archs_loose_intersection("9.0a;10.0f;10.3f" | "9.0a;10.0a;10.3a", CUDA_ARCHS)`; 8.9 is in neither
list. `VLLM_ENABLE_INKLING2_KERNELS` (`CMakeLists.txt:1377-1379`) is defined whenever that intersection is
non-empty, and it gates both the schema and the impl registration (`csrc/torch_bindings.cpp:629-663,974-980`).

- **Precompiled wheel** (`VLLM_USE_PRECOMPILED=1`, the AGENTS.md workflow). The release build uses
  `TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'` (`docker/Dockerfile:280-281`). 9.0 matches
  `9.0a`, so `torch.ops._C.inkling2_conv_mix` **is registered** but holds only sm_90a/sm_100f code. vLLM drops
  `+PTX` from the global flags (`CMakeLists.txt:219-220`, `docker/Dockerfile:277-279`), and `sm_90a` code
  cannot run on sm_89 anyway. The launch at `conv_mix.py:124` therefore fails at run time; the expected error
  is CUDA's "no kernel image is available for execution on the device" (not reproduced yet).
- **Source build with `TORCH_CUDA_ARCH_LIST=8.9`.** The intersection is empty, the op is not registered, and
  `torch.ops._C.inkling2_conv_mix` raises `AttributeError`.
- Adding `8.9` to the arch list would not help: the kernel uses SM90-only PTX (bulk async copies with mbarrier
  transactions, `conv_mix.cu:285-330`). The fix is the Triton fallback in row 2b-2, which also avoids any
  C++ change under the precompiled-wheel workflow.

## 4. What the SM89 path looks like

| Component | SM89 choice | Source of that code |
|---|---|---|
| Expert GEMMs (NVFP4 / MXFP8 weights, BF16 activations) | Humming | Hopper path (`moe_sm90.py`) |
| Router, SwiGLU, combine | `select_experts` / `swiglu` / `combine` Triton | Hopper path |
| Attention (prefill + decode) | `rel_attention` Triton (`mma.sync`) | Hopper/SM12x path |
| Prologue, short convs, AttnRes, norms, loop blend | existing Triton kernels | shared |
| Hidden-stream conv + mix | `short_conv` + `attn_res` | SM12x branch of `conv_mix.py` |
| Dense projections, LM head | cuBLAS | Hopper path |
| Vision attention | FA2 | SM12x branch |
| Router GEMM, 2,048+ tokens | non-TMA `_k_group_kernel`, 2 stages | SM12x stage cap |

New code: a PDL constexpr guard (mechanical), three gate flips, and the min-capability change. No new kernel.

## 5. Memory: weights, cache per token, and whether 32K context fits in 24 GB

### Inputs

From `config.json`:

- **Text model:** hidden 2304, 24 layers. Full-attention layers are 5, 11, 17, 23 (the complement of
  `local_layer_ids`). The other 20 use a 1,024-token sliding window. Attention has 8 KV heads × 128 dims,
  128 experts (top-8), intermediate 1152, vocab 201,024, and the loop core is layers 6-17 × 3 passes.
- **Vision:** a ViT with width 1024 and 23 layers.

### Weights (packed as in the checkpoint)

| Part | Parameters | Bytes |
|---|---|---|
| Routed experts, NVFP4, layers 0-19 | 20 × 128 × (2304·2304 + 2304·1152) = 20.4 B | 0.5 B + 1/16 B scale per param → **11.5 GB** |
| Routed experts, MXFP8, layers 20-23 | 4.08 B | 1 B + 1/32 B → **4.2 GB** |
| Embedding + LM head (BF16) | 2 × 201,024 × 2304 = 0.93 B | **1.85 GB** |
| Attention, router, loop controllers (BF16) | ≈ 27 M × 24 + 28 M ≈ 0.68 B | **1.36 GB** |
| Vision tower + projector (BF16) | ≈ 0.32 B | **0.63 GB** |
| **Total** | ≈ 26.4 B | **≈ 19.5 GB ≈ 18.2 GiB** (the recipe says "~20 GB on disk including the 0.6 GB DSpark draft") |

Dequantizing the experts is not an option. As BF16 they take 24.5 B × 2 = **48.9 GB**, and as FP8 about
**25 GB**. They have to stay packed and be dequantized inside the GEMM (Humming W4A16 / W8A16).

### Cache

The cache spans 48 attention visits and 48 conv-state visits (`model.py:182-193`, `attention.py:269-278`).

- **Full attention: 8 visits** (5, 23 once; 11 and 17 three times each). Each costs 8 × 128 × 2 (K, V) × 2 B
  = 4 KiB per token, so **32 KiB per token**. This is the only cost that grows with context length.
- **Sliding window: 40 visits.** Each request holds about 1,024 + 16 tokens per visit, roughly
  40 × 1,040 × 4 KiB ≈ **163 MiB per request**, whatever the context length.
- **Conv state: 48 visits** of a sliding window of `kernel_size` = 4 tokens. With `replicate_hidden`, the
  packed head is 2·128 + 2·(2304/8) = 832, padded to 1024, so 8 × 1024 × 2 B = 16 KiB per token slot
  (`inkling/nvidia/sconv_swa_attn.py:150-163`). A 4-token block is then 64 KiB, the same page as a 16-token
  attention block. Each request uses about 2 blocks per visit, so about **6 MiB per request**.
- **Allocator.** It buckets these into full (8), sliding (40) and conv (48) layers, takes group size 8 (the
  smallest bucket, `v1/core/kv_cache_planning.py:374-385`) and so forms 1 + 5 + 6 = 12 groups with no padding.
  The capacity it logs is the per-request block sum over all groups at `max_model_len`
  (`kv_cache_utils.py:1156-1170`).
- **One 32K-token request:** 2,048 + 5 × 65 + 6 × 2 = 2,385 blocks of 8 layers × 64 KiB, which is
  **about 1.16 GiB**. 64K is about 2.2 GiB.
- **Cross-check.** The recipe states "~69 KB of cache per token" (GB200, TP1, measured with DSpark,
  `recipes/models/thinkingmachines/Inkling-2-Mini.yaml`). That is about twice the marginal figure derived
  here. Even at 69 KB per token, 32K is about 2.2 GB.

### Does 32K fit?

- **Budget.** An RTX 4090 has 24,564 MiB (about 24.0 GiB). At `--gpu-memory-utilization 0.95` the budget is
  about 22.8 GiB. Minus 18.2 GiB of weights, **about 4.6 GiB** is left. A 32K request takes 1.2-2.2 GiB of
  that, leaving **about 2.4-3.4 GiB** for activations, Humming and CUDA-graph pools and the encoder cache.
- **Activation sizes per 2,048-token chunk:**
  - `rel_logits` (T+128) × 24 heads × 1280 × 2 B ≈ 134 MB per attention call (`attention.py:449-451`)
  - MoE gate/up rows T × 8 × 2304 × 2 B ≈ 75 MB
  - AttnRes stack ≈ 6 × T × 2304 × 2 B ≈ 57 MB
  - fp32 logits, num_seqs × 201,024 × 4 B (0.8 MB per sequence)

  At the default 8,192-token chunks these grow about 4×; `rel_logits` alone is about 0.5 GB.
- **Verdict: plausible, not proven.** It is plausible with `--max-num-batched-tokens 2048` (which also keeps
  the router GEMM off the TMA path, row 2b-4), a small `--max-num-seqs`, and capped CUDA-graph capture sizes.
  The unknowns are the size of Humming's workspace and the profiler's activation peak. The memory-fit work
  should read "Available KV cache memory" and "Maximum concurrency" from the startup log on the pod.
- **DSpark weights are not wanted.** Check that the 0.6 GB `mtp` shard is not loaded without
  `--speculative-config`; this was not traced here.

## 6. Open questions to settle on the pod

1. Does `launch_pdl=True` on its own (without `gdc_wait` in the kernel) launch cleanly on SM89, or must every
   launch site be gated? Gating them all is the safe choice.
2. The Humming SM89 heuristics: do the picks fit 99 KiB, and how fast are they? ([#4](https://github.com/suyashpatel98/vllm/issues/4))
3. Does `tl.range(warp_specialize=True)` compile on sm_89? This is moot if SM89 skips the TMA branch.
4. Measured memory (section 5), and whether the vision encoder's CUDA graphs need the padding-invariant
   linear.

## 7. For the test plan

The module-level gates in `tests/models/inkling2` exclude SM89 today:

- `test_model.py:23-25`, `test_moe.py:13-15` and `test_dspark.py:15-17` accept only `family(100)`,
  `is_device_capability(90)` or `family(120)`.
- `test_vision.py:20-22` requires a major version of 9, 10 or 12.
- `test_attention.py:17-20` has `requires_sm90` (majors 9-12).

These gates need to admit SM89 before the existing tests can guard the new dispatch.
