# Which kernels can run Inkling 2 Mini's NVFP4/MXFP8 expert GEMMs on SM89

Research for suyashpatel98/vllm#4 (map: #1). Goal: run the routed-expert GEMMs
of `thinkingmachines/Inkling-2-Mini` on an RTX 4090 (SM89) with BF16
activations, using weight-only dequantization.

Sources: this repo at `7b7af30196`, cited as `path:line`; the
`humming-kernels==0.1.20` sdist from PyPI, which matches GitHub tag
[`vllm-project/humming@v0.1.20`](https://github.com/vllm-project/humming/tree/v0.1.20)
(checked byte-for-byte on `humming/tune/sm8x.py`), cited as `humming:path:line`;
Triton `main` on GitHub; and the Mini `config.json` on Hugging Face. Nothing
here was run on a GPU.

## TL;DR

| Option | SM89 | NVFP4 (E4M3/16 + per-expert FP32) | MXFP8 (E8M0/32) | Glue for Inkling | Speed expectation |
| --- | --- | --- | --- | --- | --- |
| **Humming** (`HummingExperts`, the Hopper path) | **Yes**. SM89 tuner, compiled for `sm_89` at run time, tested in Humming CI on L4 (SM89) | Yes | Yes | **None in the expert code.** On any GPU that is neither SM100 nor SM12x, Inkling already runs `HummingExperts`. Only the capability gates need opening. | About the same as Marlin on a 4090 (Humming's own 4090 dense numbers). Memory-bound at decode. |
| **Marlin** (`moe_wna16_marlin_gemm`) | **Yes**. Built for 8.x, gated at ≥7.5 | Yes, with lossy handling of subnormal scales | Yes (BF16 activations only) | About 100 lines: a `MarlinExperts` class modeled on `HummingExperts` (repack, then 2 GEMMs plus Inkling's own SwiGLU) | Mature on Ada. vLLM ranks it ahead of Humming on every GPU except SM90. |
| **Triton dequant GEMM** | Yes for plain `tl.dot` kernels. PDL and TMA pieces are suspect | A fused kernel already exists (`nvfp4_emulation_moe.py`) | No fused kernel. vLLM's version dequantizes the whole tensor (too big) | NVFP4: small. MXFP8: a new kernel, or a port of `moe_sm120._mxfp8_gemm` | Not measured. Probably the slowest of the three. Last resort. |

**Recommendation: use Humming.** It is the path Inkling already takes on
non-Blackwell GPUs, and the package supports and tests SM89. The first 4090
session should include SM89 in the Inkling MoE test gates and run the existing
Humming and MoE-layer tests (see *Validation plan*). Keep Marlin as the fallback.

## Mini's shapes and byte budget

These numbers come from `config.json` on Hugging Face
(<https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/config.json>):
`hidden_size` 2304, `intermediate_size` 1152, `n_routed_experts` 128,
`num_experts_per_tok` 8, 24 layers, and a loop over layers 6–17 with
`num_passes: 3`.

- The GEMMs per layer are `w13`: N=2304 (gate and up rows interleaved), K=2304;
  and `w2`: N=2304, K=1152. Both are multiples of Humming's padding (256 for N,
  128 for K; `vllm/models/inkling2/nvidia/ops/moe_sm90.py:213-214`) and of
  Marlin's tiles.
- Each expert has 7.96 M parameters, so a layer has 1.02 B.
  - NVFP4 at 0.5625 B/param is 573 MB per layer.
  - MXFP8 at 1.03 B/param is 1.05 GB per layer.
  - All experts together come to about 15.7 GB.
- **Dequantizing to BF16 at load time is ruled out.** The experts alone would
  take 24.5 B × 2 B ≈ 49 GB.
- **Batch-1 decode reads about 1.84 GB of expert weights per token.** That is
  44 NVFP4 layer visits (6 + 12×3 + 2) × 35.8 MB, plus 4 MXFP8 visits × 65.7 MB.
  - At the 4090's 1008 GB/s peak, that takes about 1.8 ms.
  - At the ~610 GB/s that Humming and Marlin reach on a 4090 for W4A16 GEMV
    (see below), it takes about 3 ms.
  - Mini's per-expert GEMMs are small, so expect worse than either figure.
    These are estimates, not measurements.

## 1. Humming (`humming-kernels==0.1.20`)

### SM89 support: yes

- The README claims support for "every NVIDIA architecture from **SM75**". In
  its support matrix, BF16 activations need SM80+ and accept "Arbitrary signed
  FP (kBits ≤ 8)" weights, which covers e2m1 and e4m3
  ([README.md:16, :40-45](https://github.com/vllm-project/humming/blob/v0.1.20/README.md#L40-L45)).
- There is a dedicated SM89 tuner:
  - `heuristics_map[89] = Sm89Heuristics` ([humming/tune/__init__.py:26-38](https://github.com/vllm-project/humming/blob/v0.1.20/humming/tune/__init__.py#L26-L38)).
  - `Sm89Heuristics(Sm86Heuristics)` has `max_smem_size = 99 KiB` ([humming/tune/sm8x.py:49-114](https://github.com/vllm-project/humming/blob/v0.1.20/humming/tune/sm8x.py#L49-L114)).
  - The tuner keeps PDL off below SM90 (`"use_pdl": cls.sm_version >= 90`) and
    has an SM89-specific MoE raster rule ([humming/tune/base.py:414-418](https://github.com/vllm-project/humming/blob/v0.1.20/humming/tune/base.py#L414-L418)).
- Kernels are compiled at run time for the current device: `sm_version_str` is
  `89`, with the `a` suffix only for ≥90
  ([humming/jit/runtime.py:57-61](https://github.com/vllm-project/humming/blob/v0.1.20/humming/jit/runtime.py#L57-L61)).
  The wheel holds no prebuilt cubins, only the launcher `.so` files, so there is
  no build-time arch list to miss.
- The MMA path on SM89 works as follows:
  - `get_default_mma_type` returns `MMA` (`mma.sync`) outside SM9x, SM10x and
    SM12x ([humming/config/mma.py:120-127](https://github.com/vllm-project/humming/blob/v0.1.20/humming/config/mma.py#L120-L127)).
  - For BF16 activations, the MMA B operand is BF16
    (`get_mma_b_dtype` returns `a_dtype`, [mma.py:94-95](https://github.com/vllm-project/humming/blob/v0.1.20/humming/config/mma.py#L94-L95)).
    In other words, e2m1 and e4m3 weights are dequantized in software into BF16
    `mma.sync`.
  - `use_native_dequant` is false for BF16 below SM100
    ([humming/config/config.py:204-227](https://github.com/vllm-project/humming/blob/v0.1.20/humming/config/config.py#L204-L227)).
- The kernel's own checks pass for SM89 with BF16:
  - BF16 needs ≥80, and FP weights need exponent ≤ 8 and mantissa ≤ 7
    ([humming/kernel/humming.py:427-466](https://github.com/vllm-project/humming/blob/v0.1.20/humming/kernel/humming.py#L427-L466)).
  - The weight-scale group size must be ≥ 256/16 = 16, so both 16 (NVFP4) and
    32 (MXFP8) pass ([humming.py:396-413](https://github.com/vllm-project/humming/blob/v0.1.20/humming/kernel/humming.py#L396-L413)).
- **Humming's CI runs the full test suite on an L4, which is SM89**
  ([.buildkite/pipeline.yml:53-58](https://github.com/vllm-project/humming/blob/v0.1.20/.buildkite/pipeline.yml#L53-L58),
  [.buildkite/run-tests.sh](https://github.com/vllm-project/humming/blob/v0.1.20/.buildkite/run-tests.sh)).
  The MoE tests that run there on the current device include:
  - `grouped-contiguous-nvfp4`: BF16 × e2m1, E4M3 group 16, per-tensor scale 2.
  - `indexed-mxfp4`: BF16 × e2m1, E8M0 group 32, `GemmType.INDEXED`, the gemm
    type Inkling uses
    ([tests/kernels/humming/test_moe.py:95-109](https://github.com/vllm-project/humming/blob/v0.1.20/tests/kernels/humming/test_moe.py#L95-L109)).
  - `test_datatype.py` covers dense BF16 × `float8e4m3` and BF16 × `float4e2m1`.
- Humming publishes RTX 4090 benchmark results (`benchmarks/results/4090/`).

**Gap in the evidence.** No Humming test covers the exact combination
**BF16 × e4m3 weights × E8M0 group-32 × INDEXED MoE**, which is Inkling's
MXFP8 layers 20–23. Each part is tested on its own, and Hopper runs the full
combination in production (on WGMMA, not MMA). Test it first on the 4090.

### Formats

Inkling already maps both of its formats through Humming's ModelOpt schemas:

- NVFP4 becomes `b_dtype=float4e2m1`, `bs_dtype=float8e4m3`, group 16, with a
  per-expert FP32 `weight_scale_2` (TENSOR)
  ([humming/schema/modelopt.py:87-143](https://github.com/vllm-project/humming/blob/v0.1.20/humming/schema/modelopt.py#L87-L143)).
- MXFP8 becomes `float8e4m3` with `float8e8m0` scales, group 32
  ([modelopt.py:187-192](https://github.com/vllm-project/humming/blob/v0.1.20/humming/schema/modelopt.py#L187-L192)).

### Glue: none in the MoE code

- `Inkling2Experts` sets `self.sm100` and `self.sm120`
  (`vllm/models/inkling2/nvidia/moe.py:150-151`). When both are false,
  `process_weights_after_loading` builds `HummingExperts` from the row-major
  checkpoint tensors and the per-expert `amax / denominator` global scale
  (`moe.py:204-224`).
- `Inkling2MoE.forward` routes through `moe_sm90.select_experts` and then
  `HummingExperts.__call__` (`moe.py:385-395`, `moe.py:292-303`).
- `HummingExperts` (`moe_sm90.py:160-316`) is device-agnostic:
  - `get_heuristics_config(..., gemm_type=INDEXED)` selects `Sm89Heuristics` on
    a 4090.
  - Stream-K is turned off for determinism.
  - The flat (token × topk + slot) rows go through `moe_align_block_size(...,
    ignore_invalid_experts=True)`.
  - Inkling's own softcapped SwiGLU runs between the two GEMMs
    (`moe_sm90.py:112-157`). It handles the interleaved gate/up columns
    (`ops/activation.py:63-79`).

So an SM89 GPU already takes the Hopper expert path. These gates block it:

- `Inkling2QuantConfig.get_min_capability()` returns 90
  (`vllm/models/inkling2/quant_config.py:44-46`).
- `tests/models/inkling2/test_moe.py:12-17` skips the whole module below
  SM90, and `test_moe.py:305-306` runs the Humming test only on SM90.

`vllm/utils/humming.py` is a lazy import facade with no capability checks.
vLLM's generic Humming MoE experts accept compute capability ≥ 7.5
(`vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:431-437`).

### Speed

- Humming's 4090 results cover only dense layers with FP16 activations
  (N=K=8192). At that shape, Humming and Marlin are within a few percent of
  each other.

  | Kernel (M = 1 / 64 / 1024) | Humming | Marlin |
  | --- | --- | --- |
  | W4A16 int4 | 55 / 70 / 805 µs | 55 / 66 / 804 µs |
  | W8A16 (Humming int8 vs Marlin fp8) | 102 / 106 / 812 µs | 106 / 110 / 826 µs |

  At M=1 the effective bandwidth is about 610 GB/s, about 60% of peak
  (`benchmarks/results/4090/dense_n8192_k8192_g0/*.json` in the sdist).
- There are no published numbers for MoE or BF16 on a 4090.
- vLLM's kernel ranking puts Humming ahead of Marlin only on SM90
  (`vllm/model_executor/layers/quantization/utils/humming/priority.py:10-11`).
  So vLLM has not claimed that Humming beats Marlin on Ada.
- Cost: the first call compiles kernels just in time ("a few seconds per
  layer", README:20). The NVRTC/nvcc toolchain comes from the `[cu13]` extra
  already pinned in `requirements/cuda.txt:35`, and the compiled kernels are
  cached afterwards.

## 2. vLLM Marlin (`moe_wna16_marlin_gemm`)

### SM89 support: yes

- Marlin MoE kernels are built for `8.0+PTX` (loose intersection, so 8.9 is
  included) (`CMakeLists.txt:1498-1502`). The runtime checks are
  `is_fp4_marlin_supported` and `is_fp8_marlin_supported`, which require ≥7.5
  (`vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:32-33`,
  `marlin_utils_fp8.py:29-30`), and `MarlinExperts._supports_current_device`
  (`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:539-541`).
- The generated kernel set includes these configurations
  (`csrc/quantization/marlin/generate_kernels.py:86-110`):
  - **NVFP4**: `kFE2M1f` weights with `kFE4M3fn` scales, group 16, FP16 or BF16
    activations.
  - **MXFP8**: `kFE4M3fn` weights with `kFE8M0fnu` scales, group 32, BF16
    activations only.
- `MarlinExperts` accepts `kNvfp4Static` and `kMxfp8Static` weights with no
  activation quantization (`marlin_moe.py:555-574`).

### Formats and fidelity

- MXFP8: the E8M0 scales are reordered and kept as they are
  (`prepare_mxfp8_moe_layer_for_marlin`, `marlin_utils_fp8.py:457-540`).
- NVFP4: Marlin rewrites the E4M3 block scales into a special S0E5M3 format
  (`nvfp4_marlin_process_scales`, `marlin_utils_fp4.py:58-122`).
  - It drops the sign bit.
  - It scales all of a layer's experts by one shared power of two.
  - **It zeroes any scale that ends up below 2 after the ×2^7 shift**
    (`marlin_utils_fp4.py:117`), meaning E4M3 subnormal block scales.
  - This should be harmless for weights, but it is a deviation that Humming
    does not have. Check it against the checkpoint's minimum scale.
  - The per-expert global scale is pre-multiplied for BF16's exponent bias
    (`nvfp4_marlin_process_global_scale`, `marlin_utils_fp4.py:133-145`).

### Glue: moderate (about 100 lines)

Inkling does not use vLLM's `FusedMoE` layer, so `fused_marlin_moe` cannot be
reused directly:

- Its activations are vLLM's `MoEActivation`s, not Inkling's softcapped
  SwiGLU.
- It applies the routing weight in GEMM 2, while Inkling applies it inside the
  SwiGLU.

The raw op takes the same `sorted_ids` / `expert_ids` / `num_tokens_post_padded`
interface as Humming (`vllm/_custom_ops.py:2570-2596`). So a `MarlinExperts`
class built on the model of `HummingExperts` would:

1. At load time:
   - Repack each projection with `prepare_nvfp4_moe_layer_for_marlin` or
     `prepare_mxfp8_moe_layer_for_marlin`. Pass them a small namespace in place
     of `layer` (`num_experts`, `hidden_size`, `intermediate_size_per_partition`,
     `params_dtype`).
   - Note that the MXFP8 helper reads `torch.get_default_dtype()`
     (`marlin_utils_fp8.py:492`), which must be BF16 at that point.
   - Leave the interleaved gate/up row order alone. Marlin keeps row order, and
     1152 is a multiple of 64, so no padding is needed.
2. In forward:
   - Call `moe_align_block_size(ids, block_m ∈ {8,16,32,48,64}, E,
     ignore_invalid_experts=True)` (block-size limits in
     `csrc/moe/marlin_moe_wna16/ops.cu:434-437`).
   - Run GEMM 1 with `top_k=topk, mul_topk_weights=False`.
   - Apply `moe_sm90.swiglu`.
   - Run GEMM 2 with `top_k=1` on `T*topk` rows.
   - Pass rows to `combine` unchanged.

   This mirrors `_fused_marlin_moe` (`marlin_moe.py:119-195`).

### Speed

Marlin W4A16 and W8A16 are well established on Ada. On a 4090 they sit at
parity with Humming in Humming's own dense results (table above). Expect speed
similar to Humming.

## 3. Triton dequant-then-GEMM fallback

### SM89 support: yes for plain `tl.dot` kernels

BF16 `tl.dot` and FP8 (`float8e4nv`) loads work on SM89. Two kinds of Triton
code in this repo are suspect on SM89:

- **PDL** (`launch_pdl=True`, `tl.extra.cuda.gdc_wait()`). Triton's own PDL
  tutorial only enables it when `capability[0] >= 9`
  ([11-programmatic-dependent-launch.py:22-23](https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py#L22-L23)),
  and the gdc docstring says "Programmatic dependent launch is supported on SM90
  (Hopper) and beyond"
  ([third_party/nvidia/language/cuda/gdc.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/language/cuda/gdc.py)).
- **`tl.dot_scaled`**. Triton `main` decomposes scaled dots into a BF16 upcast
  on targets without native MX support, and handles both E8M0 and float (E4M3)
  scales
  ([DecomposeScaledBlocked.cpp:52-117](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.cpp#L52-L117)).
  Whether the Triton bundled with `torch==2.14.0` behaves this way on SM89
  still needs to be checked on hardware.

### Existing kernels

- **NVFP4: a fused kernel exists.** `fused_moe_nvfp4_emulation_kernel`
  (`vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py:50-253`,
  launcher at `:255`) is a classic `fused_moe`-style kernel:
  - It reads row-major `[E, N, K/2]` uint8 weights and `[E, N, K/16]` E4M3
    scales with a per-expert `[E]` global scale, and decodes e2m1 in registers
    before a BF16 `tl.dot`.
  - It is driven by `moe_align_block_size` outputs. That is Inkling's
    checkpoint layout exactly, so no repack is needed.
  - In vLLM's experts class it is used with fake-quantized (QDQ) activations
    (`:486-495`). Inkling would call the launcher directly with BF16
    activations.
- **MXFP8: no fused kernel exists.**
  - `Mxfp8EmulationTritonExperts` dequantizes the whole weight tensor to BF16
    and then runs `TritonExperts`
    (`vllm/model_executor/layers/fused_moe/experts/mxfp8_emulation_moe.py:61-149`).
    That costs about 2 GB of BF16 per layer, which does not fit if done at load
    time and is far too slow if done every step.
  - Inkling's `moe_sm120._mxfp8_gemm` (`vllm/models/inkling2/nvidia/ops/moe_sm120.py:93-156`)
    dequantizes the e4m3 weights in registers into a BF16 `tl.dot`. But it
    expects MXFP8-quantized activations in Inkling's tile layout (from
    `ops.moe.route`), and it uses PDL (`:111`, `:201`).
  - A fused MXFP8 kernel would therefore have to be written, either by cloning
    the NVFP4 emulation kernel with an e4m3 load and E8M0 scale, or by porting
    `_mxfp8_gemm` to BF16 rows with PDL removed. It is only needed for 4 layers.
- **The SM12x path as a whole** (`route` → `swiglu_quantize` → `moe_sm120.bmm`
  with `tl.dot_scaled("e2m1", "e2m1")`) is a W4A4/W8A8 path. On SM89 it would
  quantize activations only for Triton to upcast them again. It also depends on
  PDL throughout `ops/moe.py`. Not recommended.

### Speed

Not measured. Triton kernels that dequantize in registers are usually slower
than Marlin or Humming for W4A16 at small M, because Marlin-class kernels
preload a pre-permuted weight layout and pipeline dequantization with the MMA.
Treat this option as the "works, slower" fallback.

## Adjacent risks (outside this ticket)

These belong to the SM89 failure-list ticket, but they sit on the expert path:

- `moe_sm90.select_experts` calls `linear.k_group_partials`
  (`vllm/models/inkling2/nvidia/ops/linear.py:202-271`). Every launch there
  uses `launch_pdl=True` and `gdc_wait`.
- For ≥2048 tokens (`TMA_MIN_TOKENS`, `linear.py:199`), it switches to a
  tensor-descriptor kernel with 3–4 stages of 48 KiB or more.
  - Only SM12x gets the 2-stage reduction for its ~99 KiB of shared memory per
    block (`linear.py:216-238`).
  - SM89 has the same ~99 KiB limit (Humming encodes it as
    `max_smem_size = 99 * 1024` for SM86/89, `humming/tune/sm8x.py:50`), so
    large prefills will probably need the same reduction.
- `moe_sm90.swiglu` (`moe_sm90.py:112-157`) uses neither PDL nor TMA.
- Numerics: the SM100 and SM12x paths quantize activations (W4A4/W8A8, as in
  the recipe's variant description), while the Hopper/Humming path uses BF16
  activations (`moe_sm90.py:3-4`). H100 and H200 are listed as `verified` in
  `recipes/models/thinkingmachines/Inkling-2-Mini.yaml`. All three SM89 options
  here inherit the Hopper W4A16/W8A16 numerics.

## Validation plan for the first 4090 session (cheapest first)

1. Allow SM89 at the gates in `tests/models/inkling2/test_moe.py:12-17` and
   `:305-306`. Then run:
   - `test_humming_experts_are_deterministic[nvfp4|mxfp8]`
   - `test_moe_layer_matches_reference[nvfp4|mxfp8]`, which compares the
     Humming path against a dequantized torch reference.

   MXFP8 is the combination with the least evidence.
2. Microbenchmark `HummingExperts` at Mini's shapes (E=128, top-8, H=2304,
   I=1152) for T ∈ {1, 32, 512, 4096} with the `kernel-microbenchmark` skill.
   Record the JIT compile time on first call.
3. If step 1 or 2 fails, build `MarlinExperts` as described in §2.
