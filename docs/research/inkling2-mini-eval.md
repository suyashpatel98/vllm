# Inkling 2 Mini: published accuracy and a budget-sized eval for one RTX 4090

Research for [#6](https://github.com/suyashpatel98/vllm/issues/6) (map [#1](https://github.com/suyashpatel98/vllm/issues/1)).
Gathered 2026-11-25. HF checkpoint revision at that date: `c9d028302c` (the commits after `ed40e7da62`
are titled README and evaluation-table updates).

## TL;DR

- **The model card publishes no number a 4090 can reproduce in ~1 GPU-hour.** Every published score is an
  agentic, multi-hour or judge-based benchmark (SWE-bench, Terminal-Bench, BrowseComp, HLE, MMMU-Pro, ...),
  and the card gives no reasoning effort, sampling or token-cap settings. There is **no GSM8K** on the card.
- **vLLM PRs do publish GSM8K for this exact checkpoint on supported GPUs**, with exact commands. Those are the
  usable references.
- **Recommendation (primary):** the in-repo `tests/evals/gsm8k/gsm8k_eval.py`, completions mode, 5-shot, 256-token
  cap, greedy, all 1,319 questions. Reference: **84.69% (1117/1319) on one H200**, same script and flags
  ([vllm#69830](https://github.com/vllm-project/vllm/pull/69830)). About 321K output tokens: ~12-18 min on a 4090
  at 300-450 tok/s aggregate. Pass if **>= 82.0% (1081/1319)**, i.e. within 2.8 points.
- **Recommendation (second, if time):** lm-eval `gsm8k` through `/v1/chat/completions` with the reasoning parser,
  `reasoning_effort=low`, `max_gen_toks=1024`, `--limit 250`. Reference: **0.952 flexible / 0.904 strict on DGX
  Spark** ([vllm#69802](https://github.com/vllm-project/vllm/pull/69802)). Pass if flexible >= 0.914 (229/250) and
  strict >= 0.851 (213/250).
- **Reasoning can be lowered, not disabled.** `reasoning_effort` (`none`=0.0, `minimal`=0.1, `low`=0.2,
  `medium`=0.7, `high`=0.9 default, `xhigh`/`max`=0.99) becomes a "Thinking effort level" system message. Even
  effort 0 still emits some reasoning.
- **No published 4090 accuracy number on any subset.** The closest are SM80/SM86 runs in
  [vllm#69165](https://github.com/vllm-project/vllm/pull/69165), plus a 4090 run of unit tests only.
- **Map-level flag:** [vllm#69165](https://github.com/vllm-project/vllm/pull/69165) (open) already makes Inkling 2
  run on SM80/SM86/SM89. It has been run on an RTX 4090 and an RTX 6000 Ada (SM89). See the last section.

## 1. What is published

### Model card and `.eval_results/` (HF)

Source: [README.md §5](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and
[`.eval_results/Inkling-2-Mini.yaml`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/.eval_results/Inkling-2-Mini.yaml),
the only file in that directory.

`.eval_results/Inkling-2-Mini.yaml` (machine-readable subset, all dated 2026-11-23):

| Benchmark | Score | Harness note in the file |
|---|---|---|
| DeepSWE v1.1 | 48.2 | mini-swe-agent v2.4.6 |
| Terminal-Bench v4.0 | 5.6 | mini-swe-agent v2.4.6 |
| Terminal-Bench 2.1 | 71.6 | pi v0.83.0 |
| SWE-Bench Pro v2 (Hard) | 77.9 | mini-swe-agent v2.4.6 |
| SWE-bench Multilingual | 73.7 | bash-only scaffold |
| SWE-bench Verified | 79.2 | bash-only scaffold |
| MMMU-Pro standard (10 options) | 74.9 | (none) |
| ScreenSpot-Pro | 76.5 / 87.7 with Python tool | (none) |

The README table also lists: AA Index 20.1, Vals Index 46.5%, FrontierCode 15.1%, Vibe Code Bench 77.9%,
PMPP-Hard 455.82, BrowseComp 79.0%, DeepSearchQA 85.3%, MCP Atlas 82.7%, AA-Briefcase 1263, GDPVal-AA 1319,
JobBench 49.6%, AutomationBench-AA 41.8%, EnterpriseOps-Gym 42.0%, Vals Finance 50.1%, AA-AnalystAgent 46.3%,
Diligence Bench 48.6%, GDP.pdf 16.8%, PerceptionBench 57.5%, OmniDocBench 92.8%, Chartography 22.1/72.5%,
ZeroBench 17.0/34.0%, MuirBench 77.5%, BVB 68.4%, HLE-Diamond 15.2%, CritPt 31.4/17.0%, Apex 2025 34.4%,
ARC-AGI 84.5/49.3%, LongBench v2 66.9%, GraphWalks BFS 256K 69.7%, MRCR v2 8-needle 46.4%, AA-Omniscience,
ComplexConstraints 33.4%, and safety evals.

The card gives no sampling temperature, reasoning effort or output-token cap for any of them. It says
"Setting reasoning effort to its minimum does not fully disable reasoning" and warns of "long reasoning loops"
(README §7). `generation_config.json` sets `temperature: 1.0`, `do_sample: true`.

**Cheapest card-anchored option: MMMU-Pro standard-10 (74.9%).** It has 1,730 questions, exercises the image
path, and Thinking Machines' [inkling-verify](https://github.com/thinking-machines-lab/inkling-verify)
(`run_mmmu_pro.py`, `--limit N`) is the likely harness. But the effort and token cap behind 74.9% are unpublished.
inkling-verify defaults to the server's default effort (0.9 for Inkling 2) and a 262,144-token output cap. Its
README example adds `--with-tools` (Podman sandbox) for standard-10, while the eval file marks only ScreenSpot as
"with Python tool". A 100-question slice has a 95% interval of about ±8.7 points, so it can only catch gross
failures. Use it as an optional image smoke check, not the accuracy bar.

### vLLM blog (2026-11-20)

[vllm.ai/blog/2026-11-20-inkling2-mini](https://vllm.ai/blog/2026-11-20-inkling2-mini) gives **no scores**:
"End to end, we tested accuracy on needle-in-a-haystack up to 1M tokens, MMMU-Pro and HLE with inkling-verify,
and on GSM8K, with DSpark enabled." It also says vLLM's hidden states match the reference to within the
reference's own FA4-vs-PyTorch spread, and that the MoE outputs match bitwise.

### This repo

- `tests/evals/gsm8k/configs/` has **no Inkling config** (`grep -ri inkling tests/evals` is empty).
- `recipes/models/thinkingmachines/Inkling-2-Mini.yaml` lists hardware (GB200, B300, GB300, H100, H200,
  DGX Spark) and a throughput headline, with **no accuracy numbers**.

### GSM8K numbers for this checkpoint in upstream vLLM PRs

| Source | HW | Harness / settings | n | Result |
|---|---|---|---|---|
| [#69028](https://github.com/vllm-project/vllm/pull/69028) (adds the model) | GB200 TP1 | "temperature 0, DSpark"; harness and effort not stated | 1319 | 96.8% (96.7% no spec) |
| same | GB200 TP1 | same | first 200 | 97.0% |
| [#69830](https://github.com/vllm-project/vllm/pull/69830) ([commands](https://github.com/mglyn/vllm/pull/14#issuecomment-7155261135)) | H200 (SM90) TP1 | `tests/evals/gsm8k/gsm8k_eval.py` completions, 5-shot, 256 tok, temp 0 | 1319 | **84.69% (1117)** base; 85.44% PR |
| [#69767](https://github.com/vllm-project/vllm/pull/69767) | H200 TP1 | lm-eval chat, 5-shot, temp 0, `max_gen_toks=8192`, DSpark, default effort | 1319 | flex 0.937/0.933, strict 0.917/0.911 (2 runs) |
| [#69802](https://github.com/vllm-project/vllm/pull/69802) | DGX Spark (SM121) | lm-eval chat, multiturn 5-shot, `reasoning_effort=low`, `max_gen_toks=1024`, DSpark | first 250 | **flex 0.952 / strict 0.904** (main) |
| [#69452](https://github.com/vllm-project/vllm/pull/69452) | H200 | test[:500], temp 0, harness not stated | 500 | 96.4-97.4% |
| [#69104](https://github.com/vllm-project/vllm/pull/69104) | B200 | custom "Answer: <number>" prompt, greedy | 1319 | 96.29% |
| [#69165](https://github.com/vllm-project/vllm/pull/69165) | 2x 3090 (SM86) TP2, **unmerged PR** | lm-eval `local-completions` 5-shot | first 250 | strict 0.832 / flex 0.816 |
| same ([A100 comment](https://github.com/vllm-project/vllm/pull/69165#issuecomment-7125498226)) | A100 (SM80), **unmerged PR** | same | 250 / 1319 | strict 0.836 / 0.8294; flex 0.816 / 0.7953 |

Completions-mode scores (~83-85%) are lower than chat-mode scores (~93-97%) because the raw few-shot prompt
bypasses the chat renderer and reasoning. The two are not comparable with each other.

## 2. Reasoning effort and thinking control

- The HF `chat_template.jinja` maps `reasoning_effort` strings to `none` 0.0, `minimal` 0.1, `low` 0.2,
  `medium` 0.7, `high` 0.9, `xhigh`/`max` 0.99, and accepts a number in [0, 0.99]. The default is **0.9**. It emits
  `<|message_system|><|content_text|>Thinking effort level: X<|end_message|>` before the first non-system message.
- vLLM does not use that Jinja file. `ModelConfig` sets `tokenizer_mode = "inkling"` for
  `Inkling2LoopForConditionalGeneration` (`vllm/config/model.py`), which selects the native `InklingRenderer`
  (`vllm/renderers/inkling.py`). It applies the same name map (`_NAMED_REASONING_EFFORT`) and default
  (`DEFAULT_REASONING_EFFORT = 0.9` in `inkling_encoding.py`), and also accepts numeric strings for Inkling 2.
- **How to set it:** use the top-level request field `"reasoning_effort": "low"`, or
  `"chat_template_kwargs": {"reasoning_effort": 0.2}`. `ChatCompletionRequest.build_chat_params`
  (`vllm/entrypoints/openai/chat_completion/protocol.py`) copies the top-level field into `chat_template_kwargs`,
  and the top-level value wins. The renderer does not read `enable_thinking`.
- **It cannot be fully disabled.** The model card says so (§7), and a
  [comment on #69165](https://github.com/vllm-project/vllm/pull/69165) reports that `reasoning_effort="none"`
  still produced reasoning that used up the output-token budget.
- **The sampling default is temperature 1.0** (`generation_config.json`, which vLLM applies with
  `--generation-config auto`). Always send `temperature: 0` explicitly. lm-eval does this by default (the gsm8k
  task yaml and the `LocalChatCompletion._create_payload` default). `gsm8k_eval.py` sends it in completions mode,
  but in chat mode only with `--temperature 0`.

## 3. Recommended eval

### Budget arithmetic

- Throughput assumption: the SM89 PR measured **104 tok/s single stream and 252 tok/s at c=4** on an RTX 6000
  Ada (same AD102 die as the 4090), and 475-728 tok/s at c=16 on SM86/SM80
  ([#69165](https://github.com/vllm-project/vllm/pull/69165)). Assume 300-450 tok/s aggregate at c=16 for a
  working 4090 path. If the fork's dequant path is ~3x slower, plan for about 150 tok/s.
- KV cache is the 4090's limit: ~19.5 GB of weights leave ~2 GB, at about 69 KB per token (recipe), so roughly
  25-30K tokens. With `--max-model-len 2048`, 16 concurrent GSM8K requests (~1.2K tokens each) fit.
  `--language-model-only` skips the vision tower to free memory for this text-only eval.

### A. Primary: in-repo GSM8K, completions, full test set

Reference: H200 TP1, vLLM `7cd994e2d9`, checkpoint `c9d028302c`: **1117/1319 = 84.69%**, 320,974 output
tokens, 1 invalid answer, 1221/1319 responses hit the 256-token cap
([commands and output](https://github.com/mglyn/vllm/pull/14#issuecomment-7155261135)). H200 also runs NVFP4
experts as Humming W4A16 (weight-only dequant, [#69767](https://github.com/vllm-project/vllm/pull/69767)), which
is the closest supported analogue to an SM89 dequant path.

```bash
vllm serve thinkingmachines/Inkling-2-Mini --port 8000 \
  --max-model-len 2048 --max-num-seqs 16 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 --language-model-only
# reference also used --disable-prefix-caching; leaving prefix caching on saves ~4 min of
# 5-shot prefill and only changes batch composition (noise-level)

.venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port 8000 --num-questions 1319 \
  --num-shots 5 --max-tokens 256 --max-concurrency 16 --temperature 0 \
  --seed 42 --request-timeout-seconds 1800 --save-results gsm8k-4090.json
```

- Cost: ~321K output tokens, which is ~12-18 min at 300-450 tok/s. Add ~1.2M prompt tokens (~4 min at ~5K tok/s
  without prefix caching). If the tqdm ETA after the first minute is over ~30 min, stop and run B instead.
- **Tolerance:** pass if accuracy >= **82.0% (1081/1319)**, i.e. |Δ| <= 2.8 points. This is 2 SD of the
  difference between two independent binomial runs at p=0.847, n=1319 (SE 0.99 points per run, 1.40 for the
  difference). Correct implementations agree more closely than this: the two H200 arms above differed by 0.75
  points (39 and 29 flipped answers, paired SD ≈ 0.63 points). So a drop of more than ~1.5 points deserves a look,
  and more than 2.8 points means a bug. The invalid rate should stay near 0.1%.

### B. Second: lm-eval chat GSM8K through the reasoning parser, first 250

This exercises the path users run (native renderer, `--reasoning-parser inkling`, effort control). The reference
is DGX Spark (a recipe-verified platform), vLLM `main` arm of
[#69802](https://github.com/vllm-project/vllm/pull/69802), lm-eval 0.4.13: **flexible 0.952 ± 0.0135, strict
0.904 ± 0.0187**. That run had DSpark on. Spec decoding preserves the output distribution: #69028 measured 96.8%
with it and 96.7% without.

```bash
vllm serve thinkingmachines/Inkling-2-Mini --port 8000 --reasoning-parser inkling \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.92 --language-model-only

uv pip install "lm-eval[api]==0.4.13"
.venv/bin/lm_eval --model local-chat-completions --tasks gsm8k --apply_chat_template --fewshot_as_multiturn \
  --model_args model=thinkingmachines/Inkling-2-Mini,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,tokenized_requests=False \
  --gen_kwargs reasoning_effort=low,max_gen_toks=1024 --limit 250 --seed 0
```

- Same subset as the reference: lm-eval `--limit 250` takes the first 250 test docs. Same settings, except
  concurrency (8 vs 2) and no DSpark.
- Cost: at most 256K output tokens at the 1024 cap. At low effort the real total is likely well under half that,
  so about 10-15 min worst case.
- **Tolerance** (2 SD of an unpaired difference at n=250): flexible >= **0.914 (229/250)**, strict >= **0.851
  (213/250)**. The same 250 questions moved 2 points between two arms of #69802.
- Caveat: lm-eval sends `stop: ["Question:", ...]`, and vLLM applies stop strings to reasoning text as well.
  Low effort keeps this rare, and the reference ran under the same condition.

### Optional: image smoke (not a statistical check)

MMMU-Pro standard-10 through inkling-verify with `--limit 50 --max-output-tokens 16384`. Compare loosely with the
card's 74.9% (full set) and #69028's 78.0% (first 150, GB200 TP4). At n=50 the 95% interval is about ±12 points.
Run it only if budget remains: default effort 0.9 makes the outputs long.

### Budget fit

Model download (~20 GB) plus server start: ~8-10 min. A: ~15-20 min. B: ~10-15 min. That is ~35-45 GPU-minutes,
about $0.25-0.55 at $0.35-0.70/hr, which fits the ~1 GPU-hour target with room for one retry.

## 4. Is there a reference on the same subset from a supported GPU?

- **Yes for both recommended evals**, on GPUs the recipe lists as verified: A matches H200 (full 1319, same script
  and flags, same checkpoint revision), and B matches DGX Spark (first 250, same lm-eval command).
- **No published accuracy number exists on an RTX 4090 or any SM89 GPU**, on any subset. #69165's 4090 report
  covers unit tests only (`test_core_ops.py` 74 passed, `test_moe.py` 68 passed). The RTX 6000 Ada (SM89) report
  covers the test suite and throughput, with no accuracy. The SM8x GSM8K numbers in that PR (3090, A100) come from
  lm-eval `local-completions` (strict ~0.83), a different harness from A and B, on an unmerged branch.

## Map-level note: possible duplicate work

[vllm-project/vllm#69165](https://github.com/vllm-project/vllm/pull/69165) ("Inkling 2: run on Ampere and Ada
(SM80/SM86/SM89)", open, by Ar4ikov) is the same work the parent map plans. It gates out SM90-only features and
reuses the paths Hopper already takes. On Hopper those are Humming W4A16 expert GEMMs (per #69767), and that PR's
4090 run passes `test_humming_experts_are_deterministic`. On SM89 it uses a software e4m3 path. It has been run by others on an RTX 4090
(unit tests) and an RTX 6000 Ada (TP1 serving, 104 tok/s, DSpark 157 tok/s). AGENTS.md's duplicate-work rule
applies. The map should decide whether to build on that branch instead of writing a new SM89 path. It also
suggests the "30-100 tok/s" assumption is pessimistic for a Humming-based path.
