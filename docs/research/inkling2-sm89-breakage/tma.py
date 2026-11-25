import torch, triton
from vllm.models.inkling2.nvidia.ops import linear as L
print("TMA_MIN_TOKENS", L.TMA_MIN_TOKENS)
calls = []
orig = L._k_group_tma_kernel.run
def spy(*a, **k):
    calls.append(k.get("num_stages")); return orig(*a, **k)
L._k_group_tma_kernel.run = spy
w = torch.randn(128, 2304, device="cuda", dtype=torch.bfloat16)
for t in (1024, 2048, 4096, 8192):
    x = torch.randn(t, 2304, device="cuda", dtype=torch.bfloat16)
    try:
        p = L.k_group_partials(x, w)
        ref = x.float() @ w.float().T
        err = (p.sum(0).float() - ref).abs().max().item()
        print(t, "shape", tuple(p.shape), "max err", err, "tma launches so far", calls)
    except Exception as e:
        print(t, "FAIL", type(e).__name__, str(e)[:300])
k = L._k_group_tma_kernel
for dev_cache in getattr(k, "device_caches", {}).values():
    for key, kern in dev_cache[0].items():
        print("compiled TMA kernel shared mem:", kern.metadata.shared, "num_stages", kern.metadata.num_stages)
