import os
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule_h_kernel_torch
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    return default if v is None else int(v)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, None)
    return default if v is None else v


def _max_abs_rel(a: torch.Tensor, b: torch.Tensor):
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    abs_err = (a32 - b32).abs().max().item()
    denom = b32.abs().max().item() + 1e-12
    rel_err = abs_err / denom
    return abs_err, rel_err


def _worst_index(a: torch.Tensor, b: torch.Tensor):
    diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
    mx = diff.max()
    nz = (diff == mx).nonzero(as_tuple=False)
    return mx.item(), (nz[0].tolist() if nz.numel() else [])


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 2:
        return x.reshape(-1, *x.shape[2:])
    return x


def _make_chunk_offsets(cu_seqlens: torch.Tensor, BT: int, device: torch.device):
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        cu_seqlens = cu_seqlens.to(torch.int64)

    N = int(cu_seqlens.numel() - 1)
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)  # [N]
    chunks_per_seq = (seqlens + (BT - 1)) // BT                   # [N]

    chunk_offsets = torch.empty((N,), device=device, dtype=torch.int64)
    if N > 0:
        chunk_offsets[0] = 0
        if N > 1:
            chunk_offsets[1:] = torch.cumsum(chunks_per_seq[:-1], dim=0)

    total_chunks = int(chunks_per_seq.sum().item())
    return chunk_offsets, total_chunks


def _build_inputs(B: int, T: int, H: int, Hg: int, K: int, V: int, dtype, device):
    torch.manual_seed(_env_int("SEED", 0))

    k_fp32 = torch.randn(B, T, Hg, K, device=device, dtype=torch.float32)
    k_fp32 = F.normalize(k_fp32, p=2, dim=-1)
    k = k_fp32.to(dtype)

    v = (0.5 * torch.randn(B, T, H, V, device=device, dtype=torch.float32)).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).sigmoid().to(dtype)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=device, dtype=torch.float32)
    ).to(dtype)

    # IMPORTANT: do NOT create h0 here; varlen needs [N,H,V,K], not [B,...]
    return k, v, beta, g


def _precompute_w_u(k, v, beta, g, BT: int, cu_seqlens=None):
    g_cumsum = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g_cumsum, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g_cumsum, cu_seqlens=cu_seqlens)
    return g_cumsum, A, w, u


@pytest.mark.parametrize("store_final", [False, True])
def test_torch_h_kernel_matches_triton_h_stage(store_final):
    device = _env_str("DEVICE", "cuda")
    assert torch.cuda.is_available(), "CUDA required for Triton reference path"

    dtype_s = _env_str("DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_s in ("bf16", "bfloat16") else torch.float16

    # dims
    B = _env_int("B", 2)
    T = _env_int("T", 256)
    H = _env_int("H", 8)
    Hg = _env_int("Hg", 2)
    K = _env_int("K", 64)
    V = _env_int("V", 64)
    BT = _env_int("BT", 64)

    is_varlen = _env_int("VARLEN", 0) == 1
    cu_seqlens = None
    N = None  # number of sequences (varlen) or batch (fixed)

    if is_varlen:
        B = 1
        NSEQ = _env_int("NSEQ", 4)
        MINLEN = _env_int("MINLEN", 64)
        MAXLEN = _env_int("MAXLEN", 256)

        torch.manual_seed(_env_int("SEED", 0))
        lens = torch.randint(low=MINLEN, high=MAXLEN + 1, size=(NSEQ,), device="cpu")
        T_total = int(lens.sum().item())

        cu = torch.empty((NSEQ + 1,), dtype=torch.long)
        cu[0] = 0
        cu[1:] = torch.cumsum(lens.to(torch.long), dim=0)

        cu_seqlens = cu.to(device)
        T = T_total
        N = NSEQ
    else:
        N = B

    # inputs
    k, v, beta, g = _build_inputs(B, T, H, Hg, K, V, dtype=dtype, device=device)

    # allocate initial_state with correct leading dim
    h0 = None
    if _env_int("INIT", 1) == 1:
        h0 = (0.5 * torch.randn(N, H, V, K, device=device, dtype=torch.float32)).to(dtype)

    g_cumsum, _A, w, u = _precompute_w_u(k, v, beta, g, BT=BT, cu_seqlens=cu_seqlens)

    # ---- Triton reference ----
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=h0,
        output_final_state=store_final,
        cu_seqlens=cu_seqlens,
    )

    # token-major inputs for torch kernel
    k_tm = _flatten_bt(k)
    w_tm = _flatten_bt(w)
    v_tm = _flatten_bt(u)
    g_tm = _flatten_bt(g_cumsum)
    T_total = int(k_tm.shape[0])

    if is_varlen:
        chunk_offsets, total_chunks = _make_chunk_offsets(cu_seqlens, BT, k.device)
        T_fixed = 0
    else:
        NT = (T + BT - 1) // BT
        total_chunks = B * NT
        chunk_offsets = None
        T_fixed = T

    h_torch = torch.empty((total_chunks, H, V, K), device=device, dtype=h_ref.dtype)
    v_new_torch = torch.empty((T_total, H, V), device=device, dtype=u.dtype)

    ht_torch = None
    if store_final:
        ht_torch = torch.empty((N, H, V, K), device=device, dtype=h_ref.dtype)

    _, _vnew_out, ht_out = chunk_gated_delta_rule_h_kernel_torch(
        k=k_tm,
        v=v_tm,
        w=w_tm,
        g=g_tm,
        gk=None,
        h=h_torch,
        h0=h0,
        ht=ht_torch,
        cu_seqlens=cu_seqlens if is_varlen else None,
        chunk_offsets=chunk_offsets if is_varlen else None,
        T=T_fixed,
        BT=BT,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=(h0 is not None),
        STORE_FINAL_STATE=bool(store_final),
        SAVE_NEW_VALUE=True,
        IS_VARLEN=bool(is_varlen),
        v_new=v_new_torch,
    )

    # normalize reference layouts
    h_ref_cmp = h_ref.reshape(-1, H, V, K) if h_ref.ndim == 5 else h_ref
    v_new_ref_cmp = v_new_ref.reshape(-1, H, V) if v_new_ref.ndim == 4 else v_new_ref

    h_abs, h_rel = _max_abs_rel(h_torch, h_ref_cmp)
    v_abs, v_rel = _max_abs_rel(v_new_torch, v_new_ref_cmp)

    print(
        f"[H-stage] varlen={is_varlen} store_final={store_final} "
        f"| h abs={h_abs:.3e} rel={h_rel:.3e} "
        f"| v_new abs={v_abs:.3e} rel={v_rel:.3e}",
        flush=True,
    )

    if store_final:
        ht_abs, ht_rel = _max_abs_rel(ht_out, ht_ref)
        print(f"[H-stage] ht abs={ht_abs:.3e} rel={ht_rel:.3e}", flush=True)

    # tolerances (slightly loose for bf16; tune via env if you want)
    RTOL = float(os.getenv("RTOL", "1e-2"))
    ATOL = float(os.getenv("ATOL", "1e-2"))
    assert h_rel < RTOL or h_abs < ATOL
    assert v_rel < RTOL or v_abs < ATOL
    if store_final:
        assert ht_rel < RTOL or ht_abs < ATOL