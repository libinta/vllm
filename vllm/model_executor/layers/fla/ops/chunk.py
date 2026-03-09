# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import os
import warnings

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd

_VLLM_Q35_TC_OP = os.getenv("VLLM_Q35_TC_OP", "0") == "1"

def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    # [B,T,...] -> [T_total,...]
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


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    BT = 64
    # 1) same precompute as before
    g = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    if _VLLM_Q35_TC_OP:
        # 2) run ONLY torch H-stage
        B, T, Hg, Kd = k.shape
        _, _, H, Vd = u.shape
        device = k.device
        IS_VARLEN = cu_seqlens is not None

        # token-major inputs for torch H kernel
        k_tm = _flatten_bt(k)   # [T_total, Hg, K]
        w_tm = _flatten_bt(w)   # [T_total, H,  K]
        v_tm = _flatten_bt(u)   # [T_total, H,  V]
        g_tm = _flatten_bt(g)   # [T_total, H]
        T_total = int(k_tm.shape[0])

        # chunk layout + buffers
        if IS_VARLEN:
            # op convention: varlen expects B == 1 and inputs already flattened into T_total
            if B != 1:
                raise ValueError(f"varlen path expects B=1, got B={B}")
            chunk_offsets, total_chunks = _make_chunk_offsets(cu_seqlens, BT, device)
            T_fixed = 0
            # IMPORTANT: chunk_fwd_o expects h in 4D for varlen (matches typical triton wrapper)
            h_shape = (total_chunks, H, Vd, Kd)
        else:
            NT = (T + BT - 1) // BT
            total_chunks = B * NT
            chunk_offsets = None
            T_fixed = T
            # non-varlen: chunk_fwd_o expects h in 5D [B, NT, H, V, K]
            h_shape = (total_chunks, H, Vd, Kd)

        h_dtype = k.dtype  
        h_torch = torch.empty(h_shape, device=device, dtype=h_dtype)
        v_new_torch = torch.empty((T_total, H, Vd), device=device, dtype=u.dtype)

        ht_torch = None
        if output_final_state:
            ht_torch = torch.empty((B, H, Vd, Kd), device=device,
                                dtype=h_dtype)

        _, _, ht_out = chunk_gated_delta_rule_h_kernel_torch(
            k=k_tm,
            v=v_tm,
            w=w_tm,
            g=g_tm,
            gk=None,
            h=h_torch,
            h0=initial_state,
            ht=ht_torch,
            cu_seqlens=cu_seqlens if IS_VARLEN else None,
            chunk_offsets=chunk_offsets if IS_VARLEN else None,
            T=T_fixed,
            BT=BT,
            H=H,
            Hg=Hg,
            K=Kd,
            V=Vd,
            USE_G=True,
            USE_GK=False,
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=bool(output_final_state),
            SAVE_NEW_VALUE=True,
            IS_VARLEN=bool(IS_VARLEN),
            v_new=v_new_torch,
        )

        # 3) reshape torch outputs into what chunk_fwd_o expects
        if IS_VARLEN:
            # v_new: [T_total,H,V] -> [1,T_total,H,V]
            v_new_use = v_new_torch.view(1, T_total, H, Vd)
            # h: keep 4D [total_chunks,H,V,K]
            h_use = h_torch
        else:
            NT = (T + BT - 1) // BT
            v_new_use = v_new_torch.view(B, T, H, Vd)          # [B,T,H,V]
            h_use = h_torch.view(B, NT, H, Vd, Kd)             # [B,NT,H,V,K]

        final_state_use = ht_out if output_final_state else None

        # 4) finish O-stage using torch-produced v_new/h
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new_use,
            h=h_use,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        if SUPPRESS_LEVEL < 3:
            return g, o, A, final_state_use, None, None, None
        else:
            return g, o, A, final_state_use, w, h_use, v_new_use
    else:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        if SUPPRESS_LEVEL < 3:
            return g, o, A, final_state, None, None, None
        elif SUPPRESS_LEVEL >= 3:
            return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

def chunk_gated_delta_rule_h_kernel_torch(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    h: torch.Tensor,
    h0: torch.Tensor | None,
    ht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    T: int,
    BT: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    USE_G: bool,
    USE_GK: bool,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
    SAVE_NEW_VALUE: bool,
    IS_VARLEN: bool,
    v_new: torch.Tensor | None = None,
):
    # ---- flatten [B,T,...] -> [T_total,...] if needed
    if k.dim() == 4:
        k = k.reshape(-1, k.shape[2], k.shape[3])  # [T_total,Hg,K]
    if v.dim() == 4:
        v = v.reshape(-1, v.shape[2], v.shape[3])  # [T_total,H,V]
    if w.dim() == 4:
        w = w.reshape(-1, w.shape[2], w.shape[3])  # [T_total,H,K]
    if USE_G and g is not None and g.dim() == 3:
        g = g.reshape(-1, g.shape[2])              # [T_total,H]
    if USE_GK and gk is not None and gk.dim() == 4:
        gk = gk.reshape(-1, gk.shape[2], gk.shape[3])  # [T_total,H,K]

    device = v.device
    T_total = v.shape[0]

    assert v.shape == (T_total, H, V), (v.shape, T_total, H, V)
    assert w.shape == (T_total, H, K), (w.shape, T_total, H, K)
    assert k.shape == (T_total, Hg, K), (k.shape, T_total, Hg, K)
    if USE_G:
        assert g is not None and g.shape == (T_total, H), (None if g is None else g.shape)
    if USE_GK:
        assert gk is not None and gk.shape == (T_total, H, K), (None if gk is None else gk.shape)
    if SAVE_NEW_VALUE:
        assert v_new is not None and v_new.shape == (T_total, H, V)

    # Infer N
    # -----------------------------
    # Infer N + per-sequence helpers
    # -----------------------------
    if IS_VARLEN:
        assert cu_seqlens is not None and chunk_offsets is not None

        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            cu_seqlens = cu_seqlens.to(torch.int64)
        if chunk_offsets.dtype not in (torch.int32, torch.int64):
            chunk_offsets = chunk_offsets.to(torch.int64)

        N = int(cu_seqlens.numel() - 1)

        def _seq_bounds(n: int):
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            Tn = eos - bos
            NT = (Tn + BT - 1) // BT
            boh = int(chunk_offsets[n].item())
            return bos, eos, Tn, NT, boh

    else:
        assert T > 0
        N = int(T_total // T)
        assert N * T == T_total

        def _seq_bounds(n: int):
            bos = n * T
            eos = bos + T
            Tn = T
            NT = (Tn + BT - 1) // BT
            boh = n * NT
            return bos, eos, Tn, NT, boh


    # -----------------------------
    # state helpers
    # -----------------------------
    def _get_state(buf, n: int, ih: int):
        if buf is None:
            return None
        if buf.dim() == 3:  # [N*H, V, K]
            return buf[n * H + ih]
        if buf.dim() == 4:  # [N, H, V, K]
            return buf[n, ih]
        raise ValueError(f"Unexpected state tensor rank: {buf.dim()}")


    def _set_state(buf, n: int, ih: int, val_fp32: torch.Tensor):
        if buf is None:
            return
        out = val_fp32.to(buf.dtype)  # cast exactly once
        if buf.dim() == 3:
            buf[n * H + ih].copy_(out)
        elif buf.dim() == 4:
            buf[n, ih].copy_(out)
        else:
            raise ValueError(f"Unexpected state tensor rank: {buf.dim()}")


    # -----------------------------
    # GQA mapping + constants
    # -----------------------------
    assert H % Hg == 0
    heads_per_group = H // Hg
    EXP_CLAMP = 80.0  # fp32 exp(80) ~ 5.5e34; avoid inf


    for n in range(N):
        bos, eos, Tn, NT, boh = _seq_bounds(n)

        for ih in range(H):
            hg = ih // heads_per_group

            # init state
            if USE_INITIAL_STATE:
                init = _get_state(h0, n, ih)
                if init is None:
                    h_state = torch.zeros((V, K), device=device, dtype=torch.float32)
                else:
                    h_state = init.to(torch.float32)
            else:
                h_state = torch.zeros((V, K), device=device, dtype=torch.float32)

            for it in range(NT):
                # chunk bounds (sequence-local)
                t0 = it * BT
                t1 = min((it + 1) * BT, Tn)

                abs_t0 = bos + t0
                abs_t1 = bos + t1
                Bt = abs_t1 - abs_t0
                if Bt <= 0:
                    continue

                # store state before chunk
                h[boh + it, ih].copy_(h_state.to(h.dtype))

                # residual: b_v = v - w @ h
                w_chunk = w[abs_t0:abs_t1, ih, :].to(torch.float32)  # [Bt, K]
                v_chunk = v[abs_t0:abs_t1, ih, :].to(torch.float32)  # [Bt, V]
                b_v = v_chunk - (w_chunk @ h_state.T)                # [Bt, V]

                if SAVE_NEW_VALUE and (v_new is not None):
                    v_new[abs_t0:abs_t1, ih, :].copy_(b_v.to(v_new.dtype))

                # ---- IMPORTANT: Triton uses PER-CHUNK last index ----
                chunk_last_idx = abs_t1 - 1

                # USE_G: b_v *= exp(g_last - g_t), and h_state *= exp(g_last)
                if USE_G:
                    g_last = g[chunk_last_idx, ih].to(torch.float32)          # scalar
                    g_t = g[abs_t0:abs_t1, ih].to(torch.float32)              # [Bt]
                    b_v = b_v * torch.exp(
                        torch.clamp(g_last - g_t, -EXP_CLAMP, EXP_CLAMP)
                    ).unsqueeze(-1)
                    h_state = h_state * torch.exp(torch.clamp(g_last, -EXP_CLAMP, EXP_CLAMP))

                # USE_GK: h_state *= exp(gk_last) elementwise on K
                if USE_GK:
                    gk_last = gk[chunk_last_idx, ih, :].to(torch.float32)     # [K]
                    h_state = h_state * torch.exp(
                        torch.clamp(gk_last, -EXP_CLAMP, EXP_CLAMP)
                    ).unsqueeze(0)

                # update: h += b_v^T @ k_chunk
                k_chunk = k[abs_t0:abs_t1, hg, :].to(torch.float32)           # [Bt, K]
                h_state = h_state + (b_v.transpose(0, 1) @ k_chunk)           # [V, K]

            # epilogue: Triton stores RAW accumulator
            if STORE_FINAL_STATE:
                _set_state(ht, n, ih, h_state)


    return h, (v_new if SAVE_NEW_VALUE else None), (ht if STORE_FINAL_STATE else None)

@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
