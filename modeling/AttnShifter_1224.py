import torch
from typing import Optional, List, Dict, Union


class AttnShifter:
    """
    Extremely optimized attention probability shifter (eager-only friendly).

    Assumptions (match your observation in eager):
    - masked (padding / causal future) key columns become exactly 0 after softmax, so we can infer pad_len
      and avoid redistributing into illegal keys without building masks.
    """

    def __init__(
        self,
        layers_heads_to_modify: Optional[Dict[str, List[int]]] = None,
        delta_ratio: float = 0.4,
        first_token_idx: Union[int, List[int]] = 0,
    ):
        self.layers_heads_to_modify = layers_heads_to_modify
        self.delta_ratio = float(delta_ratio)
        # safety: keep within [0, 1] to avoid negative probs at src
        if self.delta_ratio < 0.0:
            self.delta_ratio = 0.0
        if self.delta_ratio > 1.0:
            self.delta_ratio = 1.0

        self.first_token_idx = first_token_idx  # int preferred for fast path

        # if masked cols are not strictly 0, set to e.g. 1e-8 for bf16/fp16
        # self.zero_eps = 0.0
        self.zero_eps = 1e-8

        # avoid div-by-zero / degenerate rows
        self.min_denom = 1e-6

    def _heads_for_layer(self, layer_to_modify: int, H: int) -> List[int]:
        if self.layers_heads_to_modify is None:
            return list(range(H))
        heads = self.layers_heads_to_modify.get(str(layer_to_modify))
        if heads is None:
            heads = self.layers_heads_to_modify.get(layer_to_modify, [])
        heads = [int(h) for h in heads if 0 <= int(h) < H]
        return heads

    @torch.no_grad()
    def _infer_left_pad_lengths(self, attn_weights: torch.Tensor, probe_head: int) -> torch.Tensor:
        """
        Fast pad_len inference from attn_weights itself:
        pad_len = first index where prob > zero_eps in a probe row.
        """
        # attn_weights: [B, H, Q, K]
        B, H, Q, K = attn_weights.shape
        row = attn_weights[:, probe_head, Q - 1, :]  # [B, K]
        nz = row > self.zero_eps                      # bool [B,K]

        # argmax gives first index of max (=True) along last dim; if all False -> 0 (fix below)
        pad_len = torch.argmax(nz.to(torch.int32), dim=-1).to(torch.long)  # [B]
        pad_len = torch.where(nz.any(dim=-1), pad_len, torch.zeros_like(pad_len))
        return pad_len

    def _is_contiguous(self, heads: List[int]) -> bool:
        if not heads:
            return False
        hs = sorted(set(heads))
        return hs == list(range(hs[0], hs[-1] + 1))

    @torch.no_grad()
    def _shift_inplace_single_source_bhqk(
        self,
        attn: torch.Tensor,          # view: [B, h, Q, K]
        pad_len: torch.Tensor,       # [B]
        first_token_idx: int,
    ) -> None:
        """
        In-place shift for a single source column using scalar-factor scaling.
        """
        B, h, Q, K = attn.shape
        device = attn.device
        dtype = attn.dtype

        src_idx = pad_len + int(first_token_idx)                       # [B]
        valid_src = (src_idx >= 0) & (src_idx < K)                     # [B]
        if not valid_src.any():
            return

        src_safe = src_idx.clamp(0, K - 1).to(torch.long)              # [B]
        gather_idx = src_safe.view(B, 1, 1, 1).expand(B, h, Q, 1)       # [B,h,Q,1]

        # src_prob: [B,h,Q]
        src_prob = torch.gather(attn, dim=-1, index=gather_idx).squeeze(-1)

        # sums of non-src targets: [B,h,Q]
        # (masked columns are 0, so this automatically excludes padding/future keys)
        row_sum = attn.sum(dim=-1)                                     # [B,h,Q]
        denom = row_sum - src_prob                                     # [B,h,Q]

        # skip padding rows: q < pad_len
        q_pos = torch.arange(Q, device=device).view(1, 1, Q)           # [1,1,Q]
        valid_q = (q_pos >= pad_len.view(B, 1, 1))                     # [B,1,Q]

        # compute redistribution
        redistribute = (self.delta_ratio * src_prob).to(dtype)         # [B,h,Q]

        # valid rows must have denom > 0 and be non-padding row and have valid src
        ok = (denom > self.min_denom) & valid_q & valid_src.view(B, 1, 1)
        if not ok.any():
            return

        denom_safe = denom.clamp_min(self.min_denom).to(dtype)
        factor = torch.where(ok, redistribute / denom_safe, torch.zeros_like(denom_safe))  # [B,h,Q]

        # 1) scale entire row: p *= (1 + factor)
        attn.mul_(1.0 + factor.unsqueeze(-1))

        # 2) fix src column: subtract (src_prob*factor + redistribute)
        correction = torch.where(ok, src_prob.to(dtype) * factor + redistribute, torch.zeros_like(redistribute))
        attn.scatter_add_(dim=-1, index=gather_idx, src=-correction.unsqueeze(-1))

    @torch.no_grad()
    def _shift_inplace_multi_source_bhqk(
        self,
        attn: torch.Tensor,          # view: [B, h, Q, K]
        pad_len: torch.Tensor,       # [B]
        first_token_indices: List[int],
    ) -> None:
        """
        Multi-source version:
        For each source column s in S:
          p_s' = p_s * (1 - delta_ratio)
        Redistribute total removed mass to other legal (non-padding, non-source) columns proportionally.

        Implemented via: scale whole row by (1+factor), then scatter-correct each source column.
        """
        B, h, Q, K = attn.shape
        device = attn.device
        dtype = attn.dtype

        # sanitize/unique offsets (relative indices)
        offs = sorted({int(i) for i in first_token_indices if isinstance(i, int)})
        if not offs:
            return

        L = len(offs)
        offs_t = torch.tensor(offs, dtype=torch.long, device=device)  # [L]

        # src indices per batch: [B,L]
        src_idx = pad_len.view(B, 1) + offs_t.view(1, L)
        valid_src = (src_idx >= 0) & (src_idx < K)  # [B,L]
        if not valid_src.any():
            return

        src_safe = src_idx.clamp(0, K - 1).to(torch.long)  # [B,L]

        # gather src probs: [B,h,Q,L]
        gather_idx = src_safe.view(B, 1, 1, L).expand(B, h, Q, L)
        src_prob = torch.gather(attn, dim=-1, index=gather_idx)  # [B,h,Q,L]

        # zero-out invalid sources (so later scatter-corrections are also zero)
        src_prob = src_prob * valid_src.view(B, 1, 1, L).to(dtype)

        sum_src_prob = src_prob.sum(dim=-1)                # [B,h,Q]
        row_sum = attn.sum(dim=-1)                         # [B,h,Q] (usually ~1 for non-padding rows)
        denom = row_sum - sum_src_prob                     # [B,h,Q] targets mass (non-src)

        # skip padding rows (q < pad_len)
        q_pos = torch.arange(Q, device=device).view(1, 1, Q)           # [1,1,Q]
        valid_q = (q_pos >= pad_len.view(B, 1, 1))                     # [B,1,Q]

        redistribute_total = (self.delta_ratio * sum_src_prob).to(dtype)  # [B,h,Q]

        ok = (denom > self.min_denom) & valid_q & (redistribute_total > 0)
        if not ok.any():
            return

        denom_safe = denom.clamp_min(self.min_denom).to(dtype)
        factor = torch.where(ok, redistribute_total / denom_safe, torch.zeros_like(denom_safe))  # [B,h,Q]

        # 1) scale entire row (masked columns are 0 so stay 0)
        attn.mul_(1.0 + factor.unsqueeze(-1))

        # 2) correct each source column:
        # after scaling, src became p_s*(1+factor), but we want p_s*(1-delta_ratio)
        # subtract p_s*(factor + delta_ratio)
        correction_per_source = src_prob * (factor.unsqueeze(-1) + self.delta_ratio)  # [B,h,Q,L]

        # scatter subtract per source (L is typically small; avoids big masks)
        for l in range(L):
            idx_l = src_safe[:, l].view(B, 1, 1, 1).expand(B, h, Q, 1)
            corr_l = correction_per_source[..., l]  # [B,h,Q]
            attn.scatter_add_(dim=-1, index=idx_l, src=-corr_l.unsqueeze(-1))

    @torch.no_grad()
    def shift_probs(self, attn_weights: torch.Tensor, layer_to_modify: int) -> torch.Tensor:
        """
        attn_weights: [B, H, Q, K] (softmaxed)
        Extremely optimized in-place shift.
        """
        if self.delta_ratio == 0.0:
            return attn_weights

        B, H, Q, K = attn_weights.shape

        # Prefill (prompt) has Q>1, so intervention only happens there.
        if Q == 1:
            return attn_weights
        
        heads = self._heads_for_layer(layer_to_modify, H)
        if not heads:
            return attn_weights

        # Infer pad_len once per call
        probe_head = heads[0]
        pad_len = self._infer_left_pad_lengths(attn_weights, probe_head=probe_head)  # [B]

        # Head selection without forcing a copy when possible
        if len(heads) == H:
            if isinstance(self.first_token_idx, int):
                self._shift_inplace_single_source_bhqk(attn_weights, pad_len, self.first_token_idx)
            else:
                self._shift_inplace_multi_source_bhqk(attn_weights, pad_len, list(self.first_token_idx))
            return attn_weights

        if self._is_contiguous(heads):
            hs = sorted(set(heads))
            sl = slice(hs[0], hs[-1] + 1)
            attn_view = attn_weights[:, sl, :, :]  # view
            if isinstance(self.first_token_idx, int):
                self._shift_inplace_single_source_bhqk(attn_view, pad_len, self.first_token_idx)
            else:
                self._shift_inplace_multi_source_bhqk(attn_view, pad_len, list(self.first_token_idx))
            return attn_weights

        # Non-contiguous: loop heads to avoid advanced-indexing copy
        for h in heads:
            attn_view = attn_weights[:, h:h + 1, :, :]  # view [B,1,Q,K]
            if isinstance(self.first_token_idx, int):
                self._shift_inplace_single_source_bhqk(attn_view, pad_len, self.first_token_idx)
            else:
                self._shift_inplace_multi_source_bhqk(attn_view, pad_len, list(self.first_token_idx))
        return attn_weights