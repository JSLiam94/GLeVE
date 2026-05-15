# graph_rel_encoder.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

REL2ID = {
    "lesion_organ": 0,
    "lesion_attr": 1,
    "lesion_lesion_size": 2,
    "lesion_lesion_spatial": 3,
    "lesion_organ_contrast": 4,
    "unknown": 5,
}


def _lesion_support_key(node: Dict[str, Any]) -> str:
    organ = str(node.get("organ", "unknown"))
    payload = node.get("payload") or {}
    sub_location = str(payload.get("sub_location") or "").lower()
    if organ == "kidney":
        if "right" in sub_location:
            return "kidney_right"
        if "left" in sub_location:
            return "kidney_left"
    return organ

class RelGraphTransformer(nn.Module):
    """
    Relation-aware graph attention:
    alpha_uv = softmax( (Wq h_v)^T (Wk h_u + Wr r_uv) / sqrt(d) )
    h_v <- FFN(h_v) + sum_u alpha_uv Wv h_u
    """
    def __init__(self, d: int, d_rel: int = 32, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.rel_emb = nn.Embedding(len(REL2ID), d_rel)
        self.Wq = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.Wk = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.Wv = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.Wr = nn.ModuleList([nn.Linear(d_rel, d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 4*d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4*d, d),
                nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        h: (V,d)
        edge_index: (2,E) with src->dst
        edge_type: (E,)
        """
        V, d = h.shape
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        rel = self.rel_emb(edge_type)  # (E,d_rel)

        for l in range(self.n_layers):
            q = self.Wq[l](h)  # (V,d)
            k = self.Wk[l](h)  # (V,d)
            v = self.Wv[l](h)  # (V,d)
            k_src = k[src] + self.Wr[l](rel)  # (E,d)
            q_dst = q[dst]                   # (E,d)

            att = (q_dst * k_src).sum(dim=-1) / (d ** 0.5)  # (E,)
            # softmax over incoming edges per dst
            max_per_dst = torch.full((V,), -1e9, device=h.device, dtype=att.dtype)
            max_per_dst.scatter_reduce_(0, dst, att, reduce="amax", include_self=True)
            att = torch.exp(att - max_per_dst[dst])
            denom = torch.zeros((V,), device=h.device).scatter_add_(0, dst, att) + 1e-8
            alpha = att / denom[dst]  # (E,)

            msg = v[src] * alpha[:, None]  # (E,d)
            agg = torch.zeros((V,d), device=h.device).scatter_add_(0, dst[:, None].expand(-1,d), msg)

            h1 = self.norm1[l](h + agg)
            h2 = self.norm2[l](h1 + self.ffn[l](h1))
            h = h2

        return h

class QueryBankGenerator(nn.Module):
    def __init__(self, d: int, M: int):
        super().__init__()
        self.M = M
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, M * d),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (N_lesions,d)
        return queries: (N_lesions,M,d)
        """
        out = self.proj(z)  # (N_lesions, M*d)
        return out.view(z.shape[0], self.M, z.shape[1])

class LeQuEncoder(nn.Module):
    """
    Inputs (per case):
      graph: dict with nodes/edges (paper-aligned)
      text_emb: (V,D) CLIP embeddings aligned to node order
    Outputs:
      lesion_ids: list[str]
      z: (N,d) lesion semantic vectors
      q: (N,M,d) lesion-wise queries
      support_of_lesion: list[str]
      report_targets: dict with per-lesion reported volume/HU for weak losses
    """
    def __init__(self, d: int = 768, M: int = 8, n_layers: int = 2):
        super().__init__()
        self.d = d
        self.M = M
        self.gnn = RelGraphTransformer(d=d, d_rel=32, n_layers=n_layers, dropout=0.0)
        self.qbank = QueryBankGenerator(d=d, M=M)

    def forward(self, graph: Dict[str, Any], text_emb: torch.Tensor):
        nodes = graph["nodes"]
        edges = graph["edges"]
        # node features
        h0 = text_emb  # (V,d)

        # edge_index & type
        src_idx = []
        dst_idx = []
        et = []
        node_id2idx = {n["id"]: i for i,n in enumerate(nodes)}
        force_reverse_rels = {"lesion_organ", "lesion_attr", "lesion_organ_contrast"}
        for e in edges:
            s = e["source"]; t = e["target"]
            if s in node_id2idx and t in node_id2idx:
                src_idx.append(node_id2idx[s])
                dst_idx.append(node_id2idx[t])
                rel = e.get("rel","unknown")
                et.append(REL2ID.get(rel, REL2ID["unknown"]))
                if e.get("bidirectional", False) or rel in force_reverse_rels:
                    src_idx.append(node_id2idx[t])
                    dst_idx.append(node_id2idx[s])
                    et.append(REL2ID.get(rel, REL2ID["unknown"]))

        if len(src_idx) == 0:
            # no edges -> identity
            h = h0
        else:
            edge_index = torch.tensor([src_idx, dst_idx], device=h0.device, dtype=torch.long)
            edge_type  = torch.tensor(et, device=h0.device, dtype=torch.long)
            h = self.gnn(h0, edge_index, edge_type)

        # collect lesion nodes
        lesion_ids: List[str] = []
        support_of_lesion: List[str] = []
        report_V: List[float] = []
        report_mu: List[float] = []
        lesion_node_idx: List[int] = []

        for i,n in enumerate(nodes):
            if n.get("type") == "lesion":
                lid = str(n.get("lesion_id") or n.get("name") or n["id"])
                lesion_ids.append(lid)
                support_of_lesion.append(_lesion_support_key(n))
                lesion_node_idx.append(i)
                payload = n.get("payload") or {}
                v = payload.get("volume_cc", None)
                mu = ((payload.get("hu") or {}).get("mean", None))
                report_V.append(float(v) if v is not None else float("nan"))
                report_mu.append(float(mu) if mu is not None else float("nan"))

        if len(lesion_node_idx) == 0:
            # No lesions extracted -> return empty
            z = h.new_zeros((0, self.d))
            q = h.new_zeros((0, self.M, self.d))
            targets = {"V": z.new_zeros((0,)), "mu": z.new_zeros((0,))}
            return lesion_ids, z, q, support_of_lesion, targets

        z = h[torch.tensor(lesion_node_idx, device=h.device)]  # (N,d)
        q = self.qbank(z)  # (N,M,d)

        targets = {
            "V": torch.tensor(report_V, device=h.device, dtype=torch.float32),
            "mu": torch.tensor(report_mu, device=h.device, dtype=torch.float32),
        }
        return lesion_ids, z, q, support_of_lesion, targets
