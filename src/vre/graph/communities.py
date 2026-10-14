"""Cộng đồng phường (Leiden trên đồ thị kNN hồ sơ phường) và modularity (chuyển nguyên từ `src/graph.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def modularity(edges, comms) -> float:
    """Q = Σ [L_c/m − (d_c/2m)²] (Phụ lục A.4)."""
    m, Q = len(edges), 0.0
    for c in comms:
        L_c = sum(1 for u, v in edges if u in c and v in c)
        d_c = sum((u in c) + (v in c) for u, v in edges)
        Q += L_c / m - (d_c / (2 * m)) ** 2
    return Q


def ward_profiles(df: pd.DataFrame, min_listings: int = 8) -> pd.DataFrame:
    """Hồ sơ mỗi phường: giá/m² trung vị, số tin (log) và tỷ lệ từng loại hình."""
    d = df.dropna(subset=["province", "district", "ward", "price_m2_mil", "property_type"])
    keys = ["province", "district", "ward"]
    sizes = d.groupby(keys).size()
    prof = d.groupby(keys)["price_m2_mil"].median().to_frame("median_price_m2")
    prof["log_n"] = np.log1p(sizes)
    prof = prof.join(pd.crosstab([d[k] for k in keys], d["property_type"], normalize="index"))
    return prof[sizes >= min_listings]


def ward_communities(prof: pd.DataFrame, k: int = 5, seed: int = 42) -> tuple[pd.DataFrame, float]:
    """Dựng đồ thị kNN giữa các phường theo hồ sơ rồi chạy Leiden. Trả (hồ sơ có cột community, Q)."""
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    X = prof.assign(median_price_m2=np.log(prof["median_price_m2"])).to_numpy()
    X = StandardScaler().fit_transform(X)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    edges = {tuple(sorted((i, int(j)))) for i, row in enumerate(idx) for j in row[1:]}
    G = ig.Graph(n=len(prof), edges=sorted(edges))
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition, seed=seed)
    return prof.assign(community=part.membership), G.modularity(part.membership)


def community_table(prof: pd.DataFrame) -> pd.DataFrame:
    """Mỗi cộng đồng: số phường, giá/m² TB và ví dụ lấy từ tỉnh chính (tỉnh có nhiều phường nhất trong cụm)."""
    rows = []
    for cid, group in prof.groupby("community"):
        province = group.index.get_level_values("province").value_counts().index[0]
        examples = group.xs(province, level="province", drop_level=False).index[:3]
        rows.append({"cộng đồng": cid, "số phường": len(group),
                     "giá/m² TB (triệu)": round(group["median_price_m2"].mean(), 1),
                     "tỉnh chính": province,
                     "ví dụ": ", ".join(f"{w} ({d})" for _, d, w in examples)})
    return pd.DataFrame(rows).sort_values("giá/m² TB (triệu)").reset_index(drop=True)
