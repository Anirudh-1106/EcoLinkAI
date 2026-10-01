"""
Pair neighbourhoods for a pair-aware MC-GNN scorer.

A GNN that embeds each facility on its own cannot see what two facilities
share -- it cannot even count their common neighbours (Zhang et al.,
"Labeling Trick", NeurIPS 2021). Neural Common Neighbor (Wang, Yang and
Zhang, ICLR 2024) fixes this without giving up a single message-passing run:
score a pair from the two embeddings and from the pooled embeddings of a node
set that depends on the pair.

The usual set, common neighbours N(s) & N(b), is nearly always empty here:
suppliers trade with recyclers, not with each other, so a supplier and a
recycler share almost no direct neighbours. Measured on the 2024 TRI test
pairs, it is non-empty for 8% of pairs and separates real partners at AUC
0.613. NCN's general form (its equation 8) takes higher-order sets instead,
and two of those carry the signal:

  co-partners   facilities that share a trading partner with the supplier
                and already trade with the recycler: N(b) & N2(s). Non-empty
                for 36% of pairs; their count alone reaches AUC 0.890.
  peers         the supplier's peers (tri_views' peer network) that already
                trade with the recycler: N_peer(s) & N(b). AUC 0.735.

Each set is cut to the K members most strongly tied to the supplier. Ties
through a shared partner are weighted as Adamic-Adar does, 1/log(1 + degree)
of that partner, so sharing a niche recycler says more than sharing one that
takes from thousands. The full set sizes are kept as counts.

Everything comes from the relationship and peer networks, which are built from
history only, so no set can contain information from the year being scored.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

PAIR_SET_SIZE = 16
PAIR_SET_NAMES = ("co_partners", "peers")
NO_MEMBER = -1


def _adjacency(edge_index: np.ndarray, n: int) -> sparse.csr_matrix:
    data = np.ones(edge_index.shape[1], dtype=np.float32)
    a = sparse.csr_matrix((data, (edge_index[0], edge_index[1])), shape=(n, n))
    a.data[:] = 1.0
    return a


def pair_neighbourhoods(g: dict, k: int = PAIR_SET_SIZE) -> dict:
    """
    For every scored row of g: the two member sets (padded to k with -1) and
    their full sizes.

    Returns {"co_partners": int32 (rows, k), "peers": int32 (rows, k),
             "counts": float32 (rows, 2)}.
    """
    n = len(g["node_keys"])
    rel = _adjacency(g["views"]["relationship"], n)
    peer = _adjacency(g["views"]["peer"], n)
    src, dst = g["edge_index"]

    # Adamic-Adar tie strength between a supplier and every node two hops away.
    degree = np.asarray(rel.sum(axis=1)).ravel()
    inv_log = np.divide(1.0, np.log1p(degree), out=np.zeros_like(degree, dtype=float),
                        where=degree > 1).astype(np.float32)
    senders = np.unique(src)
    two_hop = (rel[senders] @ sparse.diags(inv_log) @ rel).tocsr()
    row_of = {int(s): i for i, s in enumerate(senders)}

    rel_lists = [set(rel.indices[rel.indptr[i]:rel.indptr[i + 1]]) for i in range(n)]
    peer_lists = [peer.indices[peer.indptr[i]:peer.indptr[i + 1]] for i in range(n)]

    co = np.full((len(src), k), NO_MEMBER, dtype=np.int32)
    pe = np.full((len(src), k), NO_MEMBER, dtype=np.int32)
    counts = np.zeros((len(src), 2), dtype=np.float32)

    for r, (s, b) in enumerate(zip(src, dst)):
        s, b = int(s), int(b)
        partners_of_b = rel_lists[b]

        i = row_of[s]
        lo, hi = two_hop.indptr[i], two_hop.indptr[i + 1]
        cand, weight = two_hop.indices[lo:hi], two_hop.data[lo:hi]
        if len(cand) and partners_of_b:
            keep = np.fromiter((c in partners_of_b and c != s for c in cand), dtype=bool, count=len(cand))
            members, w = cand[keep], weight[keep]
            counts[r, 0] = len(members)
            if len(members):
                top = members[np.argsort(-w, kind="stable")[:k]]
                co[r, :len(top)] = top

        peers_using_b = [p for p in peer_lists[s] if p in partners_of_b]
        counts[r, 1] = len(peers_using_b)
        if peers_using_b:
            take = peers_using_b[:k]
            pe[r, :len(take)] = take

    return {"co_partners": co, "peers": pe, "counts": np.log1p(counts).astype(np.float32)}


__all__ = ["NO_MEMBER", "PAIR_SET_NAMES", "PAIR_SET_SIZE", "pair_neighbourhoods"]
