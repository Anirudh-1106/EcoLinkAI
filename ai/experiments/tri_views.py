"""
The three networks a multi-view MC-GNN reads, built from EPA TRI history.

  relationship  facilities that have traded, in both directions: a sender
                learns from the recyclers it used as much as they learn
                from their suppliers
  capability    a facility's waste linked to the facilities that process
                that waste -- the chemicals and recovery methods one sends
                out against those another takes in -- whether or not the
                two have ever dealt
  peer          facilities that resemble each other: the same detailed
                industry (NAICS at 3 to 6 digits) and the same waste streams,
                sent or received

Capability and peer are similarity graphs: every facility is joined to the
NEIGHBOURS_PER_FACILITY most similar others. Similarity is the cosine of
IDF-weighted profiles, so sharing a rare chemical counts for far more than
sharing lead compounds, which thousands of facilities handle.

Everything comes from history, with one addition: the waste a sender has to
place in the year being scored. That is the listing -- what the question is
about -- and never where it went. It is what lets a facility with no history
join the capability and peer networks from its first listing, exactly as a
newly registered company would on the platform.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse

NEIGHBOURS_PER_FACILITY = 10
NAICS_LEVELS = (3, 4, 5, 6)

# Rows of the similarity product computed at once; bounds the dense block to
# CHUNK x facilities floats.
CHUNK = 2048


def _profiles(keys: list[str], items_of: dict[str, set], vocab: dict) -> sparse.csr_matrix:
    """Binary facility-by-item matrix."""
    rows, cols = [], []
    for i, k in enumerate(keys):
        for it in items_of.get(k, ()):
            rows.append(i)
            cols.append(vocab[it])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(keys), len(vocab)))


def _tfidf(m: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    """Weight columns by IDF and scale each row to unit length."""
    m = (m @ sparse.diags(idf)).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags(1.0 / norms) @ m).tocsr().astype(np.float32)


def _idf(*matrices: sparse.csr_matrix) -> np.ndarray:
    """log(N / df) over every profile given, so all share one weighting."""
    n = sum(m.shape[0] for m in matrices)
    df = sum(np.asarray((m > 0).sum(axis=0)).ravel() for m in matrices)
    return np.where(df > 0, np.log(n / np.maximum(df, 1)), 0.0).astype(np.float32)


def _top_k(a: sparse.csr_matrix, b: sparse.csr_matrix, k: int, same_space: bool) -> dict[int, list[int]]:
    """For each row of a, the k rows of b it is most similar to (similarity > 0)."""
    out: dict[int, list[int]] = {}
    has_profile = np.asarray(a.getnnz(axis=1)).ravel() > 0
    bt = b.T.tocsc()
    for start in range(0, a.shape[0], CHUNK):
        stop = min(start + CHUNK, a.shape[0])
        sims = (a[start:stop] @ bt).toarray()
        if same_space:
            sims[np.arange(stop - start), np.arange(start, stop)] = 0.0
        kk = min(k, sims.shape[1] - 1) if sims.shape[1] > 1 else 0
        if kk <= 0:
            continue
        top = np.argpartition(-sims, kk, axis=1)[:, :kk]
        for r in range(stop - start):
            if not has_profile[start + r]:
                continue
            chosen = [int(c) for c in top[r] if sims[r, c] > 0]
            if chosen:
                out[start + r] = chosen
    return out


def _undirected(pairs) -> np.ndarray:
    """Edge index with both directions of every pair, duplicates and self-loops removed."""
    s = {(a, b) for a, b in pairs if a != b} | {(b, a) for a, b in pairs if a != b}
    if not s:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(s), dtype=np.int64).T


def build_views(
    nodes: list[str],
    history,
    listings: dict[str, set],
    naics_of: dict[str, str],
    k: int = NEIGHBOURS_PER_FACILITY,
) -> dict:
    """
    The three networks over `nodes`, plus what the pair-level features need.

    history   transfers strictly before the year being scored
    listings  sender -> CAS numbers it has to place in that year
    naics_of  facility -> NAICS code, possibly empty
    """
    index = {key: i for i, key in enumerate(nodes)}

    # What each facility sends out and takes in, as chemicals and as methods.
    out_items: dict[str, set] = defaultdict(set)
    in_items: dict[str, set] = defaultdict(set)
    senders_to: dict[int, set] = defaultdict(set)
    traded = set()
    for t in history:
        out_items[t.sender] |= {("cas", t.cas), ("method", t.method)}
        in_items[t.receiver] |= {("cas", t.cas), ("method", t.method)}
        senders_to[index[t.receiver]].add(index[t.sender])
        traded.add((index[t.sender], index[t.receiver]))
    for sender, cas_set in listings.items():
        out_items[sender] |= {("cas", c) for c in cas_set}

    material_vocab = {it: j for j, it in enumerate(sorted(
        {it for s in out_items.values() for it in s} | {it for s in in_items.values() for it in s}
    ))}
    out_raw = _profiles(nodes, out_items, material_vocab)
    in_raw = _profiles(nodes, in_items, material_vocab)
    material_idf = _idf(out_raw, in_raw)
    out_m, in_m = _tfidf(out_raw, material_idf), _tfidf(in_raw, material_idf)

    # Capability: what one sends out against what another takes in, both ways
    # round, so a supplier and a processor of the same waste find each other.
    cap_out = _top_k(out_m, in_m, k, same_space=True)
    cap_in = _top_k(in_m, out_m, k, same_space=True)
    capability = [(i, j) for i, js in cap_out.items() for j in js] + \
                 [(i, j) for i, js in cap_in.items() for j in js]

    # Peer: same industry at every level of the NAICS hierarchy, and the same
    # streams in the same role -- a sender resembles senders of its wastes,
    # a recycler resembles recyclers of the same intake.
    peer_items: dict[str, set] = defaultdict(set)
    for key in nodes:
        code = "".join(ch for ch in (naics_of.get(key) or "") if ch.isdigit())
        peer_items[key] |= {(f"naics{n}", code[:n]) for n in NAICS_LEVELS if len(code) >= n}
        peer_items[key] |= {("out",) + it for it in out_items.get(key, ())}
        peer_items[key] |= {("in",) + it for it in in_items.get(key, ())}
    peer_vocab = {it: j for j, it in enumerate(sorted({it for s in peer_items.values() for it in s}))}
    peer_raw = _profiles(nodes, peer_items, peer_vocab)
    peer_m = _tfidf(peer_raw, _idf(peer_raw))
    peers = _top_k(peer_m, peer_m, k, same_space=True)
    peer_pairs = [(i, j) for i, js in peers.items() for j in js]

    return {
        "views": {
            "relationship": _undirected(traded),
            "capability": _undirected(capability),
            "peer": _undirected(peer_pairs),
        },
        "out_profile": out_m,
        "in_profile": in_m,
        "peers": peers,
        "senders_to": senders_to,
        "k": k,
    }


def pair_graph_features(built: dict, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    What the two new networks say about one pair, as plain numbers.

    For models that cannot read a graph -- XGBoost, logistic regression -- so
    that they are offered the same new information the multi-view MC-GNN is,
    and a win for it cannot come from information the others were denied.

      capability_score  how well the sender's waste matches what the receiver
                        takes in (cosine of the two profiles, 0-1)
      peer_votes        share of the sender's peers that have sent to this
                        receiver before
    """
    out_m, in_m = built["out_profile"], built["in_profile"]
    capability = np.asarray(out_m[src].multiply(in_m[dst]).sum(axis=1)).ravel()
    peers, senders_to, k = built["peers"], built["senders_to"], built["k"]
    votes = np.fromiter(
        (len(set(peers.get(int(s), ())) & senders_to.get(int(r), set())) / k for s, r in zip(src, dst)),
        dtype=np.float32, count=len(src),
    )
    return np.column_stack([np.clip(capability, 0.0, 1.0), votes]).astype(np.float32)


PAIR_GRAPH_FEATURE_NAMES = ("capability_score", "peer_votes")


def view_stats(views: dict[str, np.ndarray], n_nodes: int) -> dict:
    out = {}
    for name, ei in views.items():
        deg = np.bincount(ei[0], minlength=n_nodes) if ei.size else np.zeros(n_nodes)
        out[name] = {
            "edges": int(ei.shape[1]),
            "mean_degree": round(float(deg.mean()), 2),
            "isolated_share": round(float((deg == 0).mean()), 4),
        }
    return out
