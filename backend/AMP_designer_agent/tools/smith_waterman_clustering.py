#!/usr/bin/env python3
"""
Smith-Waterman (SW) based sequence similarity and clustering for AMP Designer Agent

This script computes pairwise local alignment similarity between sequences from
backend/AMP_designer_agent/data/grampa.csv using a fast, pure-Python Smith-Waterman
implementation with a k-mer (3-mer) prefilter to avoid O(N^2) alignments.

It then performs DBSCAN on a precomputed distance matrix: distance = 1 - similarity.

Outputs are written to data/sw_clusters by default.

Usage (examples):
  python -m backend.AMP_designer_agent.tools.smith_waterman_clustering \
      --input backend/AMP_designer_agent/data/grampa.csv \
      --out-dir backend/AMP_designer_agent/data/sw_clusters \
      --sample-size 200 --pair-threshold 0.4 --eps 0.6 --min-samples 2

Notes:
- Pure Python; no external binaries required
- Reasonable defaults for short antimicrobial peptides
- Performance relies on k-mer prefilter; tune --k and --min-shared-kmers if needed
"""

import argparse
import os
import logging
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Workspace-relative default paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "backend", "AMP_designer_agent", "data", "grampa.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "backend", "AMP_designer_agent", "data", "sw_clusters")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SW scoring params (linear gap for speed)
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_PENALTY = -2  # linear gap (no affine)


def load_sequences(input_csv: str, min_len: int, max_len: int, sample_size: int | None) -> Tuple[List[str], pd.DataFrame]:
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column")

    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].astype(str).str.upper()

    mask = df["sequence"].str.len().between(min_len, max_len)
    df = df[mask]

    unique_sequences = df["sequence"].drop_duplicates().tolist()

    if sample_size is not None and len(unique_sequences) > sample_size:
        logger.info(f"Sampling {sample_size} sequences from {len(unique_sequences)} total")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(unique_sequences), size=sample_size, replace=False)
        unique_sequences = [unique_sequences[i] for i in idx]
        df = df[df["sequence"].isin(unique_sequences)]

    logger.info(f"Prepared {len(unique_sequences)} unique sequences after filtering")
    return unique_sequences, df


def kmer_set(seq: str, k: int) -> Set[str]:
    if len(seq) < k:
        return {seq}
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


def build_kmer_index(sequences: List[str], k: int) -> Dict[str, Set[int]]:
    index: Dict[str, Set[int]] = {}
    for i, s in enumerate(sequences):
        for kmer in kmer_set(s, k):
            index.setdefault(kmer, set()).add(i)
    return index


def candidate_pairs_by_kmer(sequences: List[str], k: int, min_shared_kmers: int) -> Iterable[Tuple[int, int]]:
    index = build_kmer_index(sequences, k)
    # For each sequence, collect candidates that share >= min_shared_kmers k-mers
    for i, s in enumerate(sequences):
        counts: Dict[int, int] = {}
        for kmer in kmer_set(s, k):
            for j in index.get(kmer, ()):  # sequences that contain this kmer
                if j <= i:
                    continue
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            if c >= min_shared_kmers:
                yield (i, j)


def smith_waterman_score(a: str, b: str) -> int:
    # Local alignment with linear gap penalty; 2-row DP to reduce memory
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    prev = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        curr = [0] * (lb + 1)
        ai = a[i-1]
        for j in range(1, lb + 1):
            score_diag = prev[j-1] + (MATCH_SCORE if ai == b[j-1] else MISMATCH_SCORE)
            score_up = prev[j] + GAP_PENALTY
            score_left = curr[j-1] + GAP_PENALTY
            val = score_diag
            if score_up > val:
                val = score_up
            if score_left > val:
                val = score_left
            if val < 0:
                val = 0
            curr[j] = val
            if val > best:
                best = val
        prev = curr
    return best


def sw_similarity(a: str, b: str) -> float:
    # Normalize SW score to [0,1] by max possible all-match score
    denom = MATCH_SCORE * max(len(a), len(b))
    if denom <= 0:
        return 0.0
    s = smith_waterman_score(a, b)
    sim = s / float(denom)
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def compute_similarity_matrix_sw(
    sequences: List[str],
    k: int,
    min_shared_kmers: int,
) -> np.ndarray:
    n = len(sequences)
    sim = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 1.0)

    # Only compute SW for candidate pairs
    cnt = 0
    for i, j in candidate_pairs_by_kmer(sequences, k=k, min_shared_kmers=min_shared_kmers):
        s = sw_similarity(sequences[i], sequences[j])
        sim[i, j] = s
        sim[j, i] = s
        cnt += 1
        if cnt % 1000 == 0:
            logger.info(f"Computed SW for {cnt} candidate pairs...")
    logger.info(f"Total SW alignments computed: {cnt}")
    return sim


def cluster_with_dbscan(sim: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    dist = 1.0 - sim
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dist)
    return labels


def save_pair_results(
    sim: np.ndarray,
    sequences: List[str],
    out_dir: str,
    pair_threshold: float,
) -> int:
    rows = []
    n = len(sequences)
    for i in range(n):
        for j in range(i+1, n):
            s = float(sim[i, j])
            if s >= pair_threshold and s > 0.0:
                rows.append({
                    "qidx": i,
                    "sidx": j,
                    "similarity": s,
                    "qseq": sequences[i],
                    "sseq": sequences[j],
                })
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "sw_similar_pairs.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved pairwise results to {out_path} (n={len(df)})")
    return int(len(df))


def save_cluster_files(labels: np.ndarray, sequences: List[str], df_raw: pd.DataFrame, out_dir: str, max_files: int | None) -> None:
    seq_labels = {seq: lab for seq, lab in zip(sequences, labels)}
    clusters: Dict[int, List[str]] = {}
    for seq, lab in seq_labels.items():
        if lab == -1:
            continue
        clusters.setdefault(int(lab), []).append(seq)
    clusters = dict(sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True))
    saved = 0
    for lab, seqs in clusters.items():
        if max_files is not None and saved >= max_files:
            break
        cluster_df = df_raw[df_raw["sequence"].isin(seqs)].copy()
        cluster_df.insert(0, "cluster_id", lab)
        out_path = os.path.join(out_dir, f"cluster_{lab}_size_{len(cluster_df)}.csv")
        cluster_df.to_csv(out_path, index=False)
        saved += 1
        logger.info(f"Saved cluster {lab} to {out_path} (records={len(cluster_df)})")
    logger.info(f"Total clusters saved: {saved}")


def main():
    ap = argparse.ArgumentParser(description="Smith-Waterman based clustering for AMP sequences")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to grampa.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUTDIR, help="Output directory for SW clusters")
    ap.add_argument("--min-len", type=int, default=5, help="Minimum sequence length")
    ap.add_argument("--max-len", type=int, default=100, help="Maximum sequence length")
    ap.add_argument("--sample-size", type=int, default=300, help="Optional cap on number of unique sequences (None to disable)")
    ap.add_argument("--pair-threshold", type=float, default=0.4, help="Similarity threshold for saving pairs [0-1]")
    ap.add_argument("--eps", type=float, default=0.6, help="DBSCAN eps on distance (1-sim)")
    ap.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples")
    ap.add_argument("--k", type=int, default=3, help="k-mer size for prefilter")
    ap.add_argument("--min-shared-kmers", type=int, default=2, help="Minimum shared k-mers to run SW alignment")
    ap.add_argument("--max-cluster-files", type=int, default=None, help="Limit number of cluster CSVs to save")
    args = ap.parse_args()

    if isinstance(args.sample_size, int) and args.sample_size <= 0:
        args.sample_size = None

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sequences
    sequences, df_clean = load_sequences(args.input, args.min_len, args.max_len, args.sample_size)

    # Compute similarity matrix via SW with k-mer prefilter
    sim = compute_similarity_matrix_sw(sequences, k=args.k, min_shared_kmers=args.min_shared_kmers)

    # Cluster
    labels = cluster_with_dbscan(sim, eps=args.eps, min_samples=args.min_samples)

    # Save
    num_pairs_saved = save_pair_results(sim, sequences, args.out_dir, pair_threshold=args.pair_threshold)
    save_cluster_files(labels, sequences, df_clean, args.out_dir, max_files=args.max_cluster_files)

    # Summary
    summary = {
        "total_sequences": len(sequences),
        "pair_threshold": args.pair_threshold,
        "dbscan_eps": args.eps,
        "dbscan_min_samples": args.min_samples,
        "k": args.k,
        "min_shared_kmers": args.min_shared_kmers,
        "num_pairs_saved": int(num_pairs_saved),
        "note": "Similarity = SW_score / (MATCH_SCORE * max(len(q), len(s)))",
    }
    pd.Series(summary).to_json(os.path.join(args.out_dir, "summary.json"), indent=2)
    logger.info(f"Done. Results available in: {args.out_dir}")


if __name__ == "__main__":
    main()

