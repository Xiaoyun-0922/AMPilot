#!/usr/bin/env python3
"""
BLAST-based sequence similarity and clustering for AMP Designer Agent

This script builds BLAST databases from grampa.csv sequences and performs
all-vs-all BLASTP (task=blastp-short) to compute pairwise similarities, then
clusters sequences using DBSCAN (precomputed distance = 1 - similarity).

Outputs are written to a new folder under data/: data/blast_clusters

Usage (examples):
  python -m backend.AMP_designer_agent.tools.blast_clustering \
      --input backend/AMP_designer_agent/data/grampa.csv \
      --out-dir backend/AMP_designer_agent/data/blast_clusters \
      --pair-threshold 0.5 --eps 0.6 --min-samples 2 --num-threads 4

Notes:
- Requires BLAST+ binaries in PATH: makeblastdb, blastp
- Recommended for short peptides: task=blastp-short, and consider -seg no
- For large datasets, sampling can be used via --sample-size
"""

import argparse
import os
import sys
import shutil
import logging
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Workspace-relative default paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "backend", "AMP_designer_agent", "data", "grampa.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "backend", "AMP_designer_agent", "data", "blast_clusters")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BLAST_FMT = "6 qseqid sseqid pident length evalue bitscore"


def check_blast_dependencies(blast_bin: str | None = None) -> Dict[str, str]:
    """Ensure BLAST+ tools are available; return resolved executable paths.

    If blast_bin is provided, look for executables under that directory first.
    """
    resolved: Dict[str, str] = {}
    for exe in ("makeblastdb", "blastp"):
        candidate = None
        if blast_bin:
            candidate = os.path.join(blast_bin, exe)
            if os.name == "nt" and not candidate.lower().endswith(".exe"):
                candidate_exe = candidate + ".exe"
                if os.path.isfile(candidate_exe):
                    candidate = candidate_exe
        # If not found or not provided, fall back to PATH
        if not candidate or not os.path.isfile(candidate):
            candidate = shutil.which(exe)
        if not candidate:
            raise RuntimeError(
                f"Required BLAST+ binary '{exe}' not found. Install NCBI BLAST+ or set --blast-bin to its bin directory."
            )
        resolved[exe] = candidate
    return resolved


def load_sequences(input_csv: str, min_len: int, max_len: int, sample_size: int | None) -> Tuple[List[str], pd.DataFrame]:
    """Load sequences from grampa.csv and filter/deduplicate.

    Returns unique sequence list and a cleaned DataFrame.
    """
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column")

    df = df.dropna(subset=["sequence"]).copy()
    # Normalize sequences to uppercase letters (AA alphabet)
    df["sequence"] = df["sequence"].astype(str).str.upper()

    # Length filtering
    mask = df["sequence"].str.len().between(min_len, max_len)
    df = df[mask]

    # Deduplicate by exact sequence
    unique_sequences = df["sequence"].drop_duplicates().tolist()

    # Optional sampling to cap runtime
    if sample_size is not None and len(unique_sequences) > sample_size:
        logger.info(f"Sampling {sample_size} sequences from {len(unique_sequences)} total")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(unique_sequences), size=sample_size, replace=False)
        unique_sequences = [unique_sequences[i] for i in idx]
        df = df[df["sequence"].isin(unique_sequences)]

    logger.info(f"Prepared {len(unique_sequences)} unique sequences after filtering")
    return unique_sequences, df


def write_fasta(sequences: List[str], fasta_path: str) -> Dict[str, str]:
    """Write sequences to FASTA and return id->sequence mapping.

    We assign compact IDs: seq0, seq1, ... for BLAST.
    """
    id_to_seq: Dict[str, str] = {}
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            sid = f"seq{i}"
            id_to_seq[sid] = seq
            f.write(f">{sid}\n{seq}\n")
    return id_to_seq


def run_makeblastdb(fasta_path: str, db_prefix: str, exe_paths: Dict[str, str]) -> None:
    import subprocess
    cmd = [
        exe_paths["makeblastdb"], "-in", fasta_path, "-dbtype", "prot", "-out", db_prefix,
    ]
    logger.info("Running makeblastdb...")
    subprocess.run(cmd, check=True)


def run_blastp_all_vs_all(fasta_path: str, db_prefix: str, out_path: str, num_threads: int, seg: bool, exe_paths: Dict[str, str]) -> None:
    import subprocess
    cmd = [
        exe_paths["blastp"],
        "-query", fasta_path,
        "-db", db_prefix,
        "-task", "blastp-short",
        "-outfmt", BLAST_FMT,
        "-evalue", "200",
        "-max_target_seqs", "1000000",
        "-comp_based_stats", "0",
        "-out", out_path,
        "-num_threads", str(num_threads),
    ]
    if not seg:
        cmd.extend(["-seg", "no"])
    logger.info("Running blastp all-vs-all (this may take a while)...")
    subprocess.run(cmd, check=True)


def parse_blast_table(out_path: str) -> pd.DataFrame:
    cols = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]
    df = pd.read_csv(out_path, sep="\t", header=None, names=cols)
    return df


def compute_similarity_matrix(
    hits: pd.DataFrame,
    id_to_seq: Dict[str, str],
) -> np.ndarray:
    """Compute symmetric similarity matrix from BLAST hits.

    Similarity definition:
        sim(q, s) = (pident/100) * (aln_len / max(len(q), len(s)))
    For multiple HSPs per pair, we take the maximum sim value.
    """
    ids = list(id_to_seq.keys())
    id_index = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    sim = np.zeros((n, n), dtype=np.float32)

    # Precompute lengths
    seqlen = {sid: len(id_to_seq[sid]) for sid in ids}

    # Aggregate best similarity per (q, s)
    for _, row in hits.iterrows():
        q = row["qseqid"]; s = row["sseqid"]
        if q == s:
            continue
        if q not in id_index or s not in id_index:
            continue
        i = id_index[q]; j = id_index[s]
        pident = float(row["pident"]) / 100.0
        aln_len = float(row["length"])  # aligned length
        denom = max(seqlen[q], seqlen[s])
        if denom <= 0:
            continue
        score = pident * (aln_len / denom)
        if score > sim[i, j]:
            sim[i, j] = score
        if score > sim[j, i]:
            sim[j, i] = score

    # Ensure diagonal is 1.0 (identity)
    np.fill_diagonal(sim, 1.0)
    return sim


def cluster_with_dbscan(sim: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cluster using DBSCAN on distance matrix (1 - similarity)."""
    dist = 1.0 - sim
    # Zero diagonal already; DBSCAN expects distance matrix with metric='precomputed'
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dist)
    return labels


def save_pair_results(
    pairs_df: pd.DataFrame,
    id_to_seq: Dict[str, str],
    out_dir: str,
    pair_threshold: float,
) -> int:
    """Save pairwise hits above threshold to a CSV with sequences included.

    Returns number of saved pairs.
    """
    # Compute similarity using the same formula to filter pairs
    def _sim(row) -> float:
        q = row["qseqid"]; s = row["sseqid"]
        pident = float(row["pident"]) / 100.0
        aln_len = float(row["length"])  # aligned length
        denom = max(len(id_to_seq[q]), len(id_to_seq[s]))
        return pident * (aln_len / denom)

    pairs_df = pairs_df[pairs_df["qseqid"] != pairs_df["sseqid"]].copy()
    if not pairs_df.empty:
        pairs_df["similarity"] = pairs_df.apply(_sim, axis=1)
        pairs_df = pairs_df[pairs_df["similarity"] >= pair_threshold]
        if not pairs_df.empty:
            pairs_df["qseq"] = pairs_df["qseqid"].map(id_to_seq)
            pairs_df["sseq"] = pairs_df["sseqid"].map(id_to_seq)
            pairs_df.sort_values(["similarity", "bitscore"], ascending=[False, False], inplace=True)
    out_path = os.path.join(out_dir, "blast_similar_pairs.csv")
    pairs_df.to_csv(out_path, index=False)
    logger.info(f"Saved pairwise results to {out_path} (n={len(pairs_df)})")
    return int(len(pairs_df))


def save_cluster_files(
    labels: np.ndarray,
    sequences: List[str],
    df_raw: pd.DataFrame,
    out_dir: str,
    max_files: int | None,
):
    """Save cluster CSVs mirroring the cosine-sim pipeline behavior."""
    # Map seq to label
    seq_labels = {seq: lab for seq, lab in zip(sequences, labels)}

    # Build clusters (exclude noise -1)
    clusters: Dict[int, List[str]] = {}
    for seq, lab in seq_labels.items():
        if lab == -1:
            continue
        clusters.setdefault(int(lab), []).append(seq)

    # Sort clusters by size descending
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
    ap = argparse.ArgumentParser(description="BLAST-based clustering for AMP sequences")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to grampa.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUTDIR, help="Output directory for BLAST clusters")
    ap.add_argument("--min-len", type=int, default=5, help="Minimum sequence length")
    ap.add_argument("--max-len", type=int, default=100, help="Maximum sequence length")
    ap.add_argument("--sample-size", type=int, default=2000, help="Optional cap on number of unique sequences (None to disable)")
    ap.add_argument("--pair-threshold", type=float, default=0.5, help="Similarity threshold for saving pairs [0-1]")
    ap.add_argument("--eps", type=float, default=0.6, help="DBSCAN eps on distance (1-sim)")
    ap.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples")
    ap.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1), help="BLASTP num_threads")
    ap.add_argument("--seg", action="store_true", help="Enable SEG filtering (default off)")
    ap.add_argument("--max-cluster-files", type=int, default=None, help="Limit number of cluster CSVs to save")
    ap.add_argument("--blast-bin", type=str, default=None, help="Path to BLAST+ bin directory (where blastp/makeblastdb reside)")
    args = ap.parse_args()

    # Normalize sample-size None
    if isinstance(args.sample_size, int) and args.sample_size <= 0:
        args.sample_size = None

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Check BLAST deps
    try:
        exe_paths = check_blast_dependencies(args.blast_bin)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    # 2) Load sequences
    sequences, df_clean = load_sequences(args.input, args.min_len, args.max_len, args.sample_size)

    # 3) Prepare temp workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "sequences.faa")
        db_prefix = os.path.join(tmpdir, "seqdb")
        blast_out = os.path.join(tmpdir, "blast_all_vs_all.tsv")

        id_to_seq = write_fasta(sequences, fasta_path)

        # 4) Build BLAST DB and run BLASTP
        run_makeblastdb(fasta_path, db_prefix, exe_paths)
        run_blastp_all_vs_all(fasta_path, db_prefix, blast_out, args.num_threads, seg=args.seg, exe_paths=exe_paths)

        # 5) Parse BLAST output and compute similarity matrix
        hits = parse_blast_table(blast_out)
        sim = compute_similarity_matrix(hits, id_to_seq)

    # 6) Cluster with DBSCAN
    labels = cluster_with_dbscan(sim, eps=args.eps, min_samples=args.min_samples)

    # 7) Save pairs and clusters
    # Recompute hits with IDs available after tempdir closes: we still have id_to_seq dict
    # hits was already loaded before tempdir closed
    # Save pairs
    # Remap qseqid/sseqid to ensure only present IDs
    valid_ids = set(id_to_seq.keys())
    hits = hits[hits["qseqid"].isin(valid_ids) & hits["sseqid"].isin(valid_ids)].copy()
    num_pairs_saved = save_pair_results(hits, id_to_seq, args.out_dir, pair_threshold=args.pair_threshold)

    # Save clusters mirroring cosine pipeline behavior
    save_cluster_files(labels, sequences, df_clean, args.out_dir, max_files=args.max_cluster_files)

    # 8) Save a small summary
    summary = {
        "total_sequences": len(sequences),
        "pair_threshold": args.pair_threshold,
        "dbscan_eps": args.eps,
        "dbscan_min_samples": args.min_samples,
        "num_pairs_saved": int(num_pairs_saved),
        "note": "Similarity = (pident/100) * (aln_len / max(len(q), len(s)))",
    }
    pd.Series(summary).to_json(os.path.join(args.out_dir, "summary.json"), indent=2)
    logger.info(f"Done. Results available in: {args.out_dir}")


if __name__ == "__main__":
    main()

