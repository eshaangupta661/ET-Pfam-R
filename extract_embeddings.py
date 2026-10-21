"""
CLI tool for extracting and formatting protein language model (pLM) embeddings for ET-Pfam.

Supports ESM-2 (e.g., facebook/esm2_t33_650M_UR50D) alongside the baseline ESM-1b model.
Extracted embeddings are guaranteed to have shape [1280, L] and are stored as .pk files
in data/embeddings/, matching the downstream BaseModel and ensemble classifier requirements.

Usage Examples:
    # Extract using ESM-2 (650M parameter default, 1280 dim):
    python3 extract_embeddings.py -f data/sample_sequences.fasta -m esm2

    # Extract using baseline ESM-1b model:
    python3 extract_embeddings.py -f data/sample_sequences.fasta -m baseline

    # Custom output directory and batch size:
    python3 extract_embeddings.py -f sequences.fasta -o data/embeddings/ -b 16 -d cuda
"""
import os
import sys
import json
import argparse
from src.plm.pipeline import EmbeddingPipeline
from src.plm.registry import list_available_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and format per-residue protein embeddings using modular pLMs."
    )
    parser.add_argument(
        "-f",
        "--fasta_path",
        type=str,
        required=True,
        help="Path to input FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save extracted embeddings (.pk files). Defaults to emb_path in config/env.json.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="esm2",
        help="pLM model alias or checkpoint (e.g. 'esm2', 'baseline', 'facebook/esm2_t33_650M_UR50D').",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding inference (default: 8).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or device ID). Defaults to env.json setting.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/env.json",
        help="Path to environment configuration file.",
    )
    parser.add_argument(
        "--target_dim",
        type=int,
        default=1280,
        help="Target channel dimension expected by ET-Pfam classifier (default: 1280).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files if present.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run downstream shape compatibility verification after extraction.",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available registered pLM models and exit.",
    )
    return parser.parse_args()


def load_env_settings(config_path: str):
    """Loads environment configuration if present."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    args = parse_args()

    if args.list_models:
        print("\nRegistered Protein Language Models (pLMs):")
        print("-" * 75)
        for alias, info in list_available_models().items():
            print(f"  {alias:<15} : {info['checkpoint']} ({info['native_dim']} dim)")
            print(f"                  {info['description']}")
        print("-" * 75)
        sys.exit(0)

    env_config = load_env_settings(args.config_path)

    # Determine output directory
    output_dir = args.output_dir or env_config.get("emb_path", "data/embeddings/")

    # Determine device
    device = args.device or env_config.get("device", "cuda")

    print("=" * 70)
    print("ET-Pfam Embedding Generation Pipeline")
    print("=" * 70)
    print(f"Input FASTA      : {args.fasta_path}")
    print(f"Selected pLM     : {args.model_name}")
    print(f"Output Directory : {output_dir}")
    print(f"Target Dimension : {args.target_dim} (ET-Pfam compatible)")
    print(f"Device           : {device}")
    print(f"Batch Size       : {args.batch_size}")
    print("=" * 70)

    if not os.path.isfile(args.fasta_path):
        print(f"Error: FASTA file not found at '{args.fasta_path}'.", file=sys.stderr)
        sys.exit(1)

    # Initialize the pipeline
    pipeline = EmbeddingPipeline(
        plm=args.model_name,
        output_dir=output_dir,
        device=device,
        target_dim=args.target_dim,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )

    # Process all sequences
    saved_files = pipeline.process_fasta(args.fasta_path)

    # Verify downstream shape compatibility if requested
    if args.verify and saved_files:
        sample_pid = next(iter(saved_files.keys()))
        print("\nRunning downstream shape compatibility verification...")
        pipeline.verify_compatibility(sample_pid)

    print("\n[Complete] Extracted embeddings are ready for downstream model training/testing.")


if __name__ == "__main__":
    main()
