from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dichasus_formal import run_dichasus_formal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MPRI on the locked formal DICHASUS protocol.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--support-size", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--subcarrier-count", type=int, default=193)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-query-links", type=int)
    args = parser.parse_args()
    output = run_dichasus_formal(
        protocol_path=args.protocol,
        support_size=args.support_size,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        subcarrier_count=args.subcarrier_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_query_links=args.max_query_links,
    )
    print(output)


if __name__ == "__main__":
    main()
