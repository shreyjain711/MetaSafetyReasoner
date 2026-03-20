#!/usr/bin/env python3
"""
Convert the MetaSafetyReasoner 7.5k prompt dataset into VERL-ready train/test splits.
The output schema mirrors gsm8k.py: data_source / prompt / ability / reward_model / extra_info.
"""

import argparse
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd


ABILITY = "text_generation"

DEFAULT_SYSTEM_PROMPT = (
    # "You are MetaSafetyReasoner's safety classifier. "
    # "Given a user query, categorize the risk (CIA, CSA, PDA, SSW, ELE, GEN, etc.) "
    # "and rate the risk level 0-3. Respond with structured reasoning and the final label."
    "You are helpful and harmless assistant. Answer the user's question."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="prompt_engg/datasets/data_7_5k.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--dataset_name", default="msr_7_5k")
    parser.add_argument("--system_prompt_path", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--write_json", action="store_true")
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    file = Path(path)
    if file.suffix in {".json", ".jsonl"}:
        return pd.read_json(file, lines=True)
    if file.suffix == ".parquet":
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file type: {file}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_path:
        system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8").strip()

    df = read_table(args.input_path)
    print(df.head(5))

    def normalize_model_response(val):
        """Convert model response to dict: parse JSON strings, None -> empty dict."""
        if val is None:
            return {}
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                print(f"Failed to parse JSON string: {val}")
                return {}
        if isinstance(val, dict):
            return val
        return {}

    # Build VERL-style records row by row.
    records = []
    for idx, row in df.iterrows():
        prompt_text = row["Prompt"]

        query_labels = {
            "risk_cat": row.get("risk_cat"),
            "risk_subcat": row.get("risk_subcat"),
            "risk_level": row.get("risk_level"),
        }

        extra_info = {
            "index": idx,
            "nishoak_msr_index": row.get("Nishoak_msr_index"),
            "source": row.get("Source"),
            "component_source": row.get("component_source"),
            "category": row.get("Category"),
            "is_valid": bool(row.get("is_valid", True)),
            "model_feedback": {
                "gpt_oss_20b": normalize_model_response(row.get("response_gpt-oss-20b")),
                "qwen3_30b": normalize_model_response(row.get("response_Qwen3-30B-A3B-Instruct-2507-FP8")),
                "qwen3_4b": normalize_model_response(row.get("response_Qwen3-4B-Instruct-2507-FP8")),
            },
        }

        record = {
            "id": row['Nishoak_msr_index'],
            "data_source": row['Source'],
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(prompt_text).strip()},
            ],
            "ability": ABILITY,
            "reward_model": {"style": "model", "label": query_labels},
            "extra_info": extra_info,
        }
        records.append(record)

    dataset = pd.DataFrame(records)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(dataset))
    split = int(len(dataset) * args.train_ratio)
    train_df = dataset.iloc[indices[:split]].reset_index(drop=True)
    test_df = dataset.iloc[indices[split:]].reset_index(drop=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    if args.write_json:
        train_df.to_json(output_dir / "train.json", orient="records", force_ascii=False)
        test_df.to_json(output_dir / "test.json", orient="records", force_ascii=False)

    meta = {
        "source_path": args.input_path,
        "dataset_name": args.dataset_name,
        "ability": ABILITY,
        "system_prompt": system_prompt,
        "num_rows": len(dataset),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[data_process] Done. train={len(train_df)} test={len(test_df)} -> {output_dir}")


if __name__ == "__main__":
    main()


"""
python data_process.py \
    --input_path prompt_engg/datasets/data_7_5k.json \
    --dataset_name msr_7_5k \
    --train_ratio 0.9 \
    --seed 2025 \
    --write_json
"""