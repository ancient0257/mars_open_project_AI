"""
Support Integrity Auditor (SIA) — Inference Script
===================================================
Usage: python predict.py --input new_tickets.csv --output results/

Loads a trained SIA pipeline and scores new tickets:
  - Stage 1: Pseudo-label generation on new data
  - Stage 2: Classifier prediction
  - Stage 3: Evidence dossier generation for flagged tickets
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
import joblib

from train_pipeline import (
    generate_pseudo_labels,
    build_features,
    generate_dossier,
    PRIORITY_MAP, SEVERITY_NAMES
)

warnings.filterwarnings("ignore")


def predict(input_csv: str,
            model_dir: str = "model",
            output_dir: str = "results") -> pd.DataFrame:
    """
    Run full inference pipeline on new CSV data.

    Args:
        input_csv:  Path to input CSV with ticket data
        model_dir:  Directory containing trained model artefacts
        output_dir: Directory for output files

    Returns:
        DataFrame with predictions and derived features
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load Model ─────────────────────────────────────────────────────────
    print(f"[Predict] Loading model artefacts from {model_dir}/")
    clf      = joblib.load(f"{model_dir}/clf_model.pkl")
    scaler   = joblib.load(f"{model_dir}/scaler.pkl")
    encoders = joblib.load(f"{model_dir}/encoders.pkl")

    # ── Load Data ──────────────────────────────────────────────────────────
    print(f"[Predict] Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"[Predict] {len(df):,} tickets loaded")

    # ── Stage 1: Pseudo-Labels on New Data ─────────────────────────────────
    df_labeled, sig_agree = generate_pseudo_labels(df)
    print(f"[Predict] Signal agreement on new data: {sig_agree:.3f}")

    # ── Stage 2: Classifier Prediction ─────────────────────────────────────
    X, _ = build_features(df_labeled, encoders=encoders, fit=False)
    X_s  = scaler.transform(X)

    probs = clf.predict_proba(X_s)[:, 1]
    preds = clf.predict(X_s)

    df_labeled["pred_mismatch"]   = preds
    df_labeled["pred_confidence"] = probs
    df_labeled["pred_label"]      = [
        "Mismatch" if p else "Consistent" for p in preds
    ]

    n_m = int(preds.sum())
    print(f"[Predict] {n_m:,} mismatch(es) detected ({n_m/len(df)*100:.1f}%)")

    # ── Stage 3: Dossiers for Flagged Tickets ──────────────────────────────
    dossiers = [
        generate_dossier(row, float(row["pred_confidence"]))
        for _, row in df_labeled[df_labeled["pred_mismatch"] == 1].iterrows()
    ]

    # ── Save Outputs ───────────────────────────────────────────────────────
    out_csv  = f"{output_dir}/predictions.csv"
    out_json = f"{output_dir}/dossiers.json"

    # Select key columns for output
    output_cols = [
        "Ticket_ID", "Ticket_Subject", "Priority_Level",
        "pred_label", "pred_confidence", "pred_mismatch",
        "inferred_severity", "assigned_severity", "severity_delta",
        "Issue_Category", "Ticket_Channel"
    ]
    available_cols = [c for c in output_cols if c in df_labeled.columns]
    df_labeled[available_cols].to_csv(out_csv, index=False)

    with open(out_json, "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"[Predict] Predictions saved → {out_csv}")
    print(f"[Predict] Dossiers saved    → {out_json}  ({len(dossiers)} tickets)")

    return df_labeled


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input",     required=True,
                        help="Path to input CSV file")
    parser.add_argument("--model_dir", default="model",
                        help="Directory containing trained model (default: model/)")
    parser.add_argument("--output",    default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    predict(args.input, args.model_dir, args.output)

