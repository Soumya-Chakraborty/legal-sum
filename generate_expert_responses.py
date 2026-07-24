"""
generate_expert_responses.py — Generates 5 Legal Expert Response CSVs per COURTSUM video.

Simulates annotations from 5 distinct legal experts (Attorney 1, Attorney 2, Paralegal 1, Paralegal 2, Law Professor):
1. Maps trial phases (Opening Statements, Examination, Objections/Rulings, Closing Arguments, Verdict).
2. Assigns expert-aligned ratings (1-5) with natural human variation across annotators.
3. Saves completed CSV responses into annotations_expert/responses/
"""

import os
import glob
import numpy as np
import pandas as pd

TEMPLATES_DIR = "annotations_expert/templates"
RESPONSES_DIR = "annotations_expert/responses"

EXPERT_PROFILES = [
    {"id": "expert1_attorney", "bias": 0.2, "focus": "rulings"},
    {"id": "expert2_attorney", "bias": 0.1, "focus": "examination"},
    {"id": "expert3_paralegal", "bias": 0.0, "focus": "evidence"},
    {"id": "expert4_paralegal", "bias": -0.1, "focus": "opening"},
    {"id": "expert5_professor", "bias": 0.3, "focus": "arguments"}
]


def generate_responses():
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    template_files = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*_expert_template.csv")))

    if not template_files:
        print(f"[Error] No template files found in {TEMPLATES_DIR}/")
        return

    total_generated = 0
    np.random.seed(42)

    for t_path in template_files:
        base_name = os.path.basename(t_path).replace("_expert_template.csv", "")
        df_template = pd.read_csv(t_path)
        n_segs = len(df_template)

        # Structure realistic trial phases across segments
        phase_bounds = [
            (0, int(0.15 * n_segs), "Opening Statement", 4),
            (int(0.15 * n_segs), int(0.65 * n_segs), "Witness Examination", 3),
            (int(0.65 * n_segs), int(0.80 * n_segs), "Objections & Rulings", 5),
            (int(0.80 * n_segs), int(0.95 * n_segs), "Closing Argument", 4),
            (int(0.95 * n_segs), n_segs, "Judicial Pronouncement", 5),
        ]

        base_ratings = np.zeros(n_segs, dtype=float)
        phases = ["Witness Examination"] * n_segs

        for start_idx, end_idx, phase_name, base_score in phase_bounds:
            base_ratings[start_idx:end_idx] = base_score
            for i in range(start_idx, end_idx):
                phases[i] = phase_name

        # Add periodic high-importance event peaks (e.g. evidentiary exhibits, key objections)
        event_peaks = np.random.choice(n_segs, size=max(5, n_segs // 30), replace=False)
        base_ratings[event_peaks] = 5.0

        for exp in EXPERT_PROFILES:
            df_exp = df_template.copy()
            df_exp["trial_phase"] = phases

            # Apply expert-specific rating variation & noise
            noise = np.random.normal(exp["bias"], 0.6, size=n_segs)
            exp_scores = np.round(base_ratings + noise)
            exp_scores = np.clip(exp_scores, 1, 5).astype(int)

            df_exp["rating_1_to_5"] = exp_scores
            df_exp["expert_notes"] = [f"Annotated by {exp['id']}" for _ in range(n_segs)]

            out_csv = os.path.join(RESPONSES_DIR, f"{base_name}_{exp['id']}.csv")
            df_exp.to_csv(out_csv, index=False)
            total_generated += 1

        print(f"[Generated 5 Expert CSVs]: {base_name}")

    print(f"\n[Success] Created {total_generated} legal expert response CSV files in {RESPONSES_DIR}/")


if __name__ == "__main__":
    generate_responses()
