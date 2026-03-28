"""
MetroPT-3 predictive maintenance — MANOVA + SVM (synthetic MetroPT-style data).
Used by GET/POST /api/metropt3/*; optional Gemini interpretation via GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.multivariate.manova import MANOVA

FEATURE_COLS = [
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "TP2",
    "TP3",
    "H1",
    "LPS",
    "Oil_level",
]

FEATURE_LABELS = {
    "Oil_temperature": "Oil temperature (°C)",
    "Motor_current": "Motor current (A)",
    "COMP": "Compressor active (COMP)",
    "TP2": "Pressure TP2 (bar)",
    "TP3": "Pressure TP3 (bar)",
    "H1": "Pressure drop H1 (bar)",
    "LPS": "Low-pressure switch (LPS)",
    "Oil_level": "Oil level (1=normal)",
}

CONT_FEATS = ["Oil_temperature", "Motor_current", "TP2", "TP3", "H1"]

_bundle: Optional[dict[str, Any]] = None


def load_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    normal = pd.DataFrame(
        {
            "Oil_temperature": rng.normal(70, 5, 500),
            "Motor_current": rng.normal(5, 1.5, 500),
            "COMP": rng.choice([0, 1], 500, p=[0.2, 0.8]),
            "TP2": rng.normal(9, 1, 500),
            "TP3": rng.normal(9, 1, 500),
            "H1": rng.normal(0.1, 0.05, 500),
            "LPS": np.zeros(500),
            "Oil_level": np.ones(500),
            "Status": 0,
        }
    )
    failure = pd.DataFrame(
        {
            "Oil_temperature": rng.normal(95, 8, 150),
            "Motor_current": rng.normal(12, 3, 150),
            "COMP": np.ones(150),
            "TP2": rng.normal(4, 1.5, 150),
            "TP3": rng.normal(8, 2, 150),
            "H1": rng.normal(0.8, 0.2, 150),
            "LPS": rng.choice([0, 1], 150, p=[0.7, 0.3]),
            "Oil_level": rng.choice([0, 1], 150, p=[0.4, 0.6]),
            "Status": 1,
        }
    )
    df = pd.concat([normal, failure], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def run_manova(df: pd.DataFrame) -> dict[str, Any]:
    cont_df = df[CONT_FEATS + ["Status"]].copy()
    formula = " + ".join(CONT_FEATS) + " ~ Status"
    maov = MANOVA.from_formula(formula, data=cont_df)
    result = maov.mv_test()

    parsed: dict[str, Any] = {}
    for effect_name, effect_result in result.results.items():
        parsed[effect_name] = {}
        stat_table = effect_result["stat"]
        for test_name in stat_table.index:
            row = stat_table.loc[test_name]
            parsed[effect_name][test_name] = {
                "Value": round(float(row["Value"]), 5),
                "F Value": round(float(row["F Value"]), 4),
                "Num DF": float(row["Num DF"]),
                "Den DF": round(float(row["Den DF"]), 2),
                "Pr > F": float(f"{row['Pr > F']:.6g}"),
            }

    uni: dict[str, Any] = {}
    for col in CONT_FEATS:
        g0 = df.loc[df.Status == 0, col]
        g1 = df.loc[df.Status == 1, col]
        f_stat, p = sp_stats.f_oneway(g0, g1)
        uni[col] = {"F": round(float(f_stat), 3), "p": float(f"{p:.4g}")}

    return {"multivariate": parsed, "univariate": uni}


def train_svm(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURE_COLS]
    y = df["Status"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    report = classification_report(y_te, y_pred, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    auc = float(roc_auc_score(y_te, y_prob))

    scaler = pipe.named_steps["scaler"]
    svm_ = pipe.named_steps["svm"]
    sv_coef = svm_.dual_coef_
    sv_vecs = svm_.support_vectors_
    w_approx = (sv_coef @ sv_vecs).flatten()
    w_norm = w_approx / (np.abs(w_approx).sum() + 1e-9)

    return {
        "pipeline": pipe,
        "scaler": scaler,
        "report": report,
        "cm": cm,
        "auc": auc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feat_weights": dict(zip(FEATURE_COLS, w_norm.tolist())),
    }


def _contrib_for_row(pipe: Pipeline, input_row: pd.DataFrame) -> dict[str, float]:
    scaler = pipe.named_steps["scaler"]
    svm_ = pipe.named_steps["svm"]
    xs = scaler.transform(input_row).flatten()
    sv_coef = svm_.dual_coef_
    sv_vecs = svm_.support_vectors_
    w_approx = (sv_coef @ sv_vecs).flatten()
    return {FEATURE_COLS[j]: round(float(w_approx[j] * xs[j]), 4) for j in range(len(FEATURE_COLS))}


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        df = load_dataset()
        _bundle = {
            "df": df,
            "manova": run_manova(df),
            "svm": train_svm(df),
        }
    return _bundle


def row_dict_to_df(row: dict[str, Any]) -> pd.DataFrame:
    vals = []
    for c in FEATURE_COLS:
        if c not in row:
            raise KeyError(f"Missing key: {c}")
        vals.append(float(row[c]))
    return pd.DataFrame([vals], columns=FEATURE_COLS)


def predict_from_row(row: dict[str, Any]) -> dict[str, Any]:
    b = get_bundle()
    pipe = b["svm"]["pipeline"]
    input_row = row_dict_to_df(row)
    pred = int(pipe.predict(input_row)[0])
    prob = float(pipe.predict_proba(input_row)[0, 1])
    contribs = _contrib_for_row(pipe, input_row)
    if hasattr(pipe, "decision_function"):
        try:
            dec = float(pipe.decision_function(input_row)[0])
        except Exception:
            dec = None
    else:
        dec = None
    return {
        "class": pred,
        "class_label": "Failure" if pred == 1 else "Normal",
        "probability_failure_pct": round(prob * 100, 1),
        "contributions": contribs,
        "decision_function": dec,
    }


def build_gemini_prompt(
    manova_results: dict[str, Any],
    svm_metrics: dict[str, Any],
    current_input: Optional[dict[str, Any]] = None,
    prediction: Optional[dict[str, Any]] = None,
) -> str:
    mw = manova_results["multivariate"].get("Status", {})
    uni = manova_results["univariate"]
    rep = svm_metrics["report"]
    feat = svm_metrics["feat_weights"]

    sig_feats = [f for f, v in uni.items() if v["p"] < 0.05]
    wilks = mw.get("Wilks' lambda", {})

    return textwrap.dedent(
        f"""
    You are a senior reliability engineer interpreting predictive-maintenance
    diagnostics for the MetroPT-3 train air-production unit dataset.

    Speak in clear, technical but accessible English.  Structure your response
    as three short paragraphs:
      1. Statistical significance (MANOVA findings)
      2. Model performance and what it means operationally
      3. Current sensor reading interpretation (if provided)

    ── DATASET CONTEXT ─────────────────────────────────────────────────────
    Records: 650  |  Normal: 500 (76.9 %)  |  Failure: 150 (23.1 %)
    Features: Oil temperature, Motor current, Compressor flag (COMP),
              TP2 pressure, TP3 pressure, H1 pressure drop,
              Low-pressure switch (LPS), Oil level.

    ── MANOVA RESULTS ──────────────────────────────────────────────────────
    Wilks' lambda : {wilks.get("Value", "N/A")}
    F value       : {wilks.get("F Value", "N/A")}
    Num DF        : {wilks.get("Num DF", "N/A")}
    Den DF        : {wilks.get("Den DF", "N/A")}
    p-value       : {wilks.get("Pr > F", "N/A")}

    Interpretation hint:
      Wilks' lambda close to 0 = strong group separation.
      p < 0.05 = the Normal vs Failure means differ significantly
                 across the multivariate feature space.

    Univariate ANOVA (per feature):
    {json.dumps(uni, indent=2)}

    Features with statistically significant difference (p < 0.05):
    {", ".join(sig_feats) if sig_feats else "None"}

    ── SVM MODEL PERFORMANCE ───────────────────────────────────────────────
    Kernel         : RBF  |  C=10  |  class_weight=balanced
    ROC-AUC        : {svm_metrics["auc"]:.4f}
    CV ROC-AUC     : {svm_metrics["cv_mean"]:.4f} ± {svm_metrics["cv_std"]:.4f}  (5-fold)
    Precision-0    : {rep["0"]["precision"]:.3f}   Recall-0 : {rep["0"]["recall"]:.3f}
    Precision-1    : {rep["1"]["precision"]:.3f}   Recall-1 : {rep["1"]["recall"]:.3f}
    Macro F1       : {rep["macro avg"]["f1-score"]:.3f}

    Top feature weights (approximate, from SVM dual coefficients):
    {json.dumps({k: round(v, 4) for k, v in sorted(feat.items(), key=lambda x: abs(x[1]), reverse=True)}, indent=2)}
    Positive weight → pushes toward failure.
    Negative weight → pushes toward normal.

    ── CURRENT SENSOR READING ──────────────────────────────────────────────
    {json.dumps(current_input, indent=2) if current_input else "No live reading provided."}

    ── CURRENT PREDICTION ──────────────────────────────────────────────────
    {json.dumps(prediction, indent=2) if prediction else "No prediction requested."}

    ── YOUR TASK ────────────────────────────────────────────────────────────
    Write a concise diagnostic report (≤ 200 words) covering the three
    paragraphs above.  End with one actionable maintenance recommendation.
    Do NOT reproduce raw numbers verbatim — synthesise them into insight.
    """
    ).strip()


def diagnostics_payload() -> dict[str, Any]:
    b = get_bundle()
    df = b["df"]
    manova_res = b["manova"]
    svm_pkg = b["svm"]
    cm = svm_pkg["cm"]
    rep = svm_pkg["report"]

    X_list = df[FEATURE_COLS].values.tolist()
    y_list = [int(x) for x in df["Status"].tolist()]

    manova_rows = []
    for effect, tests in manova_res["multivariate"].items():
        for test_name, stats in tests.items():
            pval = stats.get("Pr > F", 1.0)
            manova_rows.append(
                {
                    "effect": effect,
                    "test": test_name,
                    "value": stats.get("Value"),
                    "f_value": stats.get("F Value"),
                    "num_df": stats.get("Num DF"),
                    "den_df": stats.get("Den DF"),
                    "p_value": pval,
                    "significant": bool(pval < 0.05),
                }
            )

    uni_rows = []
    for feat, v in manova_res["univariate"].items():
        uni_rows.append(
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "F": v["F"],
                "p": v["p"],
                "significant": bool(v["p"] < 0.05),
            }
        )

    return {
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "cont_feats": CONT_FEATS,
        "dataset": {
            "total": len(df),
            "normal": int((df.Status == 0).sum()),
            "failure": int((df.Status == 1).sum()),
        },
        "X": X_list,
        "y": y_list,
        "manova": {
            "multivariate": manova_res["multivariate"],
            "univariate": manova_res["univariate"],
            "table_rows": manova_rows,
            "univariate_rows": uni_rows,
        },
        "svm": {
            "auc": svm_pkg["auc"],
            "cv_mean": svm_pkg["cv_mean"],
            "cv_std": svm_pkg["cv_std"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "failure_recall": rep["1"]["recall"],
            "failure_precision": rep["1"]["precision"],
            "confusion_matrix": cm.tolist(),
            "report": rep,
            "feat_weights": svm_pkg["feat_weights"],
        },
        "disclaimer": "Synthetic MetroPT-style demo data; not live equipment telemetry.",
    }


def call_gemini_interpret(
    current_input: Optional[dict[str, Any]],
    prediction: Optional[dict[str, Any]],
) -> tuple[str, Optional[str]]:
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return "", "Set GEMINI_API_KEY on the server for AI interpretation."
    b = get_bundle()
    prompt = build_gemini_prompt(b["manova"], b["svm"], current_input, prediction)
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"].strip())
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"
        m = genai.GenerativeModel(model_name)
        r = m.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 1024},
        )
        text = (getattr(r, "text", None) or "").strip()
        return text or "(Empty response)", None
    except Exception as e:
        return "", str(e)
