"""
MetroPT-3 predictive maintenance — linear-kernel SVM (fast; synthetic MetroPT-style data).
Used by GET/POST /api/metropt3/*; optional Gemini interpretation via GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
    "Oil_temperature": "Oil temp — engine / hydraulic circuit (°C)",
    "Motor_current": "Motor current — pump & compaction load (A)",
    "COMP": "Auxiliary compressor running (COMP)",
    "TP2": "Line pressure TP2 (bar)",
    "TP3": "Line pressure TP3 (bar)",
    "H1": "Restriction / ΔP across H1 (bar)",
    "LPS": "Low-pressure switch (trip)",
    "Oil_level": "Sump oil level OK (1 = normal)",
}

_bundle: Optional[dict[str, Any]] = None

# Synthetic demo: overlapping classes + ambiguous band + label noise → holdout metrics below 1.0.
# Core normal/failure counts match (≈50/50 before noise); ambiguous band is exactly half/half labels.
# LinearSVC scales to large n; RBF SVC was too slow for cold start.
_DATASET_N_NORMAL = 1500
_DATASET_N_FAILURE = 1500
_DATASET_N_AMBIGUOUS = 400
_LABEL_NOISE_RATE = 0.092
_SVM_C = 0.85
_CV_SPLITS = 3
_KERNEL = "linear"

# Defaults when user only pastes a subset of sensors (matches training scale)
_DEFAULT_READINGS: dict[str, Any] = {
    "Oil_temperature": 70.0,
    "Motor_current": 5.0,
    "COMP": 1,
    "TP2": 9.0,
    "TP3": 9.0,
    "H1": 0.1,
    "LPS": 0,
    "Oil_level": 1,
}


def label_to_sensor_key(name: str) -> Optional[str]:
    """Map pasted label text to FEATURE_COLS name."""
    n = name.lower().replace("\t", " ").strip()
    n = re.sub(r"\s+", " ", n)
    if re.search(r"oil\s*temp|truck\s*temp|hydraulic\s*temp|coolant", n):
        return "Oil_temperature"
    if re.search(r"motor\s*current", n):
        return "Motor_current"
    if re.match(r"^comp|^compressor", n):
        return "COMP"
    if "tp2" in n:
        return "TP2"
    if "tp3" in n:
        return "TP3"
    if re.search(r"\bh1\b", n):
        return "H1"
    if "lps" in n or "low-pressure" in n:
        return "LPS"
    if re.search(r"oil\s*level", n):
        return "Oil_level"
    return None


def fill_sensor_row(partial: dict[str, float]) -> dict[str, Any]:
    """Merge partial parsed values with defaults; binary features as 0/1."""
    out = dict(_DEFAULT_READINGS)
    for k, v in partial.items():
        if k in ("COMP", "LPS", "Oil_level"):
            out[k] = int(float(v) >= 0.5)
        else:
            out[k] = float(v)
    return out


def parse_metropt_sensor_paste(raw: str) -> Optional[dict[str, Any]]:
    """
    Parse tab/line-separated sensor labels + numbers, or JSON with FEATURE_COLS keys.
    Returns a full row dict for predict_from_row, or None if this does not look like sensor data.
    """
    raw = raw.strip()
    if len(raw) < 6:
        return None

    parsed: dict[str, float] = {}

    if raw.startswith("{"):
        try:
            o = json.loads(raw)
        except json.JSONDecodeError:
            return None
        for k in FEATURE_COLS:
            if k in o and o[k] is not None:
                try:
                    parsed[k] = float(o[k])
                except (TypeError, ValueError):
                    pass
        if len(parsed) < 2:
            return None
        return fill_sensor_row(parsed)

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.+?)[\t:]\s*([0-9.]+)\s*$", line, re.I)
        if not m:
            m = re.match(r"^(.+?)\s+([0-9.]+)\s*$", line, re.I)
        if not m:
            continue
        key = label_to_sensor_key(m.group(1).strip())
        if key:
            parsed[key] = float(m.group(2))

    if len(parsed) < 2:
        for m in re.finditer(
            r"(?i)(oil\s*temperature|motor\s*current|tp2\s*pressure|tp3\s*pressure|tp2|tp3|h1|lps|comp|oil\s*level|compressor)\s*[:\t]?\s*([0-9.]+)",
            raw,
        ):
            key = label_to_sensor_key(m.group(1).strip())
            if key:
                parsed[key] = float(m.group(2))

    if len(parsed) < 2:
        return None
    if not any(
        k in parsed
        for k in ("Oil_temperature", "Motor_current", "TP2", "TP3", "COMP", "H1")
    ):
        return None
    return fill_sensor_row(parsed)


def _operator_sensor_snapshot(readings: dict[str, Any]) -> str:
    """One dense line of values as if from a shop terminal."""
    bits: list[str] = []
    for k in FEATURE_COLS:
        if k not in readings:
            continue
        v = readings[k]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if k == "Oil_temperature":
            bits.append(f"oil {fv:.1f}°C")
        elif k == "Motor_current":
            bits.append(f"motor {fv:.2f} A")
        elif k == "COMP":
            bits.append(f"COMP {fv:.0f}")
        elif k in ("TP2", "TP3"):
            bits.append(f"{k} {fv:.2f} bar")
        elif k == "H1":
            bits.append(f"H1 ΔP {fv:.2f} bar")
        elif k == "LPS":
            bits.append(f"LPS {fv:.0f}")
        elif k == "Oil_level":
            bits.append(f"oil lvl {fv:.0f}")
    return " · ".join(bits) if bits else "(no parsed channels)"


def metropt3_paste_intro_text(readings: dict[str, Any], pred: dict[str, Any]) -> str:
    """Plain-text lead when the user pasted sensor readings into /ask (simulated fleet screen)."""
    contribs = pred.get("contributions") or {}
    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    cls = pred.get("class_label", "Unknown")
    risk = float(pred.get("probability_failure_pct") or 0)
    if cls == "Normal" and risk < 25:
        lamp = "GREEN"
        status = "Inside normal envelope for this scoring model — no fault class triggered."
    elif cls == "Failure" or risk >= 40:
        lamp = "RED"
        status = "Outside normal envelope — matches the stressed / fault pattern in this training model."
    else:
        lamp = "AMBER"
        status = "Borderline — elevated score vs normal baseline; treat like a watch item in a real yard."

    lines = [
        "═══ SIMULATED FLEET HEALTH SCREEN · MetroPT-3 (training data only — not live DSNY telematics or OEM bus) ═══",
        f"Screen status lamp: {lamp} · Model class: {cls}",
        status,
        f"Snapshot: {_operator_sensor_snapshot(readings)}",
        f"Estimated fault probability (model, same-day window): {risk:.1f}%",
    ]
    df = pred.get("decision_function")
    if df is not None:
        lines.append(
            f"SVM decision margin: {df:.3f} (more positive → closer to the fault side of the hyperplane in this demo)."
        )
    if top:
        lines.append(
            "Strongest contributors on this pass (weighted score, not volts or PSI by themselves): "
            + ", ".join(f"{FEATURE_LABELS.get(k, k)} ({v:+.2f})" for k, v in top)
            + "."
        )
    return "\n".join(lines)


def metropt3_paste_operator_fallback(ctx: dict[str, Any], pred: dict[str, Any]) -> str:
    """When Gemini is off: simulated dispatch-style footer (still clearly not real authority)."""
    cls = pred.get("class_label", "Unknown")
    risk = float(pred.get("probability_failure_pct") or 0)
    if cls == "Normal" and risk < 25:
        next_block = (
            "Operational note (simulated): Condition Green for this screen — you’d typically finish the walk-around, "
            "confirm no MIL / audible alarms, and roll if fluids and belts match what you see here. "
            "Log the snapshot if your yard still uses paper or tablet sign-off for auxiliary equipment."
        )
    else:
        next_block = (
            "Operational note (simulated): Condition not cleared for release — you’d hold the unit, "
            "call it in on the radio, and have shop or a lead mechanic sign off before that truck goes back "
            "in collection service. Do not use this app as an out-of-service order."
        )
    return (
        "—\n"
        "AUTHORITY: SIMULATION ONLY. Not linked to DSNY dispatch, OEM ECU, ABS, or brake systems. "
        "Does not replace DOT inspection, yard safety rules, or manufacturer bulletins.\n"
        f"{next_block}\n"
        "Full gauges + scenario sliders: /metropt3"
    )


def load_dataset() -> pd.DataFrame:
    """
    Synthetic MetroPT-style table: heavily overlapping continuous features, similar digital priors,
    an ambiguous operating band (features barely informative of label), and registry-style noise.
    Class counts are balanced before label flips (normal == failure; ambiguous split 50/50).
    Designed so a holdout linear-SVM does not reach AUC=1 / perfect confusion matrix.
    """
    rng = np.random.default_rng(42)
    n0, n1, na = _DATASET_N_NORMAL, _DATASET_N_FAILURE, _DATASET_N_AMBIGUOUS

    def _block(n: int, mean_shift: float) -> pd.DataFrame:
        """Shared generator; small mean_shift separates classes only slightly (large overlap)."""
        return pd.DataFrame(
            {
                "Oil_temperature": np.clip(rng.normal(78.0 + mean_shift * 5.5, 14.5, n), 20.0, 120.0),
                "Motor_current": np.clip(rng.normal(7.2 + mean_shift * 1.8, 4.2, n), 0.0, 25.0),
                "COMP": rng.choice([0, 1], n, p=[0.26, 0.74]),
                "TP2": np.clip(rng.normal(7.4 + mean_shift * 0.9, 3.4, n), 0.0, 15.0),
                "TP3": np.clip(rng.normal(8.0 + mean_shift * 0.55, 3.0, n), 0.0, 15.0),
                "H1": np.clip(rng.normal(0.32 + mean_shift * 0.12, 0.38, n), 0.0, 2.0),
                "LPS": rng.choice([0, 1], n, p=[0.76, 0.24]),
                "Oil_level": rng.choice([0, 1], n, p=[0.16, 0.84]),
            }
        )

    normal = _block(n0, mean_shift=-0.35)
    normal["Status"] = 0
    failure = _block(n1, mean_shift=0.55)
    failure["Status"] = 1
    # Same feature law as mid-shift; labels balanced 0/1 (not random Bernoulli — avoids skew).
    amb = _block(na, mean_shift=0.12)
    half = na // 2
    amb_labels = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(na - half, dtype=np.int64)])
    rng.shuffle(amb_labels)
    amb["Status"] = amb_labels

    df = pd.concat([normal, failure, amb], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    flip = rng.random(len(df)) < _LABEL_NOISE_RATE
    df.loc[flip, "Status"] = 1 - df.loc[flip, "Status"].astype(int)
    return df


def _sigmoid(z: float) -> float:
    z = max(-50.0, min(50.0, float(z)))
    return float(1.0 / (1.0 + np.exp(-z)))


def train_svm(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURE_COLS]
    y = df["Status"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=_SVM_C,
                    class_weight="balanced",
                    dual=False,
                    max_iter=8000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_score = pipe.decision_function(X_te)

    cv = StratifiedKFold(n_splits=_CV_SPLITS, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    report = classification_report(y_te, y_pred, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    auc = float(roc_auc_score(y_te, y_score))

    svm_ = pipe.named_steps["svm"]
    w_approx = svm_.coef_.flatten()
    w_norm = w_approx / (np.abs(w_approx).sum() + 1e-9)

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    return {
        "pipeline": pipe,
        "report": report,
        "cm": cm,
        "auc": auc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feat_weights": dict(zip(FEATURE_COLS, w_norm.tolist())),
        "n_samples": int(len(X)),
        "class_counts": {"normal": n0, "failure": n1},
        "svm_C": _SVM_C,
        "kernel": _KERNEL,
    }


def _contrib_for_row(pipe: Pipeline, input_row: pd.DataFrame) -> dict[str, float]:
    scaler = pipe.named_steps["scaler"]
    svm_ = pipe.named_steps["svm"]
    xs = scaler.transform(input_row).flatten()
    w = svm_.coef_.flatten()
    return {FEATURE_COLS[j]: round(float(w[j] * xs[j]), 4) for j in range(len(FEATURE_COLS))}


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        df = load_dataset()
        _bundle = {"df": df, "svm": train_svm(df)}
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
    dec = float(pipe.decision_function(input_row)[0])
    prob = _sigmoid(dec)
    contribs = _contrib_for_row(pipe, input_row)
    return {
        "class": pred,
        "class_label": "Failure" if pred == 1 else "Normal",
        "probability_failure_pct": round(prob * 100, 1),
        "contributions": contribs,
        "decision_function": dec,
    }


def representative_sensor_scenarios() -> dict[str, Any]:
    typical_normal = {
        "Oil_temperature": 70.0,
        "Motor_current": 5.0,
        "COMP": 1.0,
        "TP2": 9.0,
        "TP3": 9.0,
        "H1": 0.1,
        "LPS": 0.0,
        "Oil_level": 1.0,
    }
    typical_stress = {
        "Oil_temperature": 95.0,
        "Motor_current": 12.0,
        "COMP": 1.0,
        "TP2": 4.0,
        "TP3": 8.0,
        "H1": 0.8,
        "LPS": 1.0,
        "Oil_level": 0.0,
    }
    return {
        "typical_normal": {"input": typical_normal, "prediction": predict_from_row(typical_normal)},
        "typical_stress": {"input": typical_stress, "prediction": predict_from_row(typical_stress)},
    }


def metropt3_chat_context() -> dict[str, Any]:
    """Compact JSON for /ask + Gemini (no full training matrix)."""
    b = get_bundle()
    svm_pkg = b["svm"]
    rep = svm_pkg["report"]
    cm = svm_pkg["cm"]
    fw = svm_pkg["feat_weights"]
    top_feats = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    df = b["df"]
    n_tot = len(df)
    n_ok = int((df["Status"] == 0).sum())
    n_bad = int((df["Status"] == 1).sum())
    return {
        "domain": "MetroPT-3 synthetic train air-production unit (demo—not live DSNY truck telemetry)",
        "dataset": {
            "total": n_tot,
            "normal": n_ok,
            "failure": n_bad,
            "normal_pct": round(100.0 * n_ok / n_tot, 1) if n_tot else 0.0,
            "failure_pct": round(100.0 * n_bad / n_tot, 1) if n_tot else 0.0,
        },
        "svm_model": {
            "type": "Linear-kernel SVM (class-balanced), only model in this demo",
            "roc_auc_holdout": round(float(svm_pkg["auc"]), 4),
            "roc_auc_cv_5fold_mean": round(float(svm_pkg["cv_mean"]), 4),
            "roc_auc_cv_5fold_std": round(float(svm_pkg["cv_std"]), 4),
            "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
            "failure_recall": round(float(rep["1"]["recall"]), 4),
            "failure_precision": round(float(rep["1"]["precision"]), 4),
            "confusion_matrix_test": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
            "confusion_note": "rows [actual normal, actual failure]; cols [predicted normal, predicted failure]",
            "top_svm_features_by_abs_weight": [{"feature": k, "weight": round(v, 4)} for k, v in top_feats],
        },
        "scenario_predictions": representative_sensor_scenarios(),
        "full_dashboard_path": "/metropt3",
    }


def metropt3_fallback_answer_text(ctx: dict[str, Any]) -> str:
    """Plain-text answer when Gemini is off or as backup."""
    s = ctx["svm_model"]
    n = ctx["scenario_predictions"]["typical_normal"]["prediction"]
    st = ctx["scenario_predictions"]["typical_stress"]["prediction"]
    tops = s.get("top_svm_features_by_abs_weight") or []
    top_s = ", ".join(f"{x['feature']} ({x['weight']:+.3f})" for x in tops[:3]) or "n/a"

    return (
        "Predictive maintenance demo (MetroPT-3 style synthetic sensors on an air-production unit). "
        "This is not live DSNY truck telematics.\n\n"
        f"Model: linear-kernel SVM only — ROC-AUC {s['roc_auc_holdout']:.3f} on holdout; "
        f"{_CV_SPLITS}-fold CV ROC-AUC {s['roc_auc_cv_5fold_mean']:.3f} ± {s['roc_auc_cv_5fold_std']:.3f}; "
        f"macro F1 {s['macro_f1']:.3f}; failure recall {s['failure_recall']:.3f}, precision {s['failure_precision']:.3f}.\n"
        f"Largest |weight| SVM features (approx.): {top_s}.\n"
        f"Test confusion matrix [[TN, FP],[FN, TP]]: {s['confusion_matrix_test']}. {s['confusion_note']}.\n\n"
        "Example readings (estimated failure probability):\n"
        f"  • Healthy-style sensors: {n['probability_failure_pct']:.1f}% failure risk → {n['class_label']}.\n"
        f"  • Stressed-style sensors: {st['probability_failure_pct']:.1f}% failure risk → {st['class_label']}.\n\n"
        "Charts and live sliders: open /metropt3 in this app."
    )


def metropt3_for_ask() -> tuple[dict[str, Any], str]:
    """One-shot bundle for Flask /ask."""
    ctx = metropt3_chat_context()
    return ctx, metropt3_fallback_answer_text(ctx)


def build_gemini_prompt(
    svm_metrics: dict[str, Any],
    current_input: Optional[dict[str, Any]] = None,
    prediction: Optional[dict[str, Any]] = None,
) -> str:
    rep = svm_metrics["report"]
    feat = svm_metrics["feat_weights"]
    n_tot = int(svm_metrics.get("n_samples", 0))
    cc = svm_metrics.get("class_counts") or {}
    n0 = int(cc.get("normal", 0))
    n1 = int(cc.get("failure", 0))
    p0 = 100.0 * n0 / n_tot if n_tot else 0.0
    p1 = 100.0 * n1 / n_tot if n_tot else 0.0
    c_svm = float(svm_metrics.get("svm_C", _SVM_C))
    kern = str(svm_metrics.get("kernel", _KERNEL))

    return textwrap.dedent(
        f"""
    You are a senior reliability engineer interpreting predictive-maintenance
    diagnostics for the MetroPT-3 train air-production unit dataset.

    The ONLY classifier in this pipeline is a linear-kernel SVM (balanced classes).

    Speak in clear, technical but accessible English. Structure your response as
    two short paragraphs plus a third if sensor data is provided:
      1. What the SVM is doing and which features drive the boundary (use weights below)
      2. Model performance and what it means operationally
      3. Current sensor reading interpretation (if provided)

    ── DATASET CONTEXT ─────────────────────────────────────────────────────
    Records: {n_tot}  |  Normal: {n0} ({p0:.1f} %)  |  Failure: {n1} ({p1:.1f} %)
    Features: Oil temperature, Motor current, Compressor flag (COMP),
              TP2 pressure, TP3 pressure, H1 pressure drop,
              Low-pressure switch (LPS), Oil level.

    ── SVM MODEL PERFORMANCE ───────────────────────────────────────────────
    Kernel         : {kern}  |  C={c_svm:g}  |  class_weight=balanced
    ROC-AUC        : {svm_metrics["auc"]:.4f}
    CV ROC-AUC     : {svm_metrics["cv_mean"]:.4f} ± {svm_metrics["cv_std"]:.4f}  ({_CV_SPLITS}-fold)
    Precision-0    : {rep["0"]["precision"]:.3f}   Recall-0 : {rep["0"]["recall"]:.3f}
    Precision-1    : {rep["1"]["precision"]:.3f}   Recall-1 : {rep["1"]["recall"]:.3f}
    Macro F1       : {rep["macro avg"]["f1-score"]:.3f}

    Feature weights (approximate, linear SVM coefficients, L1-normalized):
    {json.dumps({k: round(v, 4) for k, v in sorted(feat.items(), key=lambda x: abs(x[1]), reverse=True)}, indent=2)}
    Positive weight → pushes toward failure.
    Negative weight → pushes toward normal.

    ── CURRENT SENSOR READING ──────────────────────────────────────────────
    {json.dumps(current_input, indent=2) if current_input else "No live reading provided."}

    ── CURRENT PREDICTION ──────────────────────────────────────────────────
    {json.dumps(prediction, indent=2) if prediction else "No prediction requested."}

    ── YOUR TASK ────────────────────────────────────────────────────────────
    Write a concise diagnostic report (≤ 200 words). End with one actionable
    maintenance recommendation. Do NOT reproduce raw numbers verbatim — synthesise insight.
    """
    ).strip()


def diagnostics_payload() -> dict[str, Any]:
    b = get_bundle()
    df = b["df"]
    svm_pkg = b["svm"]
    cm = svm_pkg["cm"]
    rep = svm_pkg["report"]

    X_list = df[FEATURE_COLS].values.tolist()
    y_list = [int(x) for x in df["Status"].tolist()]

    return {
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "dataset": {
            "total": len(df),
            "normal": int((df.Status == 0).sum()),
            "failure": int((df.Status == 1).sum()),
        },
        "X": X_list,
        "y": y_list,
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
    prompt = build_gemini_prompt(b["svm"], current_input, prediction)
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
