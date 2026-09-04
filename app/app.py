"""Day 8 — Streamlit interface for the AI Risk Manager.

Pick a transaction, see the model's confidence, the routing decision
(auto-clear / hold-for-review / auto-escalate), the SHAP-based reason,
a counterfactual, and log a reviewer verdict — the full auto-responder
pipeline in one screen.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import VAL_PATH, TARGET_COL, MODELS_DIR, PROCESSED_DIR

REVIEW_LOG_PATH = PROCESSED_DIR / "review_log.csv"
ESCALATE_ABOVE = 0.80

st.set_page_config(
    page_title="AI Risk Manager — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- styling ----------

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .app-header {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a40;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; }
    .app-header p { margin: 0.25rem 0 0 0; color: #9a9ab0; font-size: 0.9rem; }

    .metric-card {
        background: #14141f;
        border: 1px solid #2a2a40;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: left;
    }
    .metric-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8a8aa0;
        margin-bottom: 0.35rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .badge-clear { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid #2ecc71; }
    .badge-hold { background: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid #f1c40f; }
    .badge-escalate { background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid #e74c3c; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2a2a40;
    }

    .cf-box {
        border-radius: 10px;
        padding: 1rem 1.1rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .cf-success { background: rgba(46, 204, 113, 0.1); border: 1px solid rgba(46, 204, 113, 0.4); }
    .cf-warning { background: rgba(241, 196, 15, 0.1); border: 1px solid rgba(241, 196, 15, 0.4); }
    .cf-info { background: rgba(52, 152, 219, 0.1); border: 1px solid rgba(52, 152, 219, 0.4); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_threshold():
    model = joblib.load(MODELS_DIR / "xgb_fraud_model.joblib")
    with open(MODELS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold


@st.cache_data
def load_val_data():
    df = pd.read_csv(VAL_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def route(prob, clear_below):
    if prob >= ESCALATE_ABOVE:
        return "AUTO-ESCALATE", "badge-escalate"
    elif prob >= clear_below:
        return "HOLD-FOR-REVIEW", "badge-hold"
    else:
        return "AUTO-CLEAR", "badge-clear"


def counterfactual_for_row(model, row, top_feature, threshold, direction=1):
    row = row.copy()
    original_value = row[top_feature]
    step = abs(original_value) * 0.05 + 0.01
    for i in range(1, 200):
        row[top_feature] = original_value - direction * step * i
        prob = model.predict_proba(row.to_frame().T)[0, 1]
        if prob < threshold:
            return row[top_feature] - original_value, prob
    return None, None


def log_reviewer_verdict(idx, prob, decision, top_feature, contribution, actual, verdict):
    row = pd.DataFrame([{
        "transaction_id": idx,
        "confidence": round(float(prob), 4),
        "decision": decision,
        "top_reason_feature": top_feature,
        "shap_contribution": round(float(contribution), 4),
        "ground_truth": actual,
        "reviewer_verdict": verdict,
    }])
    if REVIEW_LOG_PATH.exists():
        existing = pd.read_csv(REVIEW_LOG_PATH)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(REVIEW_LOG_PATH, index=False)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- header ----------

st.markdown("""
<div class="app-header">
    <h1>🛡️ AI Risk Manager — Fraud-Spike Detector</h1>
    <p>Razorpay AI Buildathon · Track 02 · XGBoost + cost-sensitive threshold
    + SHAP explainability + counterfactuals + human review loop</p>
</div>
""", unsafe_allow_html=True)

model, threshold = load_model_and_threshold()
X_val, y_val = load_val_data()
explainer = get_explainer(model)
all_probs = model.predict_proba(X_val)[:, 1]

with st.sidebar:
    st.subheader("Pipeline settings")
    metric_card("Hold-for-review threshold", f"{threshold:.2f}")
    st.write("")
    metric_card("Auto-escalate threshold", f"{ESCALATE_ABOVE:.2f}")
    st.divider()
    st.caption(
        "🟢 Below hold threshold → **auto-clear**\n\n"
        "🟡 Between thresholds → **hold for review**\n\n"
        "🔴 Above escalate threshold → **auto-escalate**"
    )
    st.divider()
    mode = st.radio("Pick a transaction", ["Random", "By ID", "Highest risk"])

if mode == "Random":
    idx = int(np.random.choice(X_val.index))
elif mode == "By ID":
    idx = st.sidebar.number_input(
        "Transaction ID", min_value=0, max_value=len(X_val) - 1, value=0
    )
else:
    idx = int(X_val.index[np.argmax(all_probs)])

row = X_val.loc[idx]
prob = model.predict_proba(row.to_frame().T)[0, 1]
decision, badge_class = route(prob, threshold)
actual = "FRAUD" if y_val.loc[idx] == 1 else "legit"

# ---------- summary row ----------

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Transaction ID", f"#{idx}")
with c2:
    metric_card("Model confidence", f"{prob:.2%}")
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="label">Routing decision</div>'
        f'<div style="margin-top:0.3rem;"><span class="badge {badge_class}">{decision}</span></div></div>',
        unsafe_allow_html=True,
    )
with c4:
    metric_card("Ground truth (val set)", actual)

st.write("")

# ---------- explanation ----------

left, right = st.columns([3, 2])

with left:
    st.markdown('<div class="section-title">Why this decision</div>', unsafe_allow_html=True)

    row_shap = explainer.shap_values(row.to_frame().T)[0]
    shap_df = pd.DataFrame({
        "feature": X_val.columns,
        "shap_value": row_shap,
    }).reindex(np.abs(row_shap).argsort()[::-1]).head(8)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_df["shap_value"]]
    ax.barh(shap_df["feature"][::-1], shap_df["shap_value"][::-1], color=colors[::-1])
    ax.set_xlabel("SHAP contribution  (red = toward fraud, blue = away from fraud)", color="#c0c0d0")
    ax.set_title(f"Top contributing features — transaction #{idx}", color="white")
    ax.tick_params(colors="#c0c0d0")
    for spine in ax.spines.values():
        spine.set_color("#2a2a40")
    st.pyplot(fig)

    top_feature = shap_df.iloc[0]["feature"]
    top_contribution = shap_df.iloc[0]["shap_value"]
    st.markdown(
        f"**Top reason flagged:** `{top_feature}` "
        f"(SHAP contribution: {top_contribution:+.3f})"
    )

with right:
    st.markdown('<div class="section-title">Counterfactual</div>', unsafe_allow_html=True)
    st.caption("What minimal change would have flipped this decision?")

    if decision == "AUTO-CLEAR":
        st.markdown(
            '<div class="cf-box cf-info">Already below the review threshold — '
            'no counterfactual needed.</div>',
            unsafe_allow_html=True,
        )
    else:
        change, new_prob = counterfactual_for_row(model, row, top_feature, threshold)
        if change is not None:
            st.markdown(
                f'<div class="cf-box cf-success">If <b>{top_feature}</b> had been '
                f'<b>{change:+.2f}</b> different, confidence would drop to '
                f'<b>{new_prob:.2%}</b> — below the review threshold.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="cf-box cf-warning">No single-feature change to '
                f'<b>{top_feature}</b> within a reasonable range clears this '
                f'transaction — confidence relies on multiple agreeing signals, '
                f'not just one.</div>',
                unsafe_allow_html=True,
            )

st.write("")

# ---------- reviewer step ----------

st.markdown('<div class="section-title">Reviewer verdict (audit trail)</div>', unsafe_allow_html=True)
st.caption("Mark whether you agree with this routing decision — this gets logged for the audit trail.")

verdict_col, button_col = st.columns([3, 1])
with verdict_col:
    verdict = st.radio(
        "Your verdict", ["agree", "disagree"], horizontal=True, key=f"verdict_{idx}",
        label_visibility="collapsed",
    )
with button_col:
    if st.button("Log verdict", use_container_width=True):
        log_reviewer_verdict(idx, prob, decision, top_feature, top_contribution, actual, verdict)
        st.success(f"Logged: #{idx} → {verdict}")

with st.expander("View full review log"):
    if REVIEW_LOG_PATH.exists():
        st.dataframe(pd.read_csv(REVIEW_LOG_PATH), use_container_width=True)
    else:
        st.info("No reviews logged yet.")
        