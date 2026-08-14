"""
Automotive Insurance Claims Risk Triage — Streamlit Dashboard
=============================================================
FSE 570 Capstone Project — Team Connecticut

Pages:
  1. Summary Dashboard  — KPIs, PR curve, SHAP importance, triage bucket chart
  2. Review Queue       — risk-ranked claim table with filters
  3. Claim Detail       — per-claim SHAP waterfall, reason codes, RAG brief
  4. Live Scoring       — score a brand-new claim in real time

Run:
    streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Triage",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Card-style metric boxes */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"]  { font-size: 0.78rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="stMetricValue"]  { font-size: 1.6rem; font-weight: 700; color: #111827; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem; }

/* Section headers */
h2, h3 { color: #111827 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e8ecf0; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Buttons */
.stButton > button[kind="primary"] {
    background: #2563eb; color: white; border: none;
    border-radius: 8px; padding: 10px 24px; font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #1d4ed8; }

/* Brief container */
.brief-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 8px;
    font-size: 0.9rem;
    line-height: 1.7;
}
.brief-box h2, .brief-box h3 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #1e3a5f !important;
    margin-top: 16px !important;
    margin-bottom: 6px !important;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}
.brief-box ul { margin: 4px 0 8px 16px; }
.brief-box li { margin-bottom: 4px; }

/* Reason code pills */
.rc-pill {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px;
    margin-bottom: 8px; font-size: 0.84rem; font-weight: 500;
    border: 1px solid rgba(0,0,0,0.06);
}
.rc-high { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.rc-low  { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }

/* Section divider label */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9ca3af;
    margin: 0 0 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ── paths ─────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
IMPROVEMENT   = os.path.join(ROOT, "outputs", "improvement")
METRICS       = os.path.join(ROOT, "outputs", "metrics")
SHAP_DIR      = os.path.join(ROOT, "outputs", "shap")
PLOTS_DIR     = os.path.join(ROOT, "outputs", "plots")

# ══════════════════════════════════════════════════════════════
# DATA LOADING  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import joblib
    path = os.path.join(IMPROVEMENT, "best_model_improved.joblib")
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_rag():
    from rag_pipeline import RAGPipeline
    return RAGPipeline()


@st.cache_data(show_spinner="Loading data…")
def load_all_data():
    """
    Load dataset, run feature engineering + preprocessing, score every claim.

    Uses the same builder as the training pipeline (data_pipeline), so the
    dashboard is guaranteed to reproduce the exact split and feature space the
    model was trained on.
    """
    from data_pipeline import build_model_dataset

    data = build_model_dataset()
    data["feat_names"] = data["feature_names"]
    return data


@st.cache_data(show_spinner="Scoring claims…")
def score_claims(_model, X_val_t, X_test_t):
    val_prob  = _model.predict_proba(X_val_t)[:, 1]
    test_prob = _model.predict_proba(X_test_t)[:, 1]
    return val_prob, test_prob


@st.cache_data
def load_metadata():
    with open(os.path.join(IMPROVEMENT, "model_metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_shap_importance(_model, X_test_t, feat_names):
    """Compute SHAP global importance on the TEST set (consistent with dashboard data)."""
    import shap
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_test_t)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature":       feat_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp_df


@st.cache_data
def load_model_comparison():
    return pd.read_csv(os.path.join(IMPROVEMENT, "model_comparison_improved.csv"))


@st.cache_data
def load_shap_values(_model, X_val_t):
    """Compute SHAP values (cached so they only run once)."""
    import shap
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_val_t)
    return explainer, shap_values


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def compute_test_roi(test_prob, y_test,
                     avg_fraud_loss=15000, siu_cost=500, manual_cost=100):
    """Compute ROI metrics directly from test set predictions."""
    from config import PCT_SIU, PCT_MANUAL
    y_arr = np.array(y_test)
    n = len(test_prob)
    siu_cut    = int(PCT_SIU * n)
    manual_cut = int((PCT_SIU + PCT_MANUAL) * n)
    sorted_idx = np.argsort(test_prob)[::-1]

    siu_fraud    = int(y_arr[sorted_idx[:siu_cut]].sum())
    manual_fraud = int(y_arr[sorted_idx[siu_cut:manual_cut]].sum())
    fraud_caught = siu_fraud + manual_fraud
    savings      = fraud_caught * avg_fraud_loss
    costs        = siu_cut * siu_cost + (manual_cut - siu_cut) * manual_cost
    net_benefit  = savings - costs
    roi_x        = savings / max(costs, 1)

    return {
        "fraud_caught": fraud_caught,
        "savings_usd":  savings,
        "costs_usd":    costs,
        "net_benefit":  net_benefit,
        "roi_x":        roi_x,
    }


def assign_bucket(score, pct_siu=0.05, pct_manual=0.15, all_scores=None):
    """
    Bucket a score against a reference population.

    Thresholds come from RANK positions, not np.percentile. Percentile
    interpolates between neighbouring scores, which put 78 claims in a "top 5%"
    bucket of 1,542 where the rank cut takes 77 — so the Review Queue reported
    309 flagged claims while the ROI figures were computed on 308.
    """
    if all_scores is not None:
        ordered = np.sort(np.asarray(all_scores))[::-1]
        n = len(ordered)
        siu_thresh    = ordered[max(0, int(pct_siu * n) - 1)]
        manual_thresh = ordered[max(0, int((pct_siu + pct_manual) * n) - 1)]
        if score >= siu_thresh:
            return "SIU"
        elif score >= manual_thresh:
            return "Manual Review"
        else:
            return "Approve"
    return "Unknown"


BUCKET_COLOR = {"SIU": "#e74c3c", "Manual Review": "#f39c12", "Approve": "#27ae60"}
BUCKET_EMOJI = {"SIU": "🔴", "Manual Review": "🟡", "Approve": "🟢"}


def make_reason_codes(shap_row, feat_names, cat_cols, top_n=5, feature_values=None):
    """
    Convert a SHAP row into human-readable reason codes.

    feature_values: the claim's transformed feature row. Required to phrase
    one-hot features correctly. A dummy like `BasePolicy_Liability` can carry a
    large SHAP value while being 0 for this claim — the model is saying "this
    claim is NOT liability, and that matters". Without the value, the code read
    "BasePolicy = 'Liability'", which contradicted the Claim Attributes table
    directly beside it (and the generated triage brief) whenever the claim was
    Collision or All Perils.
    """
    from feature_engineering import humanize_feature

    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    codes = []
    for i in top_idx:
        fname = feat_names[i]
        sv    = float(shap_row[i])
        direction = "High risk" if sv > 0 else "Low risk"
        strength  = abs(sv)
        intensity = "STRONG" if strength > 0.5 else "MODERATE" if strength > 0.2 else "MILD"

        value = None if feature_values is None else feature_values[i]
        decoded = humanize_feature(fname, cat_cols, value)

        codes.append({
            "label":     f"[{intensity}] {direction}: {decoded}",
            "technical": fname,      # kept for audit / debugging, shown muted
            "shap":      sv,
            "direction": direction,
            "intensity": intensity,
        })
    return codes


def row_values(X, i):
    """One row of a transformed matrix as a dense 1-D array (X may be sparse)."""
    row = X[i]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row).ravel()


def build_waterfall(shap_row, feat_names, base_value, top_n=10):
    """Build a Plotly waterfall chart from SHAP values."""
    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    labels  = [feat_names[i] for i in top_idx]
    values  = [float(shap_row[i]) for i in top_idx]
    colors  = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        # Without this, an outside label on a bar that reaches the axis edge is
        # clipped by the plot area — "-0.1395" rendered as "1395".
        cliponaxis=False,
    ))
    fig.update_layout(
        # XGBoost's TreeExplainer returns contributions in MARGIN (log-odds)
        # space, not probability — a single feature showing -2.60 is impossible
        # for a probability. base + sum(contributions) passes through a sigmoid
        # to give the displayed risk score.
        title=f"SHAP Feature Contributions (base={base_value:.4f}, log-odds)",
        xaxis_title="SHAP value — impact on log-odds of fraud (not probability)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar(metadata):
    with st.sidebar:
        st.markdown("## 🔍 Insurance Claims Fraud Triage")
        st.caption("FSE 570 Capstone · Team Connecticut")
        st.divider()

        page = st.radio(
            "Navigate",
            ["📊 Summary Dashboard", "📋 Review Queue",
             "🔎 Claim Detail", "⚡ Live Scoring",
             "⚖️ Fairness Analysis", "📡 Monitoring",
             "📅 Seasonality Analysis", "🗂️ Dataset Comparison"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Model**")
        st.caption(metadata.get("best_model_name", "—"))
        st.caption(f"Test PR-AUC: **{metadata['test_pr_auc']:.4f}**")
        st.caption(f"Val PR-AUC: **{metadata['val_pr_auc']:.4f}**")
        st.caption(f"Val ROC-AUC: **{metadata['val_roc_auc']:.4f}**")
        st.caption(f"Trained: {metadata.get('trained_at','')[:10]}")

        st.divider()
        st.caption("RAG briefs powered by GPT-4o-mini" if os.environ.get("OPENAI_API_KEY") else "RAG briefs using template fallback")

    return page


# ══════════════════════════════════════════════════════════════
# PAGE 1 — SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════

def page_summary(metadata, test_prob, y_test, shap_imp, model_comparison):
    # Takes only test_prob by design. It previously also received val_prob, and
    # two charts silently mixed the splits — a validation histogram under test
    # thresholds, and test counts over a validation denominator.
    st.title("📊 Summary Dashboard")
    st.caption("Real-time fraud risk overview — test set (1,542 claims, completely unseen during training)")

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        This page gives a high-level overview of how the fraud detection model is performing.

        - **SIU (Special Investigations Unit)** — the top 5% highest-risk claims. These are automatically escalated for a full field investigation before any payment is made.
        - **Manual Review** — the next 15% of claims. A human reviewer checks these before approving.
        - **Approve** — the remaining 80% of claims pass through automated processing.
        - **PR-AUC** — the primary model performance metric. Higher is better. It measures how well the model ranks fraud above non-fraud, accounting for the fact that only ~6% of claims are fraudulent.
        - **ROI** — the estimated return on investment from catching fraud. Calculated as: (fraud losses prevented) ÷ (investigation costs).
        - **SHAP importance** — shows which claim features most influence the model's fraud score. Longer bar = more influential.
        """)


    # Compute ROI from test set (consistent with all other test set metrics)
    roi = compute_test_roi(test_prob, y_test)
    all_scores = test_prob
    y_arr = np.array(y_test)

    # Buckets are assigned by RANK, not by a percentile threshold, so that this
    # page agrees with compute_test_roi(), compute_triage_summary() and
    # outputs/improvement/triage_summary.json. `>= np.percentile(x, 95)` returned
    # 78 SIU claims where the rank cut takes 77, so the KPI cards, the enrichment
    # chart and the confusion matrix were all describing a slightly different
    # bucket than the ROI row directly beneath them (4.51x vs 4.57x enrichment,
    # precision 0.269 vs 0.273).
    from config import PCT_SIU, PCT_MANUAL
    n_scored   = len(all_scores)
    siu_cut    = int(PCT_SIU * n_scored)
    manual_cut = int((PCT_SIU + PCT_MANUAL) * n_scored)
    rank_of    = np.empty(n_scored, dtype=int)
    rank_of[np.argsort(all_scores)[::-1]] = np.arange(n_scored)

    siu_mask     = rank_of < siu_cut
    manual_mask  = (rank_of >= siu_cut) & (rank_of < manual_cut)
    approve_mask = rank_of >= manual_cut

    # Score at the bucket boundaries — for display and for the operational
    # threshold used by the confusion matrix below.
    ordered       = np.sort(all_scores)[::-1]
    siu_thresh    = float(ordered[siu_cut - 1])
    manual_thresh = float(ordered[manual_cut - 1])

    # ── KPI row ──────────────────────────────────────────────
    # Denominators must be the TEST split — the masks above come from test_prob.
    # (val and test are both 1,542 today, so this was right by coincidence.)
    n_claims = len(test_prob)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Claims",     f"{n_claims:,}")
    c2.metric("🔴 SIU",           f"{siu_mask.sum():,}",
              delta=f"{siu_mask.sum()/n_claims*100:.1f}% of claims", delta_color="off")
    c3.metric("🟡 Manual Review", f"{manual_mask.sum():,}",
              delta=f"{manual_mask.sum()/n_claims*100:.1f}% of claims", delta_color="off")
    c4.metric("🟢 Approve",       f"{approve_mask.sum():,}",
              delta=f"{approve_mask.sum()/n_claims*100:.1f}% of claims", delta_color="off")
    c5.metric("Test PR-AUC",      f"{metadata['test_pr_auc']:.4f}",
              delta=f"Val PR-AUC {metadata['val_pr_auc']:.4f}", delta_color="off")
    c6.metric("Net ROI",          f"{roi['roi_x']:.1f}x",
              delta=f"${roi['net_benefit']:,.0f} net benefit", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: PR curve + model comparison ───────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
        st.markdown("**Precision-Recall Curve**")
        from sklearn.metrics import precision_recall_curve, average_precision_score
        prec, rec, thresholds = precision_recall_curve(y_arr, test_prob)
        pr_auc = average_precision_score(y_arr, test_prob)

        # Smooth the curve with a rolling average for cleaner display
        window = 20
        prec_s = pd.Series(prec).rolling(window, min_periods=1, center=True).mean().values
        rec_s  = pd.Series(rec).rolling(window, min_periods=1, center=True).mean().values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rec_s, y=prec_s, mode="lines", fill="tozeroy",
            fillcolor="rgba(37,99,235,0.08)",
            name=f"XGBoost Optuna (PR-AUC = {pr_auc:.4f})",
            line=dict(color="#2563eb", width=2.5),
        ))
        fig.add_hline(y=float(y_arr.mean()), line_dash="dot", line_color="#9ca3af",
                      annotation_text=f"Random baseline ({y_arr.mean()*100:.1f}%)",
                      annotation_font_size=11)
        fig.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            xaxis=dict(range=[0, 1], gridcolor="#f3f4f6"),
            yaxis=dict(range=[0, 1], gridcolor="#f3f4f6"),
            height=340, margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.5, y=0.95, xanchor="center",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#e5e7eb", borderwidth=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-label">Model Comparison</p>', unsafe_allow_html=True)
        st.markdown("**All Models — Validation Set** *(used for model selection)*")
        comp = model_comparison[["Model","PR_AUC","ROC_AUC",
                                  "Precision_at_5pct","Recall_at_5pct"]].copy()
        comp.columns = ["Model","PR-AUC","ROC-AUC","Prec@5%","Rec@5%"]
        st.dataframe(
            comp.style
                .highlight_max(subset=["PR-AUC","ROC-AUC","Prec@5%","Rec@5%"],
                               color="#dbeafe")
                .format({"PR-AUC":"{:.4f}","ROC-AUC":"{:.4f}",
                         "Prec@5%":"{:.4f}","Rec@5%":"{:.4f}"}),
            use_container_width=True, hide_index=True, height=180,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Risk Score Distribution** *(test set)*")
        # Must be test_prob: the SIU/Manual cut lines below are percentiles of
        # test_prob, so plotting val_prob here drew test thresholds over a
        # validation histogram.
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=test_prob, nbinsx=40,
            marker_color="#2563eb", opacity=0.7,
            name="All claims",
        ))
        fig2.add_vline(x=float(siu_thresh), line_dash="dash",
                       line_color="#e74c3c",
                       annotation_text="SIU", annotation_font_color="#e74c3c")
        fig2.add_vline(x=float(manual_thresh), line_dash="dash",
                       line_color="#f39c12",
                       annotation_text="Manual", annotation_font_color="#f39c12")
        fig2.update_layout(
            xaxis_title="Risk Score", yaxis_title="Count",
            height=160, margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False,
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: SHAP importance + triage bucket chart ─────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-label">Explainability</p>', unsafe_allow_html=True)
        st.markdown("**Top 15 Global Feature Importances (SHAP)**")
        top15 = shap_imp[shap_imp["mean_abs_shap"] > 0].head(15).copy()
        # Clean up feature names for display
        top15["feature_display"] = top15["feature"].str.replace("_", " ").str.replace("  ", " ")
        fig = go.Figure(go.Bar(
            x=top15["mean_abs_shap"].values[::-1],
            y=top15["feature_display"].values[::-1],
            orientation="h",
            marker=dict(
                color=top15["mean_abs_shap"].values[::-1],
                colorscale=[[0, "#bfdbfe"], [1, "#1d4ed8"]],
            ),
            text=[f"{v:.4f}" for v in top15["mean_abs_shap"].values[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Mean |SHAP value|",
            height=420, margin=dict(l=10, r=60, t=10, b=10),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-label">Triage Performance</p>', unsafe_allow_html=True)
        st.markdown("**Fraud Rate by Triage Bucket**")
        buckets = ["SIU", "Manual Review", "Approve"]
        fraud_rates = [
            float(y_arr[siu_mask].mean()),
            float(y_arr[manual_mask].mean()),
            float(y_arr[approve_mask].mean()),
        ]
        counts = [int(siu_mask.sum()), int(manual_mask.sum()), int(approve_mask.sum())]
        enrichments = [r / float(y_arr.mean()) for r in fraud_rates]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=buckets, y=[r * 100 for r in fraud_rates],
            marker_color=[BUCKET_COLOR[b] for b in buckets],
            text=[f"<b>{r*100:.1f}%</b><br>{c} claims<br>{e:.1f}x enrichment"
                  for r, c, e in zip(fraud_rates, counts, enrichments)],
            textposition="outside",
            width=0.5,
        ))
        fig.add_hline(y=float(y_arr.mean()) * 100, line_dash="dot",
                      line_color="#9ca3af",
                      annotation_text=f"Base rate {y_arr.mean()*100:.1f}%",
                      annotation_font_size=11)
        fig.update_layout(
            yaxis_title="Fraud Rate (%)",
            yaxis=dict(range=[0, max(fraud_rates)*130], gridcolor="#f3f4f6"),
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROI summary ───────────────────────────────────────────
    st.markdown('<p class="section-label">Business Impact</p>', unsafe_allow_html=True)
    st.markdown("**Cost-Benefit ROI Analysis**")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Fraud Claims Caught",  f"{roi['fraud_caught']}")
    r2.metric("Losses Prevented",     f"${roi['savings_usd']:,.0f}")
    r3.metric("Investigation Costs",  f"${roi['costs_usd']:,.0f}")
    r4.metric("Net Benefit",          f"${roi['net_benefit']:,.0f}")
    r5.metric("Return on Investment", f"{roi['roi_x']:.1f}x")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrix + calibration curve ─────────────────
    st.markdown('<p class="section-label">Model Evaluation</p>', unsafe_allow_html=True)
    col_cm, col_cal = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix @ Operational Threshold (Top 5%)**")
        from sklearn.metrics import confusion_matrix as sk_cm
        # Use the same rank-based SIU mask as the KPI cards and the ROI figures,
        # so TP + FP equals the SIU count shown above rather than 78 vs 77.
        op_thresh  = siu_thresh
        y_pred_op  = siu_mask.astype(int)
        cm         = sk_cm(y_arr, y_pred_op)
        tn, fp, fn, tp = cm.ravel()

        cm_df = pd.DataFrame(
            [[f"TN: {tn}", f"FP: {fp}"],
             [f"FN: {fn}", f"TP: {tp}"]],
            index=["Actual: Legit", "Actual: Fraud"],
            columns=["Predicted: Legit", "Predicted: Fraud"],
        )
        st.dataframe(
            cm_df.style.applymap(
                lambda v: "background-color: #dbeafe" if v.startswith("TP") or v.startswith("TN")
                else "background-color: #fee2e2"
            ),
            use_container_width=True,
        )
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        st.caption(
            f"Precision: {precision:.3f} | Recall: {recall:.3f} | "
            f"Threshold: top 5% (score ≥ {op_thresh:.4f})"
        )

    with col_cal:
        st.markdown("**Calibration Curve (Reliability Diagram)**")
        st.caption(
            "The curve sits **below** the diagonal by design, not by accident: the "
            f"model is trained with `scale_pos_weight≈{15.7:.1f}` to counter a 6% "
            "base rate, which deliberately inflates the raw scores. A claim scored "
            "0.80 is not 80% likely to be fraud — it is *high risk relative to other "
            "claims*. Triage uses **rank** (top 5% / next 15%), so this does not "
            "affect any bucket assignment; it would only matter if the score were "
            "read as a literal probability."
        )
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(y_arr, all_scores, n_bins=10)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos,
            mode="lines+markers",
            name="Model",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=7),
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#9ca3af", dash="dash"),
        ))
        fig_cal.update_layout(
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
            height=280,
            margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 1], gridcolor="#f3f4f6"),
            yaxis=dict(range=[0, 1], gridcolor="#f3f4f6"),
            legend=dict(x=0.05, y=0.95),
        )
        st.plotly_chart(fig_cal, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — REVIEW QUEUE
# ══════════════════════════════════════════════════════════════

def page_queue(df_raw, test_prob, y_test):
    st.title("📋 Review Queue")
    st.caption("Claims ranked by fraud risk score — highest risk first.")

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        This is the working queue an insurance investigator would use every day.

        - Claims are **ranked by risk score** — the highest-risk claims appear at the top.
        - **🔴 SIU** claims (top 5%) should be investigated before any payment is made.
        - **🟡 Manual Review** claims (next 15%) need a human to verify the details.
        - **🟢 Approve** claims (bottom 80%) can be processed automatically.
        - Use the **filters on the left** to focus on a specific bucket or score range.
        - Click **View Claim →** at the bottom to jump to the full detail view for any claim.
        - **Actual Fraud** column shows the ground truth — useful for evaluating model accuracy.
        """)


    all_scores = test_prob
    y_arr      = np.array(y_test)

    # Build display dataframe
    queue = pd.DataFrame({
        "Risk Score":    test_prob,
        "Actual Fraud":  y_arr,
    }).reset_index(drop=True)
    queue["Rank"] = queue["Risk Score"].rank(ascending=False).astype(int)
    queue["Triage Bucket"] = queue["Risk Score"].apply(
        lambda s: assign_bucket(s, all_scores=all_scores)
    )
    queue["Bucket Icon"] = queue["Triage Bucket"].map(BUCKET_EMOJI)

    # Add raw claim fields for display
    raw_reset = df_raw.iloc[y_test.index].reset_index(drop=True)
    for col in ["PolicyNumber", "Age", "BasePolicy", "VehicleCategory",
                "Fault", "PoliceReportFiled", "WitnessPresent", "AgentType",
                "Deductible", "AddressChange_Claim"]:
        if col in raw_reset.columns:
            queue[col] = raw_reset[col].values

    queue = queue.sort_values("Risk Score", ascending=False).reset_index(drop=True)

    # ── Sidebar filters ───────────────────────────────────────
    with st.sidebar:
        st.markdown("**Queue Filters**")
        bucket_filter = st.multiselect(
            "Triage Bucket",
            ["SIU", "Manual Review", "Approve"],
            default=["SIU", "Manual Review"],
        )
        score_range = st.slider("Risk Score Range", 0.0, 1.0, (0.0, 1.0), 0.01)

    filtered = queue[
        (queue["Triage Bucket"].isin(bucket_filter)) &
        (queue["Risk Score"] >= score_range[0]) &
        (queue["Risk Score"] <= score_range[1])
    ]

    st.caption(f"Showing {len(filtered):,} of {len(queue):,} claims")

    # ── Display columns ───────────────────────────────────────
    display_cols = ["Rank", "Bucket Icon", "Triage Bucket", "Risk Score",
                    "Actual Fraud", "Age", "BasePolicy", "VehicleCategory",
                    "Fault", "PoliceReportFiled", "WitnessPresent"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    def color_row(row):
        color = {"SIU": "#fde8e8", "Manual Review": "#fef9e7", "Approve": "#eafaf1"}
        bg = color.get(row["Triage Bucket"], "")
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(
        filtered[display_cols].style
            .apply(color_row, axis=1)
            .format({"Risk Score": "{:.4f}"}),
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    # ── Claim selector for detail view ────────────────────────
    st.divider()
    st.markdown("**Jump to Claim Detail**")
    selected_rank = st.number_input(
        "Enter Rank", min_value=1, max_value=len(queue), value=1, step=1
    )
    if st.button("View Claim →"):
        st.session_state["selected_rank"] = int(selected_rank)
        st.session_state["page_override"] = "🔎 Claim Detail"
        st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CLAIM DETAIL
# ══════════════════════════════════════════════════════════════

def page_detail(df_raw, test_prob, y_test, shap_values, explainer,
                feat_names, cat_cols, rag, X_test_t):
    st.title("🔎 Claim Detail")

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        This page gives a full breakdown of a single claim — why the model scored it the way it did.

        - **Risk Score** — the model's estimated probability that this claim is fraudulent (0 = definitely legitimate, 1 = definitely fraud).
        - **Triage Bucket** — which investigation tier this claim falls into based on its score.
        - **SHAP waterfall chart** — shows exactly which features pushed the score up (red = increases fraud risk) or down (green = decreases fraud risk). This is the "why" behind the score.
        - **Reason Codes** — plain-English summary of the top risk drivers, suitable for a non-technical investigator.
        - **Triage Brief** — click the button to generate an AI-written investigation brief that cites specific internal fraud guidelines. Works without an OpenAI key using the template mode.
        - Use the **Rank** input to navigate between claims (Rank 1 = highest risk claim).
        """)


    all_scores = test_prob
    sorted_idx = np.argsort(test_prob)[::-1]

    rank = st.session_state.get("selected_rank", 1)
    rank = st.number_input("Claim Rank", min_value=1,
                            max_value=len(test_prob), value=rank, step=1)
    st.session_state["selected_rank"] = rank

    claim_pos  = sorted_idx[rank - 1]
    risk_score = float(test_prob[claim_pos])
    actual     = int(np.array(y_test)[claim_pos])
    bucket     = assign_bucket(risk_score, all_scores=all_scores)

    # Raw claim data
    raw_idx   = y_test.index[claim_pos]
    claim_row = df_raw.loc[raw_idx]

    # ── Header ────────────────────────────────────────────────
    bucket_bg = {"SIU": "#fef2f2", "Manual Review": "#fffbeb", "Approve": "#f0fdf4"}
    bucket_border = {"SIU": "#fca5a5", "Manual Review": "#fcd34d", "Approve": "#86efac"}
    st.markdown(
        f'<div style="background:{bucket_bg[bucket]};border:1px solid {bucket_border[bucket]};'
        f'border-radius:10px;padding:16px 24px;margin-bottom:16px;display:flex;'
        f'align-items:center;gap:24px;">'
        f'<div><div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#6b7280;">Risk Score</div>'
        f'<div style="font-size:2rem;font-weight:800;color:{BUCKET_COLOR[bucket]};">'
        f'{risk_score:.4f}</div></div>'
        f'<div><div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#6b7280;">Triage Bucket</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:{BUCKET_COLOR[bucket]};">'
        f'{BUCKET_EMOJI[bucket]} {bucket}</div></div>'
        f'<div><div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#6b7280;">Actual Fraud</div>'
        f'<div style="font-size:1.4rem;font-weight:700;">'
        f'{"✅ Yes" if actual else "❌ No"}</div></div>'
        f'<div style="margin-left:auto;text-align:right;">'
        f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#6b7280;">Rank</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#374151;">'
        f'#{rank} of {len(test_prob)}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Two-column layout ─────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Claim Attributes")
        display_fields = [
            "PolicyNumber", "Age", "Sex", "MaritalStatus",
            "BasePolicy", "PolicyType", "VehicleCategory", "VehiclePrice",
            "AgeOfVehicle", "Fault", "PoliceReportFiled", "WitnessPresent",
            "AgentType", "Deductible", "AddressChange_Claim",
            "PastNumberOfClaims", "NumberOfSuppliments",
            "Days_Policy_Accident", "Days_Policy_Claim",
            "Month", "DayOfWeek", "MonthClaimed", "DayOfWeekClaimed",
        ]
        rows = {f: claim_row.get(f, "—") for f in display_fields if f in claim_row.index}
        st.dataframe(
            pd.DataFrame(rows.items(), columns=["Field", "Value"]),
            use_container_width=True, hide_index=True, height=420,
        )

    with right:
        st.subheader("SHAP Feature Contributions")
        shap_row = shap_values[claim_pos]
        fig = build_waterfall(shap_row, feat_names,
                               float(explainer.expected_value))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-label">Top Risk Drivers</p>', unsafe_allow_html=True)
        st.markdown("**Reason Codes**")
        codes = make_reason_codes(shap_row, feat_names, cat_cols, top_n=5,
                                  feature_values=row_values(X_test_t, claim_pos))
        for c in codes:
            css_class = "rc-high" if c["direction"] == "High risk" else "rc-low"
            icon = "⬆" if c["direction"] == "High risk" else "⬇"
            badge = f'<span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;opacity:0.7;">[{c["intensity"]}]</span>'
            st.markdown(
                f'<div class="rc-pill {css_class}">'
                f'<span style="font-size:1rem;">{icon}</span>'
                f'<span>{badge} {c["label"].split("] ", 1)[-1]}</span>'
                f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.55;'
                f'font-family:ui-monospace,monospace;">{c["technical"]}</span>'
                f'<span style="margin-left:12px;font-size:0.78rem;opacity:0.7;">'
                f'SHAP: {c["shap"]:+.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── RAG Triage Brief ──────────────────────────────────────
    st.markdown('<p class="section-label">AI-Generated Investigation Brief</p>',
                unsafe_allow_html=True)
    st.markdown("**📄 Triage Brief**")

    claim_data = {
        k: str(claim_row[k])
        for k in ["BasePolicy", "Fault", "VehicleCategory", "VehiclePrice",
                  "PoliceReportFiled", "WitnessPresent", "AgentType",
                  "AddressChange_Claim", "Deductible", "Age",
                  "PastNumberOfClaims"]
        if k in claim_row.index
    }
    reason_code_strs = [c["label"] for c in codes]

    if st.button("Generate Triage Brief", type="primary"):
        with st.spinner("Retrieving guidelines and generating brief…"):
            result = rag.generate(
                claim_data=claim_data,
                reason_codes=reason_code_strs,
                risk_score=risk_score,
                triage_bucket=bucket,
            )
        method_label = "🤖 GPT-4o-mini" if result["method"] == "llm" else "📋 Template"
        st.caption(f"Generated via: {method_label}")

        with st.expander("📚 Retrieved Guideline Passages", expanded=False):
            for i, p in enumerate(result["passages"], 1):
                st.markdown(f"**[Source {i}: `{p['source']}`]**")
                st.caption(p["text"][:300] + "…")
                if i < len(result["passages"]):
                    st.divider()

        # Render brief in styled container
        st.markdown(
            f'<div class="brief-box">{result["brief"]}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — LIVE SCORING
# ══════════════════════════════════════════════════════════════

def page_live(model, preprocessor, cat_cols, num_cols, feat_names, rag, all_scores):
    st.title("⚡ Live Claim Scoring")
    st.caption("Enter a new claim's details to get an instant fraud risk score, "
               "SHAP explanation, and triage brief.")

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        This page lets you score a **brand new claim** that was never part of the training or test data.

        - Fill in the claim details using the form below and click **Score This Claim →**.
        - The model instantly returns a **fraud risk score**, **triage bucket**, and **SHAP explanation**.
        - The **gauge chart** shows the score as a percentage compared to the 6% base fraud rate.
        - **Reason codes** explain in plain English which factors drove the score up or down.
        - A **triage brief** is automatically generated with recommended investigation steps.
        - Try different combinations — for example, set Base Policy to "Liability", Police Report to "No", and Witness to "No" to see a high-risk scenario.
        """)


    from feature_engineering import (
        DAYS_POLICY_MAP, PAST_CLAIMS_MAP, AGE_VEHICLE_MAP,
        AGE_POLICY_HOLDER_MAP, NUM_SUPPLIMENTS_MAP, NUM_CARS_MAP,
        ADDRESS_CHANGE_MAP, VEHICLE_PRICE_MAP, MONTH_MAP, DOW_MAP,
    )

    with st.form("live_claim_form"):
        st.markdown("### Policy & Coverage")
        c1, c2, c3 = st.columns(3)
        base_policy    = c1.selectbox("Base Policy",
                            ["Liability", "Collision", "All Perils"])
        vehicle_cat    = c2.selectbox("Vehicle Category",
                            ["Sedan", "Sport", "Utility"])
        vehicle_price  = c3.selectbox("Vehicle Price",
                            list(VEHICLE_PRICE_MAP.keys()))

        st.markdown("### Claim Details")
        c4, c5, c6 = st.columns(3)
        fault          = c4.selectbox("Fault",
                            ["Policy Holder", "Third Party"])
        police_report  = c5.selectbox("Police Report Filed", ["No", "Yes"])
        witness        = c6.selectbox("Witness Present",     ["No", "Yes"])

        c7, c8, c9 = st.columns(3)
        agent_type     = c7.selectbox("Agent Type", ["External", "Internal"])
        address_change = c8.selectbox("Address Change (before claim)",
                            list(ADDRESS_CHANGE_MAP.keys()))
        deductible     = c9.selectbox("Deductible", [300, 400, 500, 700])

        st.markdown("### Policyholder")
        c10, c11, c12 = st.columns(3)
        age            = c10.slider("Age", 16, 80, 35)
        past_claims    = c11.selectbox("Past Number of Claims",
                            list(PAST_CLAIMS_MAP.keys()))
        num_supps      = c12.selectbox("Number of Supplements",
                            list(NUM_SUPPLIMENTS_MAP.keys()))

        c19, c20, c21 = st.columns(3)
        sex            = c19.selectbox("Sex", ["Male", "Female"])
        marital        = c20.selectbox("Marital Status",
                            ["Single", "Married", "Widow", "Divorced"])
        accident_area  = c21.selectbox("Accident Area", ["Urban", "Rural"])

        c22, _, _ = st.columns(3)
        age_vehicle    = c22.selectbox("Age of Vehicle",
                            list(AGE_VEHICLE_MAP.keys()), index=2)

        st.markdown("### Timing")
        c13, c14, c15 = st.columns(3)
        month          = c13.selectbox("Accident Month", list(MONTH_MAP.keys()))
        dow            = c14.selectbox("Accident Day of Week", list(DOW_MAP.keys()))
        days_policy    = c15.selectbox("Days Since Policy (accident)",
                            list(DAYS_POLICY_MAP.keys()))

        c16, c17, c18 = st.columns(3)
        month_claimed  = c16.selectbox("Claim Month", list(MONTH_MAP.keys()))
        dow_claimed    = c17.selectbox("Claim Day of Week", list(DOW_MAP.keys()))
        days_claim     = c18.selectbox("Days Since Policy (claim)",
                            list(DAYS_POLICY_MAP.keys()))

        submitted = st.form_submit_button("Score This Claim →", type="primary")

    if not submitted:
        return

    # Build a single-row DataFrame matching the raw dataset schema.
    # AgeOfPolicyHolder is DERIVED from the Age slider — hardcoding it meant an
    # age-70 claim was scored with a policyholder band of "26 to 30".
    # Year comes from config (training data is 1994-96; 2024 was out of range).
    from config import LIVE_SCORING_YEAR
    from feature_engineering import derive_age_band

    ASSUMED = {                     # not exposed in the form — see caption below
        "Make": "Honda", "RepNumber": 5, "DriverRating": 3,
        "WeekOfMonth": 2, "WeekOfMonthClaimed": 2, "NumberOfCars": "1 vehicle",
    }

    row = {
        "WeekOfMonth": ASSUMED["WeekOfMonth"],
        "WeekOfMonthClaimed": ASSUMED["WeekOfMonthClaimed"],
        "Age": age, "RepNumber": ASSUMED["RepNumber"],
        "Deductible": deductible, "DriverRating": ASSUMED["DriverRating"],
        "Year": LIVE_SCORING_YEAR,
        "Month": month, "DayOfWeek": dow,
        "Make": ASSUMED["Make"], "AccidentArea": accident_area,
        "DayOfWeekClaimed": dow_claimed, "MonthClaimed": month_claimed,
        "Sex": sex, "MaritalStatus": marital,
        "Fault": fault,
        "PolicyType": f"{vehicle_cat} - {base_policy}",
        "VehicleCategory": vehicle_cat,
        "VehiclePrice": vehicle_price,
        "Days_Policy_Accident": days_policy,
        "Days_Policy_Claim": days_claim,
        "PastNumberOfClaims": past_claims,
        "AgeOfVehicle": age_vehicle,
        "AgeOfPolicyHolder": derive_age_band(age),
        "PoliceReportFiled": police_report,
        "WitnessPresent": witness,
        "AgentType": agent_type,
        "NumberOfSuppliments": num_supps,
        "AddressChange_Claim": address_change,
        "NumberOfCars": ASSUMED["NumberOfCars"],
        "BasePolicy": base_policy,
        "FraudFound_P": 0,
        "PolicyNumber": 999999,
    }

    from feature_engineering import engineer_features, REPLACED_CATEGORICALS
    df_single = pd.DataFrame([row])
    df_eng    = engineer_features(df_single)

    cols_to_remove = [c for c in REPLACED_CATEGORICALS if c in df_eng.columns]
    X_single = df_eng.drop(columns=["FraudFound_P", "PolicyNumber"] + cols_to_remove,
                            errors="ignore")

    # Align columns to the training feature space. Missing categoricals must be
    # filled with "Unknown" (the value the fitted imputer expects) — filling them
    # with 0 would push an unseen numeric through the one-hot encoder.
    for col in cat_cols:
        if col not in X_single.columns:
            X_single[col] = "Unknown"
    for col in num_cols:
        if col not in X_single.columns:
            X_single[col] = 0
    X_single = X_single[cat_cols + num_cols]

    st.caption(
        "Assumed for fields not on this form — "
        + ", ".join(f"{k}: {v}" for k, v in ASSUMED.items())
        + f", Year: {LIVE_SCORING_YEAR}. "
        "AgeOfPolicyHolder is derived from the Age slider."
    )

    X_t       = preprocessor.transform(X_single)
    risk_score = float(model.predict_proba(X_t)[0, 1])
    bucket     = assign_bucket(risk_score, all_scores=all_scores)

    # ── Results ───────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">Scoring Result</p>', unsafe_allow_html=True)

    bucket_bg     = {"SIU": "#fef2f2", "Manual Review": "#fffbeb", "Approve": "#f0fdf4"}
    bucket_border = {"SIU": "#fca5a5", "Manual Review": "#fcd34d", "Approve": "#86efac"}

    m1, m2 = st.columns([1, 2])
    with m1:
        st.markdown(
            f'<div style="background:{bucket_bg[bucket]};border:2px solid {bucket_border[bucket]};'
            f'border-radius:12px;padding:24px;text-align:center;">'
            f'<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.06em;color:#6b7280;margin-bottom:8px;">Fraud Risk Score</div>'
            f'<div style="font-size:3rem;font-weight:800;color:{BUCKET_COLOR[bucket]};">'
            f'{risk_score:.4f}</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:{BUCKET_COLOR[bucket]};'
            f'margin-top:8px;">{BUCKET_EMOJI[bucket]} {bucket}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with m2:
        risk_pct = risk_score * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            delta={"reference": 6.0, "valueformat": ".1f",
                   "suffix": "% vs base rate"},
            number={"suffix": "%", "font": {"size": 36, "color": BUCKET_COLOR[bucket]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#9ca3af", "tickfont": {"size": 11}},
                "bar":  {"color": BUCKET_COLOR[bucket], "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20],  "color": "#f0fdf4"},
                    {"range": [20, 50], "color": "#fffbeb"},
                    {"range": [50, 100],"color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": "#374151", "width": 2},
                    "thickness": 0.75,
                    "value": risk_pct,
                },
            },
            title={"text": "Fraud Probability", "font": {"size": 14, "color": "#6b7280"}},
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)
    shap_row    = shap_values[0]

    st.subheader("SHAP Explanation")
    fig = build_waterfall(shap_row, feat_names, float(explainer.expected_value))
    st.plotly_chart(fig, use_container_width=True)

    codes = make_reason_codes(shap_row, feat_names, cat_cols, top_n=5,
                              feature_values=row_values(X_t, 0))
    st.markdown('<p class="section-label">Top Risk Drivers</p>', unsafe_allow_html=True)
    st.markdown("**Reason Codes**")
    for c in codes:
        css_class = "rc-high" if c["direction"] == "High risk" else "rc-low"
        icon = "⬆" if c["direction"] == "High risk" else "⬇"
        badge = f'<span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;opacity:0.7;">[{c["intensity"]}]</span>'
        st.markdown(
            f'<div class="rc-pill {css_class}">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span>{badge} {c["label"].split("] ", 1)[-1]}</span>'
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.55;'
            f'font-family:ui-monospace,monospace;">{c["technical"]}</span>'
            f'<span style="margin-left:12px;font-size:0.78rem;opacity:0.7;">'
            f'SHAP: {c["shap"]:+.4f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # RAG brief
    st.divider()
    st.markdown('<p class="section-label">AI-Generated Investigation Brief</p>',
                unsafe_allow_html=True)
    st.markdown("**📄 Triage Brief**")
    claim_data = {
        "BasePolicy": base_policy, "Fault": fault,
        "VehicleCategory": vehicle_cat, "VehiclePrice": vehicle_price,
        "PoliceReportFiled": police_report, "WitnessPresent": witness,
        "AgentType": agent_type, "AddressChange_Claim": address_change,
        "Deductible": deductible, "Age": age,
        "PastNumberOfClaims": past_claims,
    }
    with st.spinner("Generating triage brief…"):
        result = rag.generate(
            claim_data=claim_data,
            reason_codes=[c["label"] for c in codes],
            risk_score=risk_score,
            triage_bucket=bucket,
        )
    method_label = "🤖 GPT-4o-mini" if result["method"] == "llm" else "📋 Template"
    st.caption(f"Generated via: {method_label}")
    st.markdown(
        f'<div class="brief-box">{result["brief"]}</div>',
        unsafe_allow_html=True,
    )




# ══════════════════════════════════════════════════════════════
# PAGE 5 — FAIRNESS ANALYSIS
# ══════════════════════════════════════════════════════════════

def page_fairness(df_raw, test_prob, y_test):
    st.title("⚖️ Fairness Analysis")
    st.caption("Checks whether the model treats different demographic groups equitably.")

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        Insurance companies are legally required not to discriminate against customers based on protected attributes like age, sex, or marital status.
        This page checks whether the fraud model flags certain groups at a disproportionately high rate.

        - **Flag Rate** — the percentage of claims in that group that were sent to SIU or Manual Review.
        - **Fraud Rate** — the actual percentage of claims in that group that were genuinely fraudulent.
        - **False Positive Rate** — the percentage of legitimate claims in that group that were incorrectly flagged.
        - **Precision** — of the flagged claims in that group, how many were actually fraud.
        - **Disparate Impact Ratio** — the *least*-flagged group's flag rate divided by this group's flag rate. The least-flagged group therefore scores 1.00, and a group falls below **0.80 (the 80% rule)** when it is flagged at more than 1.25× the least-flagged group's rate — a potential legal concern under US insurance regulation. Because being flagged means being investigated, the rule surfaces the *most*-flagged group here.
        - Groups with fewer than 30 claims are excluded — small samples produce unreliable ratios.
        - 🟢 Green = no concern. 🔴 Red = potential bias worth investigating.
        """)


    from fairness_analysis import compute_fairness_report, DISPARATE_IMPACT_THRESHOLD

    MIN_GROUP_SIZE = 30   # groups smaller than this are excluded from DI calculation

    with st.spinner("Computing fairness metrics…"):
        raw_test = df_raw.iloc[y_test.index].reset_index(drop=True)
        reports, analysis = compute_fairness_report(raw_test, test_prob, np.array(y_test))

    # ── Overall flag rates ────────────────────────────────────
    st.markdown("### What is Disparate Impact?")
    st.info(
        f"The **80% rule**: each group is scored as "
        f"(least-flagged group's flag rate) ÷ (this group's flag rate). A score below "
        f"{DISPARATE_IMPACT_THRESHOLD:.2f} means the group is flagged at more than "
        f"{1/DISPARATE_IMPACT_THRESHOLD:.2f}× the least-flagged group's rate, which may "
        f"indicate bias. This is the standard used in US insurance regulation."
    )
    st.caption(
        f"⚠️ Groups with fewer than {MIN_GROUP_SIZE} claims are excluded from the "
        f"disparate impact calculation — small samples produce unreliable ratios."
    )

    for attr, df in reports.items():
        st.markdown(f"---")
        st.markdown(f"### {attr}")

        # Split into included and excluded groups. The "age not recorded" bucket
        # (rows with Age == 0, which is not a real age) is excluded alongside
        # small groups — it is a data-quality artifact, not a demographic.
        from fairness_analysis import AGE_UNKNOWN_LABEL
        keep = (df["n"] >= MIN_GROUP_SIZE) & (df["group"] != AGE_UNKNOWN_LABEL)
        included = df[keep].copy()
        excluded = df[~keep].copy()

        if excluded.shape[0] > 0:
            excl_names = ", ".join(excluded["group"].tolist())
            st.caption(
                f"Excluded from disparate impact (n < {MIN_GROUP_SIZE}, or age not "
                f"recorded): **{excl_names}**"
            )

        # Recompute disparate impact on the included groups, using the
        # least-flagged group WITH A NON-ZERO flag rate as the reference. A group
        # that is never flagged (65+, n=48) cannot be a ratio denominator — using
        # it drove every other group to 0.000 and flagged them all as "Concern".
        if not included.empty and included["flag_rate"].max() > 0:
            nonzero  = included.loc[included["flag_rate"] > 0, "flag_rate"]
            min_flag = nonzero.min() if len(nonzero) else 0.0
            included["disparate_impact"] = included["flag_rate"].apply(
                lambda x: round(min_flag / x, 4) if (x > 0 and min_flag > 0) else None
            )
            included["di_flag"] = included["disparate_impact"].apply(
                lambda x: "n/a (never flagged)" if pd.isna(x)
                else ("⚠️ Concern" if x < DISPARATE_IMPACT_THRESHOLD else "✅ OK")
            )
            never_flagged = included.loc[included["flag_rate"] <= 0, "group"].tolist()
            if never_flagged:
                st.caption(
                    f"⚪ Never flagged by the model, so no ratio is defined: "
                    f"**{', '.join(never_flagged)}** — worth noting in its own right."
                )

        # Color-code the DI flag column
        def highlight_di(row):
            if "Concern" in str(row.get("di_flag", "")):
                return ["background-color: #fef2f2"] * len(row)
            return ["background-color: #f0fdf4"] * len(row)

        display_cols = ["group", "n", "fraud_rate", "flag_rate",
                        "false_pos_rate", "precision",
                        "disparate_impact", "di_flag"]
        display_cols = [c for c in display_cols if c in included.columns]

        if not included.empty:
            st.dataframe(
                included[display_cols].style
                    .apply(highlight_di, axis=1)
                    .format({
                        "fraud_rate":       "{:.3f}",
                        "flag_rate":        "{:.3f}",
                        "false_pos_rate":   "{:.3f}",
                        "precision":        "{:.3f}",
                        "disparate_impact": "{:.3f}",
                    }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No groups meet the minimum sample size threshold.")

        # Bar chart: flag rate per group (included only)
        if not included.empty:
            col_l, col_r = st.columns(2)
            with col_l:
                # The colour scale is pinned to an absolute 0 -> 30% range.
                # Auto-scaling stretched the palette across whatever spread the
                # data happened to have, so Married (0.20095) rendered red and
                # Single (0.2008) green — a 0.0002 difference shown as the full
                # green-to-red span.
                fig = px.bar(
                    included, x="group", y="flag_rate",
                    color="flag_rate",
                    color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
                    range_color=[0.0, 0.30],
                    title=f"Flag Rate by {attr}",
                    labels={"flag_rate": "Flag Rate", "group": attr},
                )
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                                  showlegend=False,
                                  plot_bgcolor="#ffffff",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                if "disparate_impact" in included.columns:
                    colors = ["#e74c3c" if di < DISPARATE_IMPACT_THRESHOLD else "#27ae60"
                              for di in included["disparate_impact"]]
                    fig2 = go.Figure(go.Bar(
                        x=included["group"], y=included["disparate_impact"],
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in included["disparate_impact"]],
                        textposition="outside",
                    ))
                    fig2.add_hline(y=DISPARATE_IMPACT_THRESHOLD, line_dash="dash",
                                   line_color="black",
                                   annotation_text="80% rule threshold")
                    fig2.update_layout(
                        title=f"Disparate Impact Ratio — {attr}",
                        yaxis=dict(range=[0, 1.3]),
                        height=300, margin=dict(l=10, r=10, t=40, b=10),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    # ── Summary ───────────────────────────────────────────────
    st.markdown("---")
    any_concern = any(
        "Concern" in str(row.get("di_flag", ""))
        for df in reports.values()
        for _, row in df[df["n"] >= MIN_GROUP_SIZE].iterrows()
        if "di_flag" in df.columns
    )
    if any_concern:
        st.warning(
            "⚠️ Some groups fall below the 80% rule — they are flagged for "
            "investigation at more than 1.25× the rate of the least-flagged group."
        )
        st.markdown(
            """
**What this does and does not mean.**

A low ratio means a group carries a disproportionate share of the investigation
burden. It is not automatically evidence of bias — it may track a real
difference in fraud rates. On this test split, males commit fraud at **2.36×**
the female rate, while the model flags them at **1.74×** the female rate, so the
model's disparity is *smaller* than the disparity in the labels it learned from.

**The thing to check here is not proxies — it is direct use.** `Sex`,
`MaritalStatus` and `Age` are not inferred through correlated features; they are
one-hot encoded straight into the feature matrix (`Sex_Female` ranks 25th of 90
by mean |SHAP|, and age-derived features rank 13th, 14th and 17th). Searching
for indirect proxies is the wrong first move when the protected attribute itself
is an input.

**What that implies.** Sex-based differentiation in insurance is restricted in
several US states and prohibited across the EU, though rules for fraud
*investigation* triage differ from rules for pricing — a compliance question,
not a modelling one. The measurable next step is a leave-one-out test: retrain
without `Sex` and compare PR-AUC against the change in disparate impact. Given
its modest SHAP contribution, the accuracy cost is likely to be small.
"""
        )
    else:
        st.success("✅ No disparate impact concerns detected across all demographic groups.")


# ══════════════════════════════════════════════════════════════
# PAGE 6 — MONITORING
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Running monitoring pipeline…")
def _run_monitoring_cached(raw_test, test_prob, y_arr, raw_train):
    """Cached so that re-rendering the page does not recompute 6 batches of KS
    tests, and with write_outputs=False so viewing a page writes no CSVs."""
    from monitoring import run_monitoring
    return run_monitoring(raw_test, test_prob, y_arr,
                          reference_df=raw_train, write_outputs=False)


def page_monitoring(df_raw, test_prob, y_test, raw_train):
    st.title("📡 Model Monitoring")
    st.caption(
        "Simulated production monitoring — the test set is split into "
        "chronological batches and each is compared against the training data."
    )

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        In production, claims arrive continuously and fraud patterns shift, so a model can quietly lose accuracy. This page demonstrates the machinery you would use to catch that.

        **What this is and is not.** All 15,420 claims come from a single static 1994–96 collection that was split randomly, so there is no real time axis along which this model could decay. The batches below are a *simulation* of claims arriving over time. The expected — and correct — result here is **stable**; the value is in showing the detection mechanics work, not in finding drift that cannot exist in a random split.

        - **Reference** — each batch is compared against the **training split**, which is what "has the incoming data drifted from what the model was trained on?" actually means.
        - **Batches** — the test set ordered by (Year, Month), then cut into 6 equal groups. Ordering by month alone would interleave 1994/1995/1996 within each batch.
        - **Drift Detection** — a two-sample KS test per feature, with **Benjamini-Hochberg correction**. Without it, 5 batches × 7 features = 35 tests at p<0.05 produce ~1.75 false alarms by chance, which is why an uncorrected version flags drift almost every run.
        - **Excluded features** — `Year` (the batching key, so testing it is circular) and `RepNumber` (a staff identifier, not a distribution).
        - **Fraud Rate / Avg Risk Score per Batch** — sustained movement in either is the signal that would justify retraining.
        - ✅ Stable = no feature drifted after correction. ⚠️ Drift = at least one survived it.
        """)


    from monitoring import get_drift_summary, N_BATCHES

    raw_test = df_raw.iloc[y_test.index].reset_index(drop=True)
    results  = _run_monitoring_cached(raw_test, test_prob, np.array(y_test), raw_train)

    stats_df     = results["batch_stats"]
    drift_results = results["drift_results"]
    drift_summary = get_drift_summary(drift_results)

    # ── KPI row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches Analysed", len(stats_df))
    c2.metric("Avg Fraud Rate",   f"{stats_df['fraud_rate'].mean():.3f}")
    c3.metric("Avg Risk Score",   f"{stats_df['avg_risk_score'].mean():.4f}")
    drift_status = "⚠️ Drift Detected" if drift_summary["any_drift"] else "✅ Stable"
    c4.metric("Drift Status", drift_status)

    st.markdown("---")

    # ── Trend charts ──────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Fraud Rate per Batch**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats_df["label"], y=stats_df["fraud_rate"],
            mode="lines+markers", line=dict(color="#e74c3c", width=2),
            marker=dict(size=8),
        ))
        fig.add_hline(y=stats_df["fraud_rate"].mean(), line_dash="dot",
                      line_color="#9ca3af",
                      annotation_text=f"Mean {stats_df['fraud_rate'].mean():.3f}")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=40),
                          plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(gridcolor="#f3f4f6"),
                          yaxis=dict(gridcolor="#f3f4f6"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Average Risk Score per Batch**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=stats_df["label"], y=stats_df["avg_risk_score"],
            mode="lines+markers", line=dict(color="#2563eb", width=2),
            marker=dict(size=8), name="Avg score",
        ))
        fig2.add_trace(go.Scatter(
            x=pd.concat([stats_df["label"], stats_df["label"][::-1]]),
            y=pd.concat([
                stats_df["avg_risk_score"] + stats_df["std_risk_score"],
                (stats_df["avg_risk_score"] - stats_df["std_risk_score"])[::-1],
            ]),
            fill="toself", fillcolor="rgba(37,99,235,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="± std",
        ))
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=40),
                           plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(gridcolor="#f3f4f6"),
                           yaxis=dict(gridcolor="#f3f4f6"),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Drift table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Drift Detection Results**")
    st.caption(
        f"Reference: **{results['reference_kind']}** · "
        f"Features tested: {', '.join(results['drift_features'])} · "
        f"Excluded: {', '.join(results['excluded'])} · "
        f"KS test with Benjamini-Hochberg correction at α=0.05"
    )
    if drift_results:
        drift_df = pd.DataFrame([
            {
                "Batch":            d["batch_label"],
                "Dataset Drift":    "⚠️ Yes" if d["dataset_drift"] else "✅ No",
                "Drifted (BH-adj)": d["n_drifted_features"],
                "Drifted (raw p)":  d.get("n_drifted_uncorrected", "—"),
                "Total Features":   d["n_features"],
                "Drift Share":      f"{d['drift_share']:.1%}",
            }
            for d in drift_results
        ])
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.info("No drift results available.")

    # ── Batch stats table ─────────────────────────────────────
    st.markdown("---")
    st.markdown("**Batch Statistics**")
    display_stats = stats_df[[
        "label", "n", "fraud_rate", "avg_risk_score",
        "siu_count", "manual_count", "siu_fraud_rate"
    ]].copy()
    display_stats.columns = [
        "Batch", "Claims", "Fraud Rate", "Avg Risk Score",
        "SIU Count", "Manual Count", "SIU Fraud Rate"
    ]
    st.dataframe(
        display_stats.style.format({
            "Fraud Rate":     "{:.3f}",
            "Avg Risk Score": "{:.4f}",
            "SIU Fraud Rate": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════
# PAGE 7 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Running seasonality analysis…")
def _run_temporal_cached(raw_test, test_prob, y_arr):
    """Cached, and write_outputs=False so viewing the page writes no CSVs."""
    from temporal_analysis import run_temporal_analysis
    return run_temporal_analysis(raw_test, test_prob, y_arr, write_outputs=False)


def page_temporal(df_raw, test_prob, y_test):
    st.title("📅 Seasonality Analysis")
    st.caption(
        "Model performance by calendar month, pooled across 1994–1996 — "
        "which months the model handles well, not change over time."
    )

    with st.expander("ℹ️", expanded=False):
        st.markdown(f"""
        Fraud may be seasonal: certain months see more staged accidents or more opportunistic claims. This page evaluates the model **separately on each calendar month's claims**.

        **Read this as seasonality, not decay.** Months are pooled across all three years in the dataset — "December" means every December from 1994 to 1996, not a point on a timeline. The dataset is one static historical collection split randomly, so it contains no chronological axis along which the model could degrade. Drift is covered on the Monitoring page.

        **Sample sizes are small.** Each month holds roughly 130 test claims and between 3 and 13 fraud cases, and Precision@5% is computed over just 5–7 claims. Months with fewer than {5} fraud cases have their PR-AUC and Precision@5% **suppressed rather than plotted**, because a ranking metric built on 3 positives is noise. The `Fraud` column in the table shows what each number rests on.

        - **PR-AUC by Month** — how well the model ranks fraud above non-fraud within that month.
        - **Fraud Rate by Month** — the actual share of claims that were fraudulent.
        - **Precision@5% by Month** — of that month's top 5% highest-risk claims, how many were fraud.
        - **Avg Risk Score by Month** — the model's average output level.
        - **Best/Worst Month** — computed only over months with a sufficient sample.
        - 🔵 Blue = best scored month. 🔴 Red = worst scored month.
        """)


    from temporal_analysis import get_temporal_summary

    raw_test   = df_raw.iloc[y_test.index].reset_index(drop=True)
    df_metrics = _run_temporal_cached(raw_test, test_prob, np.array(y_test))

    if df_metrics.empty:
        st.warning("Not enough data per month for temporal analysis.")
        return

    summary = get_temporal_summary(df_metrics)

    # ── KPI row ───────────────────────────────────────────────
    n_scored = summary.get("n_months_scored", 0)
    c1, c2, c3, c4 = st.columns(4)
    # delta_color="off" on both: these are captions, not improvements. Streamlit
    # renders a delta as a green up-arrow by default, which made a sample-size
    # caveat and a standard deviation both read as good news.
    c1.metric("Months Analysed",  summary["n_months"],
              delta=f"{n_scored} scored · {summary['n_months'] - n_scored} below "
                    f"5 fraud cases",
              delta_color="off")
    if n_scored:
        c2.metric("Mean PR-AUC",  f"{summary['mean_pr_auc']:.4f}",
                  delta=f"± {summary['std_pr_auc']:.4f} across months",
                  delta_color="off")
        c3.metric("Best Month",   f"{summary['best_month']} ({summary['best_pr_auc']:.4f})")
        c4.metric("Worst Month",  f"{summary['worst_month']} ({summary['worst_pr_auc']:.4f})")
    else:
        c2.metric("Mean PR-AUC", "—")
        c3.metric("Best Month",  "—")
        c4.metric("Worst Month", "—")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**PR-AUC by Month**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_metrics["month"], y=df_metrics["pr_auc"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=9),
            name="PR-AUC",
        ))
        fig.add_hline(y=summary["mean_pr_auc"], line_dash="dot",
                      line_color="#9ca3af",
                      annotation_text=f"Mean {summary['mean_pr_auc']:.4f}")
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6", title="PR-AUC"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Fraud Rate by Month**")
        fig2 = go.Figure(go.Bar(
            x=df_metrics["month"],
            y=df_metrics["fraud_rate"],
            marker_color="#e74c3c",
            opacity=0.8,
        ))
        fig2.add_hline(y=summary["mean_fraud_rate"], line_dash="dot",
                       line_color="#9ca3af",
                       annotation_text=f"Mean {summary['mean_fraud_rate']:.3f}")
        fig2.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6", title="Fraud Rate"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("**Precision@5% by Month**")
        fig3 = go.Figure(go.Scatter(
            x=df_metrics["month"], y=df_metrics["precision_5pct"],
            mode="lines+markers",
            line=dict(color="#27ae60", width=2.5),
            marker=dict(size=9, symbol="square"),
        ))
        fig3.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6", title="Precision@5%"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("**Average Risk Score by Month**")
        fig4 = go.Figure(go.Scatter(
            x=df_metrics["month"], y=df_metrics["avg_risk_score"],
            mode="lines+markers",
            line=dict(color="#f39c12", width=2.5),
            marker=dict(size=9, symbol="diamond"),
        ))
        fig4.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6", title="Avg Risk Score"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Monthly metrics table ─────────────────────────────────
    st.markdown("---")
    st.markdown("**Monthly Metrics Table**")
    st.caption(
        "Blank PR-AUC / Precision@5% means that month had fewer than 5 fraud "
        "cases — too few for a ranking metric to be meaningful."
    )
    display_metrics = df_metrics.drop(columns=["sufficient"], errors="ignore").rename(columns={
        "month":          "Month",
        "month_num":      "Month #",
        "n":              "Claims",
        "n_fraud":        "Fraud",
        "fraud_rate":     "Fraud Rate",
        "pr_auc":         "PR-AUC",
        "roc_auc":        "ROC-AUC",
        "precision_5pct": "Precision@5%",
        "avg_risk_score": "Avg Risk Score",
        "siu_fraud_rate": "SIU Fraud Rate",
    })
    st.dataframe(
        display_metrics.style.format({
            "Fraud Rate":     "{:.4f}",
            "PR-AUC":         "{:.4f}",
            "ROC-AUC":        "{:.4f}",
            "Precision@5%":   "{:.4f}",
            "Avg Risk Score": "{:.4f}",
            "SIU Fraud Rate": "{:.4f}",
        # na_rep so suppressed months render as an em dash rather than the
        # literal string "None", which contradicted the caption above.
        }, na_rep="—").highlight_max(
            subset=["PR-AUC", "Precision@5%"],
            color="#dbeafe",
        ).highlight_min(
            subset=["PR-AUC", "Precision@5%"],
            color="#fee2e2",
        ),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def page_dataset_comparison():
    st.title("🗂️ Dataset Comparison")
    st.caption("Comparative analysis of two heterogeneous fraud datasets used in this project.")

    with st.expander("ℹ️", expanded=False):
        st.markdown("""
        This page compares the two datasets used in this project:

        - **Automotive Insurance Claims** (`fraud_oracle.csv`) — the primary dataset used for model training and evaluation.
        - **Medicare Healthcare Provider Fraud** — a secondary dataset used for cross-domain comparative analysis.

        Both datasets share the core challenge of fraud detection under severe class imbalance,
        validating that the PR-AUC metric and triage methodology are domain-agnostic.
        """)

    import glob

    medicare_dir = os.path.join("outputs", "medicare")

    # ── Dataset Profile Table ─────────────────────────────────
    st.markdown('<p class="section-label">Dataset Profiles</p>', unsafe_allow_html=True)

    profile_data = {
        "Attribute": [
            "Domain", "Source", "Unit of Analysis", "Total Records",
            "Total Raw Claim Records", "Features (raw)", "Target Variable",
            "Fraud Rate", "Fraud Cases", "Imbalance Ratio (legit:fraud)",
            "Recommended Metric", "Time Period", "Geography",
        ],
        "Automotive Insurance (fraud_oracle)": [
            "Automotive Insurance", "Kaggle — Shivam Bansal, 2021",
            "Individual claim", "15,420",
            "15,420", "33", "FraudFound_P (0/1)",
            "5.99%", "923", "15.7:1",
            "PR-AUC", "1994–1996", "USA",
        ],
        "Medicare Provider Fraud": [
            "Healthcare / Medicare", "Kaggle — RohitRox, 2019",
            "Healthcare provider", "5,410 providers",
            "558,211", "25–30 per file", "PotentialFraud (Yes/No)",
            "9.35%", "506", "9.7:1",
            "PR-AUC", "2009–2010", "USA (state + county)",
        ],
    }

    df_profile = pd.DataFrame(profile_data).set_index("Attribute")
    st.dataframe(df_profile, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Insight Cards ─────────────────────────────────────
    st.markdown('<p class="section-label">Key Comparative Insights</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combined Raw Records", "573,631",
              delta="15,420 auto + 558,211 Medicare", delta_color="off")
    c2.metric("Auto Fraud Rate", "5.99%",
              delta="15.7:1 imbalance ratio", delta_color="off")
    c3.metric("Medicare Fraud Rate", "9.35%",
              delta="9.7:1 imbalance ratio", delta_color="off")
    c4.metric("Shared Best Metric", "PR-AUC",
              delta="Both datasets imbalanced", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plots ─────────────────────────────────────────────────
    plot_files = {
        "class_imbalance_comparison.png":      "Class Imbalance — Both Datasets",
        "fraud_rate_comparison.png":           "Fraud Rate Comparison",
        "feature_type_comparison.png":         "Feature Categories (Heterogeneity)",
        "claim_amount_distributions.png":      "Claim Amount Distributions",
        "medicare_fraud_by_claim_volume.png":  "Medicare: Fraud Rate by Claim Volume",
        "methodology_comparison.png":          "Methodology Applicability",
    }

    missing = [f for f in plot_files if not os.path.exists(os.path.join(medicare_dir, f))]

    if missing:
        st.warning(
            "Comparison plots not found. Run `python medicare_comparison.py` to generate them.",
            icon="⚠️",
        )
    else:
        # Row 1
        st.markdown('<p class="section-label">Visualizations</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Class Imbalance — Both Datasets**")
            st.image(os.path.join(medicare_dir, "class_imbalance_comparison.png"),
                     use_container_width=True)
        with col2:
            st.markdown("**Fraud Rate Comparison**")
            st.image(os.path.join(medicare_dir, "fraud_rate_comparison.png"),
                     use_container_width=True)

        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Feature Categories (Heterogeneity)**")
            st.image(os.path.join(medicare_dir, "feature_type_comparison.png"),
                     use_container_width=True)
        with col4:
            st.markdown("**Claim Amount Distributions**")
            st.image(os.path.join(medicare_dir, "claim_amount_distributions.png"),
                     use_container_width=True)

        # Row 3
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Medicare: Fraud Rate by Provider Claim Volume**")
            st.image(os.path.join(medicare_dir, "medicare_fraud_by_claim_volume.png"),
                     use_container_width=True)
        with col6:
            st.markdown("**Methodology Applicability Across Both Domains**")
            st.image(os.path.join(medicare_dir, "methodology_comparison.png"),
                     use_container_width=True)

    st.divider()

    # ── Data Processing Pipeline ──────────────────────────────
    st.markdown('<p class="section-label">Medicare Data Processing Pipeline</p>',
                unsafe_allow_html=True)
    st.markdown("The Medicare dataset consists of **4 separate CSV files** that were cleaned, "
                "aggregated, and integrated into a single provider-level feature matrix:")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Step 1 — File Integration (4 → 1)**")
        st.code("""# Join 4 files on Provider ID
labels   → 5,410 providers + fraud label
inpatient  → 40,474 claims  (aggregated by Provider)
outpatient → 517,737 claims (aggregated by Provider)
beneficiary→ 138,556 patients (aggregated via claims)

Result: 1 unified provider feature matrix""", language="text")

        st.markdown("**Step 2 — Claim-level → Provider-level Aggregation**")
        st.code("""inp.groupby("Provider").agg(
    InpatientClaims         = ("ClaimID",               "count"),
    AvgInpatientReimbursed  = ("InscClaimAmtReimbursed", "mean"),
    TotalInpatientAmt       = ("InscClaimAmtReimbursed", "sum"),
    AvgDeductible           = ("DeductibleAmtPaid",      "mean"),
    UniquePatients_IP       = ("BeneID",                 "nunique"),
    UniquePhysicians_IP     = ("AttendingPhysician",     "nunique"),
)""", language="python")

    with col_b:
        st.markdown("**Step 3 — Missing Value Handling**")
        st.code("""# Providers with no inpatient/outpatient claims
# receive 0 for all aggregated features
df = df.fillna(0)

# Missing value rate before fill:
# InpatientClaims:  0.0%  (all providers have claims)
# OutpatientClaims: 8.2%  (some providers inpatient-only)
# BeneficiaryData:  12.4% (not all patients have records)""",
                language="python")

        st.markdown("**Step 4 — Label Encoding**")
        st.code("""# Convert string label to binary
labels["Fraud"] = (
    labels["PotentialFraud"] == "Yes"
).astype(int)
# "Yes" → 1 (fraud)
# "No"  → 0 (legitimate)""", language="python")

        st.markdown("**Final Feature Matrix**")
        st.markdown("""
| Property | Value |
|---|---|
| Rows (providers) | 5,410 |
| Columns (features) | 22 |
| Fraud cases | 506 (9.35%) |
| Missing values | 0 (after fillna) |
| Source files joined | 4 |
| Raw claims processed | 558,211 |
""")

    st.divider()

    # ── Model Results ─────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">Model Results — XGBoost (Optuna)</p>',
                unsafe_allow_html=True)

    results_path = os.path.join("outputs", "medicare", "medicare_model_results.json")
    medicare_dir = os.path.join("outputs", "medicare")

    if not os.path.exists(results_path):
        st.warning("Model results not found. Run `python medicare_modeling.py` first.", icon="⚠️")
    else:
        import json
        with open(results_path) as f:
            med_results = json.load(f)
        m = med_results["metrics"]

        # ── KPI comparison ────────────────────────────────────
        # Auto-side figures are read from the training artifacts rather than
        # hardcoded. The enrichment comparison in particular used to pit the
        # auto VALIDATION number (5.4x) against Medicare's TEST number (8.9x);
        # both sides are now test-set.
        with open(os.path.join(IMPROVEMENT, "model_metadata.json")) as f:
            auto_meta = json.load(f)
        auto_val_pr  = auto_meta["val_pr_auc"]
        auto_test_pr = auto_meta["test_pr_auc"]
        auto_enrich  = auto_meta.get("siu_enrichment_test")
        if auto_enrich is None:   # metadata predates triage_summary.json
            with open(os.path.join(IMPROVEMENT, "triage_summary.json")) as f:
                auto_enrich = json.load(f)["test"]["buckets"]["SIU"]["enrichment"]

        st.markdown("**Performance vs. Automotive Insurance Model**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Auto Val PR-AUC",      f"{auto_val_pr:.4f}")
        c2.metric("Medicare Val PR-AUC",  f"{m['val_pr_auc']:.4f}",
                  delta=f"{m['val_pr_auc']-auto_val_pr:+.4f} vs auto")
        c3.metric("Auto Test PR-AUC",     f"{auto_test_pr:.4f}")
        c4.metric("Medicare Test PR-AUC", f"{m['test_pr_auc']:.4f}",
                  delta=f"{m['test_pr_auc']-auto_test_pr:+.4f} vs auto")
        c5.metric("Auto SIU Enrichment (test)",  f"{auto_enrich:.1f}×")
        c6.metric("Medicare SIU Enrichment (test)", f"{m['siu_enrichment']:.1f}×",
                  delta=f"{m['siu_enrichment']-auto_enrich:+.1f}× vs auto")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Triage table ──────────────────────────────────────
        col_t, col_p = st.columns(2)
        with col_t:
            st.markdown("**Triage Bucket Performance (Medicare Test Set)**")
            triage_df = pd.DataFrame(med_results["triage"])
            st.dataframe(triage_df, use_container_width=True, hide_index=True)
            st.caption(f"5-fold CV PR-AUC: {m['cv_mean']:.4f} ± {m['cv_std']:.4f}")

        with col_p:
            st.markdown("**PR Curve — Medicare Model**")
            pr_path = os.path.join(medicare_dir, "medicare_pr_curve.png")
            if os.path.exists(pr_path):
                st.image(pr_path, width=620)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP comparison ───────────────────────────────────
        st.markdown("**Key Finding: Domain-Specific Fraud Signals (SHAP)**")
        shap_comp_path = os.path.join(medicare_dir, "shap_comparison.png")
        if os.path.exists(shap_comp_path):
            st.image(shap_comp_path, width=950)
        st.caption(
            "Auto insurance fraud is driven by **claim characteristics** (fault, police reports, vehicle type). "
            "Medicare fraud is driven by **billing volume** (total reimbursed, inpatient claim count). "
            "Completely different top features confirm that domain-specific feature engineering is essential — "
            "a generic model trained on one domain would not transfer to the other."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model performance comparison chart ────────────────
        st.markdown("**PR-AUC Comparison: Auto vs. Medicare**")
        perf_path = os.path.join(medicare_dir, "model_performance_comparison.png")
        if os.path.exists(perf_path):
            st.image(perf_path, width=700)

    st.divider()

    # ── Why Two Datasets ──────────────────────────────────────
    st.markdown('<p class="section-label">Why Two Datasets?</p>', unsafe_allow_html=True)
    st.markdown("""
    Using two heterogeneous datasets serves three purposes:

    1. **Validates methodology generalizability** — PR-AUC, gradient boosting, SHAP explainability,
       and triage bucket routing all apply to both domains, confirming these are domain-agnostic fraud detection techniques.

    2. **Demonstrates heterogeneity** — Automotive insurance fraud is driven by claim characteristics
       (fault, police reports, vehicle type), while Medicare fraud is driven by billing patterns
       (claim volume, reimbursement amounts, diagnosis code diversity). Different top features per domain
       confirm that domain-specific feature engineering is essential.

    3. **Contextualizes class imbalance** — Both datasets are severely imbalanced (6% and 9.4% fraud rates),
       reinforcing why accuracy is a misleading metric and PR-AUC is the correct evaluation criterion
       for fraud detection regardless of domain.
    """)


# ══════════════════════════════════════════════════════════════
def main():
    # Load everything
    metadata   = load_metadata()
    data       = load_all_data()
    model      = load_model()
    rag        = load_rag()
    model_comp = load_model_comparison()

    val_prob, test_prob = score_claims(model, data["X_val_t"], data["X_test_t"])
    # Use test set for dashboard — completely unseen data, most honest representation
    display_prob = test_prob
    display_y    = data["y_test"]
    explainer, shap_values = load_shap_values(model, data["X_test_t"])
    # SHAP global importance computed on test set for full consistency
    shap_imp = load_shap_importance(model, data["X_test_t"], data["feat_names"])

    # Sidebar + navigation
    page = render_sidebar(metadata)

    # Allow queue page to redirect to detail
    if "page_override" in st.session_state:
        page = st.session_state.pop("page_override")

    if page == "📊 Summary Dashboard":
        page_summary(metadata, display_prob, display_y, shap_imp, model_comp)

    elif page == "📋 Review Queue":
        page_queue(data["df_raw"], display_prob, display_y)

    elif page == "🔎 Claim Detail":
        page_detail(
            data["df_raw"], display_prob, display_y,
            shap_values, explainer,
            data["feat_names"], data["cat_cols"], rag,
            data["X_test_t"],
        )

    elif page == "⚡ Live Scoring":
        page_live(
            model, data["preprocessor"],
            data["cat_cols"], data["num_cols"],
            data["feat_names"], rag, display_prob,
        )

    elif page == "⚖️ Fairness Analysis":
        page_fairness(data["df_raw"], display_prob, display_y)

    elif page == "📡 Monitoring":
        # Drift is measured against the TRAINING split — that is what "has the
        # incoming data drifted from what the model was trained on?" means.
        from data_pipeline import raw_rows_for
        page_monitoring(data["df_raw"], display_prob, display_y,
                        raw_rows_for(data, "train"))

    elif page == "📅 Seasonality Analysis":
        page_temporal(data["df_raw"], display_prob, display_y)

    elif page == "🗂️ Dataset Comparison":
        page_dataset_comparison()


if __name__ == "__main__":
    main()
