"""
Support Integrity Auditor (SIA) — Streamlit Web App
=====================================================
Interactive dashboard for CRM priority mismatch detection.

Run: streamlit run app.py

Pages:
  🏠 Dashboard  — Overview metrics, mismatch distributions, heatmaps
  🔍 Single Ticket — Manual form input with evidence dossier
  📦 Batch Predict — CSV upload with downloadable results
  📊 Methodology — Pipeline architecture, ablation, metrics
"""

import json, os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from train_pipeline import (
    generate_pseudo_labels,
    build_features,
    generate_dossier,
    PRIORITY_MAP, SEVERITY_NAMES,
    URGENCY_KEYWORDS, ESCALATION_PHRASES,
    CAT_SEVERITY_MAP, CHANNEL_WEIGHT_MAP,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background: #1e2130; border-radius: 10px; padding: 16px;
        border-left: 4px solid; margin-bottom: 8px;
    }
    .crisis  { border-color: #ef4444; }
    .alarm   { border-color: #f59e0b; }
    .pass    { border-color: #10b981; }
    .badge-crisis { background:#ef4444; color:#fff; padding:2px 10px;
                    border-radius:12px; font-size:13px; font-weight:700; }
    .badge-alarm  { background:#f59e0b; color:#000; padding:2px 10px;
                    border-radius:12px; font-size:13px; font-weight:700; }
    .badge-consistent { background:#10b981; color:#fff; padding:2px 10px;
                    border-radius:12px; font-size:13px; font-weight:700; }
    div[data-testid="stExpander"] { border: 1px solid #2d3348; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_dir: str = "model"):
    """Load trained XGBoost model and preprocessors."""
    try:
        clf      = joblib.load(f"{model_dir}/clf_model.pkl")
        scaler   = joblib.load(f"{model_dir}/scaler.pkl")
        encoders = joblib.load(f"{model_dir}/encoders.pkl")
        return clf, scaler, encoders
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_labeled_data(output_dir: str = "outputs"):
    """Load pseudo-labeled dataset if available."""
    path = f"{output_dir}/labeled_tickets.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics(output_dir: str = "outputs"):
    """Load metrics JSON."""
    paths = [f"{output_dir}/metrics.json", "model/metrics.json"]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


@st.cache_data
def load_dossiers(output_dir: str = "outputs"):
    """Load sample dossiers."""
    path = f"{output_dir}/dossiers.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def score_single_ticket(row_dict: dict, clf, scaler, encoders):
    """Run full pipeline on a single ticket dict."""
    df_single = pd.DataFrame([row_dict])
    df_labeled, sig_agree = generate_pseudo_labels(df_single)
    X, _ = build_features(df_labeled, encoders=encoders, fit=False)
    X_s = scaler.transform(X)
    prob = float(clf.predict_proba(X_s)[0, 1])
    pred = int(clf.predict(X_s)[0])
    row  = df_labeled.iloc[0]
    dossier = generate_dossier(row, prob)
    return pred, prob, row, dossier


def score_dataframe(df: pd.DataFrame, clf, scaler, encoders):
    """Run full pipeline on a DataFrame."""
    df_labeled, sig_agree = generate_pseudo_labels(df)
    X, _ = build_features(df_labeled, encoders=encoders, fit=False)
    X_s = scaler.transform(X)
    probs = clf.predict_proba(X_s)[:, 1]
    preds = clf.predict(X_s)
    df_labeled["pred_mismatch"]   = preds
    df_labeled["pred_confidence"] = probs
    df_labeled["pred_label"]      = [
        "Mismatch" if p else "Consistent" for p in preds
    ]
    return df_labeled, sig_agree


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def priority_badge(label: str) -> str:
    colors = {"Low": "#22c55e", "Medium": "#f59e0b",
              "High": "#f97316", "Critical": "#ef4444", "Unknown": "#6b7280"}
    c = colors.get(label, "#6b7280")
    return f'<span style="background:{c};color:white;padding:2px 12px;border-radius:12px;font-weight:600;font-size:0.9em">{label}</span>'


def mismatch_badge(mt: str) -> str:
    if mt == "Hidden Crisis":
        return '<span class="badge-crisis">🚨 Hidden Crisis</span>'
    elif mt == "False Alarm":
        return '<span class="badge-alarm">⚠️ False Alarm</span>'
    return '<span class="badge-consistent">✅ Consistent</span>'


def render_dossier(d: dict):
    """Render a single evidence dossier as styled HTML + expanders."""
    mt      = d.get("mismatch_type", "")
    is_crisis = mt == "Hidden Crisis"
    border  = "#ef4444" if is_crisis else "#f59e0b"
    bg      = "#fef2f2" if is_crisis else "#fffbeb"

    st.markdown(f"""
    <div style="border:2px solid {border};border-radius:12px;padding:20px;background:{bg};margin-bottom:16px">
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
        <span style="font-weight:700;font-size:1.1em">🎫 {d['ticket_id']}</span>
        {mismatch_badge(mt)}
        <span style="color:#6b7280;font-size:0.9em">Δ {d['severity_delta']} levels</span>
        <span style="color:#6b7280;font-size:0.9em">confidence {d['confidence']}</span>
      </div>
      <div style="display:flex;gap:24px;margin-bottom:12px;align-items:center;">
        <div><span style="color:#6b7280;font-size:0.8em">ASSIGNED</span><br>{priority_badge(d['assigned_priority'])}</div>
        <div style="color:#9ca3af;padding-top:16px;font-size:1.2em">→</div>
        <div><span style="color:#6b7280;font-size:0.8em">INFERRED</span><br>{priority_badge(d['inferred_severity'])}</div>
      </div>
      <p style="font-style:italic;color:#374151;margin-bottom:12px">{d['constraint_analysis']}</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Feature Evidence (click to expand)"):
        for ev in d.get("feature_evidence", []):
            sig   = ev.get("signal", "").replace("_", " ").title()
            val   = ev.get("value", "")
            src   = ev.get("source_field", "")
            detail = ev.get("weight") or ev.get("interpretation", "")
            st.markdown(f"- **{sig}** ← `{src}`")
            st.markdown(f"  → Value: `{val}`")
            st.markdown(f"  → {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    clf, scaler, encoders = load_model()
    model_ready = clf is not None

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h1 style='text-align:center;font-size:48px;margin:0'>🛡️</h1>", unsafe_allow_html=True)
        st.title("🛡️ SIA")
        st.caption("Support Integrity Auditor")
        st.divider()

        page = st.radio(
            "Navigate",
            ["🏠 Dashboard", "🔍 Single Ticket", "📦 Batch Predict", "📊 Methodology"],
            label_visibility="collapsed"
        )
        st.divider()

        # Model status
        if model_ready:
            st.success("✅ Model loaded")
        else:
            st.error("❌ Model not found")
            st.info("Run: `python train_pipeline.py` first")

        # Metrics in sidebar
        metrics = load_metrics()
        if metrics:
            st.subheader("📈 Model Metrics")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            st.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
            col1, col2 = st.columns(2)
            col1.metric("Rec (Consistent)", f"{metrics.get('recall_consistent', 0):.4f}")
            col2.metric("Rec (Mismatch)", f"{metrics.get('recall_mismatch', 0):.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # 🏠 DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    if page == "🏠 Dashboard":
        st.title("🛡️ Support Integrity Auditor — Dashboard")
        st.caption("Priority mismatch intelligence across enterprise CRM tickets")

        df = load_labeled_data()
        if df is None:
            st.warning("No labeled data found. Run `python train_pipeline.py` first.")
            st.stop()

        # KPIs
        n_total    = len(df)
        n_mismatch = int(df["mismatch_label"].sum())
        n_crisis   = int(((df["inferred_severity"] > df["assigned_severity"]) & (df["mismatch_label"] == 1)).sum())
        n_false    = n_mismatch - n_crisis

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tickets", f"{n_total:,}")
        c2.metric("Flagged Mismatches", f"{n_mismatch:,}",
                  f"{n_mismatch/n_total*100:.1f}%")
        c3.metric("Hidden Crises 🚨", f"{n_crisis:,}")
        c4.metric("False Alarms ⚠️", f"{n_false:,}")

        st.divider()

        # Row 1: Mismatch type pie + Category bar chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Mismatch Type Distribution")
            mdf = df[df["mismatch_label"] == 1].copy()
            mdf["type"] = np.where(
                mdf["inferred_severity"] > mdf["assigned_severity"],
                "Hidden Crisis", "False Alarm"
            )
            fig = px.pie(
                mdf, names="type", color="type",
                color_discrete_map={"Hidden Crisis": "#ef4444", "False Alarm": "#f59e0b"},
                hole=0.45
            )
            fig.update_traces(textinfo="label+percent")
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Mismatch Rate by Issue Category")
            cat_agg = df.groupby("Issue_Category").agg(
                total=("mismatch_label", "count"),
                mismatches=("mismatch_label", "sum")
            ).reset_index()
            cat_agg["rate"] = cat_agg["mismatches"] / cat_agg["total"] * 100
            fig2 = px.bar(
                cat_agg.sort_values("rate", ascending=True),
                x="rate", y="Issue_Category", orientation="h",
                color="rate", color_continuous_scale="OrRd",
                labels={"rate": "Mismatch Rate (%)", "Issue_Category": ""}
            )
            fig2.update_layout(margin=dict(t=0, b=0, l=0, r=20), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig2, width='stretch')

        # Row 2: Severity delta heatmap
        st.subheader("🌡️ Severity Delta Heatmap (Category × Channel)")
        if "Ticket_Channel" in df.columns:
            hm = df[df["mismatch_label"] == 1].copy()
            hm["delta"] = hm["inferred_severity"] - hm["assigned_severity"]
            pivot = hm.pivot_table(
                index="Issue_Category", columns="Ticket_Channel",
                values="delta", aggfunc="mean"
            )
            fig3 = px.imshow(
                pivot, color_continuous_scale="RdBu_r",
                zmin=-2, zmax=2, text_auto=".2f",
                labels={"color": "Avg Δ"}
            )
            fig3.update_layout(height=350, margin=dict(t=20, b=0))
            st.plotly_chart(fig3, width='stretch')

        # Row 3: Confusion matrix + Scatter
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Assigned vs Inferred Priority")
            priority_cross = pd.crosstab(
                df[df["mismatch_label"] == 1]["Priority_Level"],
                df[df["mismatch_label"] == 1]["inferred_severity"].map(
                    lambda x: SEVERITY_NAMES.get(int(x), "?")
                )
            )
            fig4 = px.imshow(
                priority_cross, text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "Inferred Severity", "y": "Assigned Priority", "color": "Count"}
            )
            fig4.update_layout(height=320, margin=dict(t=20, b=0))
            st.plotly_chart(fig4, width='stretch')

        with col4:
            st.subheader("Resolution Time vs NLP Score (mismatch tickets)")
            sample = df[df["mismatch_label"] == 1].sample(
                min(1000, n_mismatch), random_state=42
            )
            sample["type"] = np.where(
                sample["inferred_severity"] > sample["assigned_severity"],
                "Hidden Crisis", "False Alarm"
            )
            fig5 = px.scatter(
                sample, x="kw_total", y="Resolution_Time_Hours",
                color="type", opacity=0.5, size_max=4,
                color_discrete_map={"Hidden Crisis": "#ef4444", "False Alarm": "#f59e0b"},
                labels={"kw_total": "Keyword Score", "Resolution_Time_Hours": "Res. Time (hrs)"},
                hover_data=["Ticket_ID", "Priority_Level"]
            )
            fig5.update_layout(height=320, margin=dict(t=20, b=0))
            st.plotly_chart(fig5, width='stretch')

        # Sample dossiers
        st.subheader("📄 Sample Evidence Dossiers")
        dossiers = load_dossiers()
        if dossiers:
            for d in dossiers[:3]:
                render_dossier(d)
        else:
            st.info("No dossiers found. Run `python train_pipeline.py` first.")

    # ═════════════════════════════════════════════════════════════════════════
    # 🔍 SINGLE TICKET
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Single Ticket":
        st.title("🔍 Single Ticket Analysis")
        st.caption("Enter ticket details to detect priority mismatches in real-time.")

        if not model_ready:
            st.error("Model not found. Run `python train_pipeline.py` first.")
            st.stop()

        with st.form("ticket_form"):
            st.markdown("### Ticket Details")
            col1, col2 = st.columns(2)
            with col1:
                subject  = st.text_input("Ticket Subject",
                                         "App crashing - Cannot access account")
                category = st.selectbox("Issue Category",
                                        ["Technical", "Billing", "Account", "Fraud", "General Inquiry"])
                priority = st.selectbox("Assigned Priority",
                                        ["Low", "Medium", "High", "Critical"], index=1)
                rt = st.number_input("Resolution Time (hours)", 1, 200, 24)
            with col2:
                channel = st.selectbox("Ticket Channel",
                                       ["Email", "Chat", "Web Form", "Phone"])
                sat = st.slider("Satisfaction Score", 1, 5, 3)
                ticket_id = st.text_input("Ticket ID", "MANUAL-001")

            desc = st.text_area(
                "Ticket Description",
                "I cannot log into my account. The application crashes every time "
                "I open it. This has been happening for 3 days and I need access "
                "immediately for work.",
                height=100
            )
            submitted = st.form_submit_button("🔍 Analyze Ticket", type="primary", width='stretch')

        if submitted:
            row_dict = {
                "Ticket_ID": ticket_id,
                "Ticket_Subject": subject,
                "Ticket_Description": desc,
                "Issue_Category": category,
                "Priority_Level": priority,
                "Ticket_Channel": channel,
                "Resolution_Time_Hours": int(rt),
                "Satisfaction_Score": int(sat),
                "Customer_Email": "user@example.com",
                "Customer_Name": "Form Input",
                "Submission_Date": "2025-01-01",
                "Assigned_Agent": "Agent",
            }

            with st.spinner("Analyzing ticket..."):
                pred, prob, row, dossier = score_single_ticket(row_dict, clf, scaler, encoders)

            # Result header
            if pred:
                mt = dossier["mismatch_type"]
                color = "#ef4444" if mt == "Hidden Crisis" else "#f59e0b"
                emoji = "🚨" if mt == "Hidden Crisis" else "⚠️"
                st.markdown(f"""
                <div class="metric-box {'crisis' if mt == 'Hidden Crisis' else 'alarm'}">
                    <b style="font-size:20px">{emoji} MISMATCH DETECTED</b><br>
                    {mismatch_badge(mt)}&nbsp; Confidence: <b>{prob*100:.1f}%</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-box pass">
                    <b style="font-size:20px">✅ CONSISTENT — No Mismatch Detected</b><br>
                    Confidence: <b>{prob*100:.1f}%</b>
                </div>""", unsafe_allow_html=True)

            # Metrics row
            inferred_lbl = SEVERITY_NAMES.get(int(row["inferred_severity"]), "Medium")
            delta_val = int(row["severity_delta"])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Assigned Priority", priority)
            col2.metric("Inferred Severity", inferred_lbl)
            col3.metric("Severity Delta",
                        f"+{delta_val}" if delta_val > 0 else str(delta_val),
                        delta_color="inverse" if delta_val < 0 else "normal")
            col4.metric("Confidence", f"{prob*100:.1f}%")

            col5, col6, col7 = st.columns(3)
            col5.metric("Keyword Score", f"{row['kw_total']:.2f}")
            col6.metric("Text Severity", f"{int(row['text_sev'])}/3")
            col7.metric("RT Severity", f"{int(row['rt_sev'])}/3")

            # Show dossier only when classifier flags a mismatch
            if pred:
                st.divider()
                st.subheader("📄 Evidence Dossier")
                render_dossier(dossier)

                with st.expander("🔗 Raw Dossier JSON"):
                    st.code(json.dumps(dossier, indent=2), language="json")
            else:
                # Classifier says consistent — note any pseudo-label disagreement
                if delta_val != 0:
                    st.info(
                        f"ℹ️ The pseudo-label signals suggest a potential "
                        f"**{dossier['mismatch_type']}** (Δ={delta_val:+d}), "
                        f"but the classifier disagrees with low confidence "
                        f"({prob*100:.1f}%). This ticket may warrant manual review."
                    )

    # ═════════════════════════════════════════════════════════════════════════
    # 📦 BATCH PREDICT
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📦 Batch Predict":
        st.title("📦 Batch CSV Analysis")
        st.info(
            "Upload a CSV with the same schema as the training dataset. "
            "Required columns: **Ticket_Subject**, **Ticket_Description**, "
            "**Priority_Level**, **Issue_Category**, **Ticket_Channel**, "
            "**Resolution_Time_Hours**."
        )

        if not model_ready:
            st.error("Model not found. Run `python train_pipeline.py` first.")
            st.stop()

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.info(f"Loaded **{len(df_up):,}** tickets")
            st.dataframe(df_up.head(5), width='stretch')

            if st.button("🚀 Run Prediction", type="primary", width='stretch'):
                with st.spinner(f"Scoring {len(df_up)} tickets..."):
                    df_result, sig_agree = score_dataframe(df_up, clf, scaler, encoders)

                mismatches = df_result[df_result["pred_mismatch"] == 1]
                n_m = len(mismatches)

                st.success(
                    f"✅ Scored! **{n_m}** mismatch(es) detected "
                    f"({n_m/len(df_result)*100:.1f}%) | "
                    f"Signal Agreement: {sig_agree:.3f}"
                )

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Tickets", len(df_result))
                col2.metric("Mismatches", n_m)
                col3.metric("Signal Agreement", f"{sig_agree:.3f}")

                # Distribution chart
                cnt = df_result["pred_label"].value_counts().reset_index()
                cnt.columns = ["Label", "Count"]
                fig = px.pie(
                    cnt, names="Label", values="Count", color="Label",
                    color_discrete_map={"Mismatch": "#ef4444", "Consistent": "#10b981"},
                    title="Prediction Distribution", hole=0.4
                )
                st.plotly_chart(fig, width='stretch')

                # Flagged tickets table
                if n_m > 0:
                    st.markdown("### 🚩 Flagged Tickets")
                    display_cols = [
                        c for c in [
                            "Ticket_ID", "Ticket_Subject", "Priority_Level",
                            "pred_label", "pred_confidence", "inferred_severity",
                            "severity_delta", "Issue_Category", "Ticket_Channel"
                        ] if c in mismatches.columns
                    ]
                    show = mismatches[display_cols].head(50).copy()
                    if "inferred_severity" in show.columns:
                        show["inferred_severity"] = show["inferred_severity"].map(SEVERITY_NAMES)
                    if "pred_confidence" in show.columns:
                        show["pred_confidence"] = (show["pred_confidence"] * 100).round(1).astype(str) + "%"
                    st.dataframe(show, width='stretch')

                    # Download buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        # Generate dossiers for download
                        dossiers = [
                            generate_dossier(row, float(row["pred_confidence"]))
                            for _, row in mismatches.head(100).iterrows()
                        ]
                        st.download_button(
                            "⬇️ Download Dossiers (JSON)",
                            data=json.dumps(dossiers, indent=2),
                            file_name="dossiers.json",
                            mime="application/json",
                            width='stretch'
                        )
                    with col_b:
                        csv_buf = mismatches[display_cols].to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Predictions (CSV)",
                            data=csv_buf,
                            file_name="predictions.csv",
                            mime="text/csv",
                            width='stretch'
                        )

    # ═════════════════════════════════════════════════════════════════════════
    # 📊 METHODOLOGY
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📊 Methodology":
        st.title("📊 Model Methodology & Architecture")

        # Architecture diagram
        st.subheader("🏗️ Pipeline Architecture")
        st.markdown("""
        ```
        Raw Tickets (CSV)
               │
               ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 1 — PSEUDO-LABEL GENERATION      │
        │                                         │
        │  Signal 1 (w=0.60): NLP Keyword Score   │
        │    • Urgency vs trivial keyword balance │
        │    • Subject ×0.7 + Description ×0.3    │
        │    → text_sev ∈ {0,1,2,3}               │
        │                                         │
        │  Signal 2 (w=0.40): Resolution-Time     │
        │    • RT / category_median (inverted)    │
        │    • Faster = more urgent               │
        │    → rt_sev ∈ {0,1,2,3}                 │
        │                                         │
        │  Fusion → inferred_severity             │
        │  Mismatch = |delta| ≥ 2                 │
        └─────────────────────────────────────────┘
               │  pseudo-labels
               ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 2 — XGBOOST CLASSIFIER           │
        │                                         │
        │  Features: TF-IDF (700-d) + struct (12) │
        │  Imbalance: SMOTE + scale_pos_weight    │
        │  → pred_mismatch + confidence           │
        └─────────────────────────────────────────┘
               │
               ▼
        ┌─────────────────────────────────────────┐
        │  STAGE 3 — EVIDENCE DOSSIER             │
        │                                         │
        │  • mismatch_type: Hidden Crisis | False │
        │  • feature_evidence: grounded, traced   │
        │  • constraint_analysis: 2-3 sentences   │
        └─────────────────────────────────────────┘
        ```
        """)

        st.divider()

        # Metrics
        metrics = load_metrics()
        if metrics:
            st.subheader("📐 Evaluation Metrics")
            THRESHOLDS = {"accuracy": 0.83, "macro_f1": 0.82,
                          "recall_consistent": 0.78, "recall_mismatch": 0.78}

            cols = st.columns(4)
            for i, (k, v) in enumerate(metrics.items()):
                t = THRESHOLDS.get(k)
                if t:
                    with cols[i % 4]:
                        st.metric(
                            k.replace("_", " ").title(),
                            f"{v*100 if k == 'accuracy' else v:.4f}",
                            delta=f"≥ {t}" if v >= t else f"< {t}"
                        )

        st.divider()

        # Ablation explanation
        st.subheader("🔬 Ablation: Why Two Signals?")
        st.markdown("""
        | Signal | Description | Role |
        |--------|-------------|------|
        | **NLP Keywords** (Signal 1) | Urgency + trivial keyword balance in ticket text | Primary severity signal — reads *what* the customer wrote |
        | **Resolution Time** (Signal 2) | Category-normalised RT, inverted (faster = more urgent) | Corroborating proxy — reads *how fast* it was handled |

        **Why fusion is necessary:**
        - Signal 1 alone: ~42% agreement with assigned priority — strong but noisy
        - Signal 2 alone: ~20% agreement with assigned priority — weak alone, valuable as corroboration
        - **Fused**: The two signals are genuinely independent (~25% pairwise agreement), so when they agree, confidence is high
        - Weight 0.60/0.40 chosen because NLP carries higher signal-to-noise than RT proxy
        """)

        st.divider()

        # Mismatch types
        st.subheader("🎯 Mismatch Types")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚨 Hidden Crisis
            **Inferred severity > Assigned priority**
            - Customer wrote urgent keywords but ticket was labeled low
            - Long resolution time corroborates elevated effort
            - **Risk**: SLA breach, customer churn, undetected critical issues
            """)
        with col2:
            st.markdown("""
            ### ⚠️ False Alarm
            **Inferred severity < Assigned priority**
            - Ticket labeled high/critical but NLP signal is weak
            - Short resolution time suggests routine issue
            - **Risk**: Wasted agent capacity, distorted SLA metrics, queue inflation
            """)

        st.divider()

        # Imbalance strategy
        st.subheader("⚖️ Class Imbalance Strategy")
        st.markdown("""
        - **SMOTE** oversampling applied to training set (never test set)
        - **`scale_pos_weight`** in XGBoost as secondary control
        - Final class ratio after SMOTE: 1:1 (Consistent : Mismatch)

        **Adversarial Robustness:**
        - Char n-gram features make the model robust to typos, leetspeak, and keyword obfuscation
        - TF-IDF weighting down-weights common words that adversaries might inject
        """)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
