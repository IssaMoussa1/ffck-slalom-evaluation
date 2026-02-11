import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="FFCK Slalom – Évaluation relative", layout="wide")
st.title("FFCK Slalom – Évaluation relative (V2)")
st.caption("Écart au meilleur (s et %), densité, IQC (si référence), ratios inter-catégories, export")

# -----------------------
# Sidebar parameters
# -----------------------
st.sidebar.header("Paramètres")
use_iqc = st.sidebar.checkbox("Activer IQC si référence disponible", value=True)

# -----------------------
# Uploads
# -----------------------
st.markdown("### 1) Charger les données")

need_upload = ("courses" not in st.session_state) or (st.session_state.get("courses") is None)

with st.expander("📂 Charger / modifier les fichiers (cliquer pour ouvrir/fermer)", expanded=need_upload):
    c_file = st.file_uploader("Résultats courses (CSV)", type=["csv"], key="courses")
    r_file = st.file_uploader("Références bassin (optionnel, CSV)", type=["csv"], key="refs")
    h_file = st.file_uploader("Ratios historiques (optionnel, CSV)", type=["csv"], key="ratios")

if c_file is None:
    st.info("Charge un fichier de résultats pour commencer.")
    st.stop()

courses = pd.read_csv(c_file)
refs = pd.read_csv(r_file) if r_file else None
ratios_hist = pd.read_csv(h_file) if h_file else None

required = ["event_id","date","bassin","run_id","categorie","athlete","time_final"]
missing = [c for c in required if c not in courses.columns]
if missing:
    st.error(f"Colonnes manquantes: {missing}")
    st.stop()

courses = courses.copy()
courses["time_final"] = pd.to_numeric(courses["time_final"], errors="coerce")
if "status" not in courses.columns:
    courses["status"] = np.where(courses["time_final"].notna(), "OK", "NA")

# -----------------------
# Selectors
# -----------------------
st.markdown("### 2) Sélection course")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    event_id = st.selectbox("event_id", sorted(courses["event_id"].unique()))

with c2:
    run_id = st.selectbox(
        "run_id",
        sorted(courses[courses["event_id"] == event_id]["run_id"].unique())
    )

with c3:
    bassin = st.selectbox(
        "bassin",
        sorted(courses[
            (courses["event_id"] == event_id) &
            (courses["run_id"] == run_id)
        ]["bassin"].unique())
    )

with c4:
    boat_choice = st.selectbox("Embarcation", ["Tous", "Kayak", "Canoë"], index=0)

with c5:
    sex_choice = st.selectbox("Sexe", ["Tous", "Homme", "Femme"], index=0)

# Base filtrée sur course
sub = courses[
    (courses["event_id"] == event_id) &
    (courses["run_id"] == run_id) &
    (courses["bassin"] == bassin)
].copy()

# Mapping catégories autorisées
cats_allowed = []
if boat_choice in ["Tous", "Kayak"] and sex_choice in ["Tous", "Homme"]:
    cats_allowed.append("K1H")
if boat_choice in ["Tous", "Kayak"] and sex_choice in ["Tous", "Femme"]:
    cats_allowed.append("K1F")
if boat_choice in ["Tous", "Canoë"] and sex_choice in ["Tous", "Homme"]:
    cats_allowed.append("C1H")
if boat_choice in ["Tous", "Canoë"] and sex_choice in ["Tous", "Femme"]:
    cats_allowed.append("C1F")

sub = sub[sub["categorie"].isin(cats_allowed)].copy()

sub_ok = sub[sub["status"]=="OK"].dropna(subset=["time_final"]).copy()

if sub_ok.empty:
    st.warning("Aucun temps valide (OK) sur cette sélection.")
    st.stop()

# -----------------------
# Helper functions
# -----------------------
def get_iqc(bassin_name, cat, best_time):
    if (not use_iqc) or (refs is None):
        return 1.0, None
    needed = {"bassin","categorie_ref","time_ref"}
    if not needed.issubset(set(refs.columns)):
        return 1.0, None
    r = refs[(refs["bassin"]==bassin_name) & (refs["categorie_ref"]==cat)]
    if r.empty:
        return 1.0, None
    time_ref = float(r.iloc[0]["time_ref"])
    return time_ref / best_time, time_ref

def density_norm(sd, best):
    # plus sd/best est faible → plus la densité est forte (0..1)
    if best <= 0 or np.isnan(sd):
        return 0.0
    x = sd / best
    return float(np.clip(1.0 / (1.0 + 300*x), 0.0, 1.0))

def format_time_mmss(x):
    """Convertit des secondes (float) en mm:ss.cc"""
    if pd.isna(x):
        return ""
    x = float(x)
    m = int(x // 60)
    s = x - 60*m
    return f"{m}:{s:05.2f}"  # ex: 1:35.27


# -----------------------
# Compute indicators
# -----------------------
cats = sorted(sub_ok["categorie"].unique())
res_frames = []
summary_rows = []

for cat in cats:
    s = sub_ok[sub_ok["categorie"]==cat].copy()
    best = float(s["time_final"].min())
    mean = float(s["time_final"].mean())
    sd = float(s["time_final"].std(ddof=1)) if len(s)>1 else 0.0

    s["rank"] = s["time_final"].rank(method="min").astype(int)
    s["pct_best"] = (s["time_final"] / best) * 100.0
    s["gap_s"] = s["time_final"] - best
    s["gap_pct"] = s["pct_best"] - 100.0
    s["time_final_display"] = s["time_final"].apply(format_time_mmss)
    s["gap_s_display"] = s["gap_s"].round(2).astype(str) + " s"
    s["gap_pct_display"] = s["gap_pct"].round(2).astype(str) + " %"


    iqc, time_ref = get_iqc(bassin, cat, best)
    dens = density_norm(sd, best)

    s["IQC"] = iqc
    s["density_norm"] = dens

    res_frames.append(s)

    summary_rows.append({
        "categorie": cat,
        "n_ok": int(len(s)),
        "best_time": round(best, 2),
        "mean_time": round(mean, 2),
        "sd_time": round(sd, 2),
        "density_norm": round(dens, 3),
        "IQC": round(iqc, 3),
        "time_ref": (None if time_ref is None else round(time_ref, 2))
    })

res = pd.concat(res_frames, ignore_index=True)

# -----------------------
# Ratios inter-catégories (calcul)
# -----------------------
def best_time(cat_name):
    x = sub_ok[sub_ok["categorie"]==cat_name]["time_final"]
    return float(x.min()) if len(x) else None

ratio_defs = [
    ("K1H/C1H","K1H","C1H"),
    ("K1F/C1H","K1F","C1H"),
    ("C1F/C1H","C1F","C1H")
]

rows = []
for ratio_name, num_cat, den_cat in ratio_defs:
    num = best_time(num_cat)
    den = best_time(den_cat)
    if num is None or den is None:
        continue
    today = num / den

    hist_mean = hist_sd = n = None
    if ratios_hist is not None and {"bassin","ratio_name","ratio_value"}.issubset(set(ratios_hist.columns)):
        rr = ratios_hist[(ratios_hist["bassin"]==bassin) & (ratios_hist["ratio_name"]==ratio_name)]
        if not rr.empty:
            hist_mean = float(rr.iloc[0]["ratio_value"])
            hist_sd = float(rr.iloc[0]["ratio_sd"]) if "ratio_sd" in rr.columns else None
            n = int(rr.iloc[0]["n_events"]) if "n_events" in rr.columns else None

    rows.append({
        "ratio": ratio_name,
        "today": round(today, 4),
        "hist_mean": (None if hist_mean is None else round(hist_mean, 4)),
        "hist_sd": (None if hist_sd is None else round(hist_sd, 4)),
        "n_events": n,
        "delta_vs_hist": (None if hist_mean is None else round(today - hist_mean, 4))
    })





# -----------------------
# Tabs: Tableaux / Visuels
# -----------------------
tab_table, tab_viz = st.tabs(["Tableaux", "Visuels"])

with tab_table:
    st.markdown("### 3) Synthèse par catégorie")
    summary_df = pd.DataFrame(summary_rows).rename(columns={
        "categorie": "Catégorie",
        "n_ok": "Nb athlètes classés",
        "best_time": "Meilleur temps (s)",
        "mean_time": "Temps moyen (s)",
        "sd_time": "Écart-type (s)",
        "density_norm": "Densité course",
        "IQC": "Qualité course",
        "time_ref": "Référence bassin (s)"
    })
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### 4) Performances individuelles (métriques enrichies)")
    for cat in cats:
        st.subheader(cat)
        view = res[res["categorie"]==cat][
            ["rank","athlete","time_final_display","gap_s_display","gap_pct_display","IQC","density_norm"]
        ].sort_values(["rank"])


        view_display = view.rename(columns={
            "rank": "Rang",
            "athlete": "Athlète",
            "time_final_display": "Temps final",
            "gap_s_display": "Écart au 1er",
            "gap_pct_display": "Écart (%)",
            "IQC": "Qualité course",
            "density_norm": "Densité course"
        })

        st.dataframe(view_display, use_container_width=True)

    st.markdown("### 5) Ratios inter-catégories (du jour + historique)")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Ratios non calculables (catégories manquantes).")

        st.markdown("### 6) Export")
    export_cols = [
        "event_id","date","bassin","run_id","categorie","athlete",
        "rank","time_final_display","gap_s_display","gap_pct_display",
        "IQC","density_norm"
    ]
    csv = res[export_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Télécharger résultats enrichis (CSV)",
        data=csv,
        file_name=f"eval_relative_{event_id}_{run_id}.csv",
        mime="text/csv"
    )

with tab_viz:
    st.markdown("### Visuels (lecture rapide commission)")

    # Choix Top N
    top_n = st.slider("Top N", min_value=3, max_value=20, value=10, step=1)

    # -----------------------
    # 1) Distribution (histogramme)
    # -----------------------
    st.markdown("#### 1) Distribution des temps (densité)")
    for cat in cats:
        s = res[res["categorie"]==cat].copy()
        fig = px.histogram(
            s, x="time_final",
            nbins=20,
            title=f"{cat} — Distribution des temps"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # 2) Top N vs reste : écart au 1er
    # -----------------------
    st.markdown("#### 2) Top N vs reste — Écart au meilleur (%)")
    for cat in cats:
        s = res[res["categorie"]==cat].copy().sort_values("gap_pct")
        best_time = float(s["time_final"].min())

        s["group"] = np.where(s["rank"] <= top_n, f"Top {top_n}", "Reste")
        # On visualise les gaps : top N en détail + reste agrégé (médiane)
        top_df = s[s["group"]==f"Top {top_n}"][["athlete","gap_pct","group"]].copy()

        reste = s[s["group"]=="Reste"]["gap_pct"]
        if len(reste) > 0:
            top_df = pd.concat([
                top_df,
                pd.DataFrame([{
                    "athlete": "Reste (médiane)",
                    "gap_pct": float(np.median(reste)),
                    "group": "Reste"
                }])
            ], ignore_index=True)

        fig = px.bar(
            top_df, x="athlete", y="gap_pct", color="group",
            title=f"{cat} — Écart au 1er : Top {top_n} vs reste",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Écart au meilleur (%)")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # 3) Top N vs reste : % du meilleur
    # -----------------------
    # 4) Ratios today vs historique
    # -----------------------
    st.markdown("#### 4) Ratios inter-catégories — today vs historique")
    if rows:
        ratios_df = pd.DataFrame(rows).copy()
        st.dataframe(ratios_df, use_container_width=True)

        # delta_vs_hist en bar
        if "delta_vs_hist" in ratios_df.columns:
            ratios_df["delta_vs_hist"] = pd.to_numeric(ratios_df["delta_vs_hist"], errors="coerce")
            fig = px.bar(
                ratios_df.dropna(subset=["delta_vs_hist"]),
                x="ratio", y="delta_vs_hist",
                title="Écart au ratio historique (today - hist)"
            )
            fig.update_layout(xaxis_title="", yaxis_title="Delta ratio")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ratios non disponibles pour cette sélection.")
