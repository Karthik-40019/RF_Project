
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import joblib
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RF Insight | Signal Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "signal_metrics.csv"

NOTEBOOK_MAP_DIR = ROOT / "notebooks" / "outputs" / "maps"
OUTPUT_MAP_DIR = ROOT / "outputs" / "maps"
OUTPUT_PLOTS_DIR = ROOT / "outputs" / "plots"

# ============================================================
# 2. THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #FFFDF7;
        color: #292929;
    }

    [data-testid="stSidebar"] {
        background: #FFF4C4;
        border-right: 1px solid #E9D58A;
    }

    [data-testid="stHeader"] {
        background: rgba(255,253,247,0.94);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #292929;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #EFE0A7;
        padding: 17px;
        border-radius: 15px;
        box-shadow: 0 2px 7px rgba(70,55,0,0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #746A4B;
    }

    div[data-testid="stMetricValue"] {
        color: #292929;
    }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #F0E7C8;
        border-radius: 14px;
        padding: 8px;
    }

    .hero {
        background: linear-gradient(
            120deg, #FFF0A6 0%, #FFF8D9 52%, #FFFFFF 100%
        );
        padding: 32px;
        border: 1px solid #E9D58A;
        border-radius: 20px;
        margin-bottom: 24px;
    }

    .hero-eyebrow {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #876500;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 38px;
        line-height: 1.15;
        font-weight: 750;
        color: #242424;
        margin: 0 0 12px 0;
    }

    .hero-description {
        color: #625A43;
        font-size: 16px;
        max-width: 760px;
        line-height: 1.7;
    }

    .hero-tag {
        display: inline-block;
        background: #F2CC45;
        color: #352900;
        padding: 9px 14px;
        border-radius: 9px;
        font-weight: 650;
        font-size: 12px;
        margin-top: 10px;
    }

    .insight-card {
        background: #FFFFFF;
        border: 1px solid #EFE0A7;
        border-radius: 14px;
        padding: 18px;
        min-height: 150px;
    }

    .insight-icon {
        font-size: 24px;
    }

    .insight-title {
        font-weight: 750;
        font-size: 16px;
        margin: 9px 0;
        color: #292929;
    }

    .muted {
        color: #766F5D;
        font-size: 13px;
    }

    hr {
        border-color: #EDE3C5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)

    numeric_columns = [
        "Latitude",
        "Longitude",
        "Signal Strength (dBm)",
        "Signal Quality (%)",
        "Data Throughput (Mbps)",
        "Latency (ms)",
        "BB60C Measurement (dBm)",
        "srsRAN Measurement (dBm)",
        "BladeRFxA9 Measurement (dBm)",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


if not DATA_PATH.is_file():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Expected: data/raw/signal_metrics.csv inside RF_Project."
    )
    st.stop()

df = load_data(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime,
)

required_columns = {
    "Latitude",
    "Longitude",
    "Signal Strength (dBm)",
    "Network Type",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(f"Required columns missing: {sorted(missing_columns)}")
    st.stop()

signal_data = df["Signal Strength (dBm)"].dropna()

# ============================================================
# 4. NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:12px 0 8px 0;">
        <div style="font-size:27px;font-weight:800;color:#282828;">
            📡 RF Insight
        </div>
        <div style="font-size:12px;color:#746747;margin-top:5px;">
            SIGNAL ANALYSIS & COVERAGE INTELLIGENCE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Data Explorer",
        "Signal Analysis",
        "Coverage Map",
        "Model Evaluation",
        "ML Predictions",
        "About Project",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("**PROJECT STATUS**")
st.sidebar.caption("Dataset loaded successfully")
st.sidebar.caption(f"{len(df):,} measurement records")
st.sidebar.markdown("---")
st.sidebar.caption("RF Insight • Academic Project")

# ============================================================
# 5. OVERVIEW — PREMIUM DASHBOARD
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">
                RF INTELLIGENCE PLATFORM
            </div>
            <div class="hero-title">
                Understand your RF environment.
            </div>
            <div class="hero-description">
                Explore measurement data, discover network patterns,
                investigate geographical coverage, and evaluate
                machine-learning predictions in one workspace.
            </div>
            <div class="hero-tag">
                📡 RF SIGNAL STRENGTH ANALYSIS & COVERAGE VISUALIZATION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(df)
    total_missing = int(df.isna().sum().sum())
    duplicate_records = int(df.duplicated().sum())

    mean_signal = (
        signal_data.mean() if not signal_data.empty else np.nan
    )
    median_signal = (
        signal_data.median() if not signal_data.empty else np.nan
    )

    st.markdown("### Project at a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Measurements",
        f"{total_records:,}",
        help="Number of rows loaded from the CSV.",
    )

    c2.metric(
        "Average Signal",
        f"{mean_signal:.2f} dBm" if pd.notna(mean_signal) else "N/A",
        help="Mean of available signal-strength measurements.",
    )

    c3.metric(
        "Median Signal",
        f"{median_signal:.2f} dBm" if pd.notna(median_signal) else "N/A",
        help="Median of available signal-strength measurements.",
    )

    c4.metric(
        "Missing Cells",
        f"{total_missing:,}",
        help="Total missing cells across the loaded dataset.",
    )

    st.write("")

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.markdown("### Signal distribution")
        st.caption(
            "Understand the spread and frequency of measured signal values."
        )

        if not signal_data.empty:
            fig = px.histogram(
                x=signal_data,
                nbins=35,
                color_discrete_sequence=["#E8B923"],
                labels={"x": "Signal Strength (dBm)", "count": "Records"},
            )

            fig.update_traces(
                marker_line_color="#FFFFFF",
                marker_line_width=0.7,
                hovertemplate=(
                    "Signal: %{x:.2f} dBm"
                    "<br>Records: %{y}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_white",
                height=365,
                bargap=0.08,
                margin=dict(l=15, r=15, t=15, b=15),
                xaxis_title="Signal Strength (dBm)",
                yaxis_title="Number of measurements",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.warning("No valid signal measurements are available.")

    with right:
        st.markdown("### Network landscape")
        st.caption("Share of records in each network category.")

        network_data = (
            df["Network Type"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Network Type")
            .reset_index(name="Measurements")
        )

        if not network_data.empty:
            fig = px.pie(
                network_data,
                names="Network Type",
                values="Measurements",
                hole=0.64,
                color_discrete_sequence=[
                    "#E8B923",
                    "#F4D875",
                    "#83B692",
                    "#7DA7D9",
                    "#C3A6E8",
                    "#E89A75",
                ],
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate=(
                    "%{label}<br>Records: %{value:,}"
                    "<br>Share: %{percent}<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_white",
                height=365,
                margin=dict(l=10, r=10, t=15, b=15),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.06,
                    xanchor="center",
                    x=0.5,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info("No network categories are available.")

    # ---------- DATA-DRIVEN INSIGHTS ----------

    st.markdown("---")
    st.markdown("### Dataset health & insights")

    i1, i2, i3 = st.columns(3, gap="medium")

    with i1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-icon">🧹</div>
                <div class="insight-title">Data completeness</div>
                <div class="muted">
                    {f"No missing cells detected." if total_missing == 0
                    else f"{total_missing:,} missing cells need investigation."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i2:
        if not signal_data.empty:
            signal_message = (
                f"Observed range: {signal_data.min():.2f} to "
                f"{signal_data.max():.2f} dBm."
            )
        else:
            signal_message = "No valid signal measurements available."

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-icon">📡</div>
                <div class="insight-title">Signal summary</div>
                <div class="muted">{signal_message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i3:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-icon">🔎</div>
                <div class="insight-title">Exact duplicates</div>
                <div class="muted">
                    {duplicate_records:,} exact duplicate rows detected
                    in the currently loaded dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- NETWORK SUMMARY ----------

    st.markdown("---")
    st.markdown("### Network performance snapshot")

    summary_columns = [
        column for column in [
            "Signal Strength (dBm)",
            "Data Throughput (Mbps)",
            "Latency (ms)",
        ]
        if column in df.columns
    ]

    if summary_columns:
        summary_df = (
            df.groupby("Network Type", dropna=False)[summary_columns]
            .mean()
            .round(2)
            .reset_index()
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These averages describe the collected dataset. "
            "They are not universal rankings of network technologies."
        )

    with st.expander("Preview measurement records"):
        st.dataframe(
            df.head(8),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# 6. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("Dataset Explorer")
    st.caption(
        "Inspect, filter, summarize and export the original RF dataset."
    )

    filtered = df.copy()

    network_options = sorted(
        df["Network Type"].dropna().astype(str).unique().tolist()
    )

    selected_networks = st.multiselect(
        "Network Type",
        options=network_options,
        default=network_options,
    )

    filtered = filtered[
        filtered["Network Type"].astype(str).isin(selected_networks)
    ]

    if "Locality" in filtered.columns:
        locality_options = sorted(
            filtered["Locality"].dropna().astype(str).unique().tolist()
        )

        selected_localities = st.multiselect(
            "Locality",
            options=locality_options,
        )

        if selected_localities:
            filtered = filtered[
                filtered["Locality"].astype(str).isin(selected_localities)
            ]

    a, b, c = st.columns(3)
    a.metric("Matching Records", f"{len(filtered):,}")
    b.metric("Columns", f"{len(filtered.columns)}")
    c.metric("Exact Duplicate Rows", f"{int(filtered.duplicated().sum()):,}")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=430,
    )

    st.download_button(
        "Download Filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_rf_measurements.csv",
        mime="text/csv",
    )

    st.markdown("### Column information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique(dropna=False).values,
    })

    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("### Statistical summary")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True,
    )

# ============================================================
# 7. SIGNAL ANALYSIS — ORGANIZED ANALYSIS BLOCKS
# ============================================================

elif page == "Signal Analysis":

    st.title("Signal Analysis")
    st.caption(
        "A structured view of statistical quality, network performance, "
        "and relationships in the RF measurement dataset."
    )

    # Keep the page focused on analyses of the measured data.
    # Geographic plots are grouped under Coverage Map; model diagnostics
    # are grouped under Model Evaluation to avoid duplicating charts.

    block1, block2 = st.tabs([
        "Distribution & Statistical Quality",
        "Signal & Network Performance",
    ])

    # --------------------------------------------------------
    # BLOCK 1: SIGNAL DISTRIBUTION, NORMALITY AND OUTLIERS
    # Steps 8, 25 and 26
    # --------------------------------------------------------
    with block1:
        st.subheader("Signal Distribution & Statistical Quality")
        st.write(
            "Understand the spread of measured signal values and identify "
            "statistical properties or unusual observations."
        )

        signal_valid = df["Signal Strength (dBm)"].dropna()

        if signal_valid.empty:
            st.warning("No valid signal-strength values are available.")
        else:
            left, right = st.columns([1.5, 1], gap="large")

            with left:
                st.markdown("#### Signal strength distribution")
                fig = px.histogram(
                    x=signal_valid,
                    nbins=35,
                    color_discrete_sequence=["#E8B923"],
                    labels={"x": "Signal Strength (dBm)", "count": "Measurements"},
                )
                fig.update_layout(
                    template="plotly_white",
                    height=390,
                    bargap=0.08,
                    xaxis_title="Signal Strength (dBm)",
                    yaxis_title="Number of measurements",
                    margin=dict(l=15, r=15, t=25, b=15),
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("#### Descriptive statistics")
                stats = pd.DataFrame({
                    "Statistic": ["Count", "Mean", "Median", "Std. deviation", "Minimum", "25th percentile", "75th percentile", "Maximum"],
                    "Value": [
                        signal_valid.count(), signal_valid.mean(), signal_valid.median(),
                        signal_valid.std(), signal_valid.min(), signal_valid.quantile(0.25),
                        signal_valid.quantile(0.75), signal_valid.max(),
                    ],
                })
                stats["Value"] = stats["Value"].map(lambda v: f"{v:,.3f}")
                st.dataframe(stats, use_container_width=True, hide_index=True)
                st.caption("These statistics describe the available measurements; they do not imply a normal distribution.")

            st.markdown("#### Normality assessment")
            st.write(
                "Normality was tested in the notebook using the D’Agostino–Pearson test. "
                "The recorded test rejected normality (p-value approximately 3.16 × 10⁻⁸⁴). "
                "With a large dataset, even modest departures from a normal distribution can be statistically significant. "
                "Random Forest regression does not require the target to be normally distributed."
            )

            st.markdown("#### IQR outlier assessment")
            q1 = signal_valid.quantile(0.25)
            q3 = signal_valid.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (signal_valid < lower_bound) | (signal_valid > upper_bound)
            o1, o2, o3 = st.columns(3)
            o1.metric("Lower IQR bound", f"{lower_bound:.2f} dBm")
            o2.metric("Upper IQR bound", f"{upper_bound:.2f} dBm")
            o3.metric("Potential outliers", f"{int(outlier_mask.sum()):,}")
            st.caption(
                "IQR flags potential outliers; it does not prove they are errors. "
                "Unusually weak readings may be real RF observations, so they should not be removed automatically."
            )

    # --------------------------------------------------------
    # BLOCK 2: SIGNAL QUALITY, THROUGHPUT, LATENCY, NETWORKS
    # Steps 9, 10, 11, 20.2 and 21
    # --------------------------------------------------------
    with block2:
        st.subheader("Signal & Network Performance")
        st.write(
            "Explore how signal strength relates to throughput and latency, "
            "and compare the measured signal distributions across network types."
        )

        # Step 9: keep the analysis documented, but explain the constant column.
        st.markdown("#### Signal strength vs signal quality")
        if "Signal Quality (%)" in df.columns:
            quality_values = df["Signal Quality (%)"].dropna().unique()
            if len(quality_values) <= 1:
                st.warning(
                    "Signal Quality (%) has no variation in this dataset "
                    f"(observed values: {quality_values.tolist()}). "
                    "A relationship plot would not provide meaningful insight, "
                    "so it is intentionally not displayed."
                )
            else:
                fig = px.scatter(
                    df.dropna(subset=["Signal Strength (dBm)", "Signal Quality (%)"]),
                    x="Signal Strength (dBm)", y="Signal Quality (%)",
                    color="Network Type" if "Network Type" in df.columns else None,
                    opacity=0.65,
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Signal Quality (%) is not present in the dataset.")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Signal strength vs throughput")
            if "Data Throughput (Mbps)" in df.columns:
                plot_df = df.dropna(subset=["Signal Strength (dBm)", "Data Throughput (Mbps)"]).copy()
                fig = px.scatter(
                    plot_df,
                    x="Signal Strength (dBm)",
                    y="Data Throughput (Mbps)",
                    color="Network Type" if "Network Type" in plot_df.columns else None,
                    opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_layout(template="plotly_white", height=390)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Throughput data is unavailable.")

        with right:
            st.markdown("#### Signal strength vs latency")
            if "Latency (ms)" in df.columns:
                plot_df = df.dropna(subset=["Signal Strength (dBm)", "Latency (ms)"]).copy()
                fig = px.scatter(
                    plot_df,
                    x="Signal Strength (dBm)",
                    y="Latency (ms)",
                    color="Network Type" if "Network Type" in plot_df.columns else None,
                    opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_layout(template="plotly_white", height=390)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Latency data is unavailable.")

        st.markdown("#### Signal strength by network type")
        box_df = df.dropna(subset=["Network Type", "Signal Strength (dBm)"]).copy()
        if not box_df.empty:
            fig = px.box(
                box_df,
                x="Network Type",
                y="Signal Strength (dBm)",
                color="Network Type",
                color_discrete_sequence=["#E8B923", "#E6A23C", "#80B918", "#4D908E", "#5B8DB8"],
            )
            fig.update_layout(template="plotly_white", showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("There are no usable network-type signal records to compare.")

        st.markdown("#### Correlation heatmap")
        numeric = df.select_dtypes(include=np.number)
        corr_columns = [
            col for col in [
                "Signal Strength (dBm)",
                "Data Throughput (Mbps)",
                "Latency (ms)",
                "Latitude",
                "Longitude",
            ] if col in numeric.columns
        ]
        if len(corr_columns) >= 2:
            corr = numeric[corr_columns].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdYlBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            fig.update_layout(template="plotly_white", height=460)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Correlation indicates association, not causation. Constant columns are excluded because they cannot provide a useful correlation.")
        else:
            st.info("Not enough numeric columns are available for correlation analysis.")

    st.info(
        "Geographical analyses (Steps 12, 13, 15.2, 16, 17 and 23) are grouped under Coverage Map. "
        "Model diagnostics (Steps 31, 32 and 35) are grouped under Model Evaluation so the same plots are not duplicated here."
    )

# ============================================================
# 8. COVERAGE MAP
# ============================================================

elif page == "Coverage Map":

    st.title("Coverage Studio")
    st.caption(
        "Explore the existing interactive measurement map and "
        "any saved continuous coverage visualization."
    )

    available_maps = {}

    candidate_maps = [
        (
            "Measurement Marker Map",
            NOTEBOOK_MAP_DIR / "rf_coverage_map_esri.html",
        ),
        (
            "Continuous Coverage Heatmap",
            NOTEBOOK_MAP_DIR / "rf_coverage_heatmap.html",
        ),
        (
            "Measurement Marker Map",
            OUTPUT_MAP_DIR / "rf_coverage_map_esri.html",
        ),
        (
            "Continuous Coverage Heatmap",
            OUTPUT_MAP_DIR / "rf_coverage_heatmap.html",
        ),
    ]

    for name, path in candidate_maps:
        if path.is_file() and name not in available_maps:
            available_maps[name] = path

    if available_maps:
        map_choice = st.selectbox(
            "Visualization layer",
            list(available_maps.keys()),
        )

        selected_path = available_maps[map_choice]

        st.caption(f"Source file: {selected_path.relative_to(ROOT)}")

        html = selected_path.read_text(encoding="utf-8")

        components.html(
            html,
            height=700,
            scrolling=True,
        )
    else:
        st.warning("No saved interactive HTML map was found.")
        st.code(
            f"{NOTEBOOK_MAP_DIR}\n{OUTPUT_MAP_DIR}",
            language="text",
        )

    
    # ---------- SAVED ANALYSIS VISUALIZATIONS ----------

    st.markdown("### Saved Analysis Visualizations")

    image_headings = {
        "Step-12.png": "Geographical Distribution of RF Measurements",
        "Step-13.png": "Geographical RF Signal Strength Map",
        "Step-16.png": "Geographical RF Coverage by Signal Category",
        "Step-17.png": "Interpolated RF Signal Strength Heatmap",
        "Step-23.png": "Regional Average RF Signal Map",
    }

    if OUTPUT_MAP_DIR.is_dir():

        image_files = [
            path for path in sorted(OUTPUT_MAP_DIR.glob("Step-*.png"))
            if path.name in image_headings
        ]

        if image_files:

            image_lookup = {
                image_headings[path.name]: path
                for path in image_files
            }

            selected_heading = st.selectbox(
                "Choose a visualization",
                list(image_lookup.keys()),
            )

            selected_image = image_lookup[selected_heading]

            st.image(
                str(selected_image),
                caption=selected_heading,
                use_container_width=True,
            )

        else:
            st.info("No recognized analysis images were found.")
    else:
        st.info("The saved analysis images folder does not exist.")


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")
    st.caption(
        "Results from the completed notebook evaluation. "
        "These are recorded results, not freshly computed predictions."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Mean Absolute Error", "3.2516 dB")
    m2.metric("Root Mean Squared Error", "4.1365 dB")
    m3.metric("R² Score", "0.3945")

    st.markdown("### Random Forest vs baseline")

    comparison = pd.DataFrame({
        "Metric": ["MAE (dB)", "RMSE (dB)", "R² Score"],
        "Random Forest": [3.2516, 4.1365, 0.3945],
        "Baseline Model": [4.2540, 5.3163, -0.0001],
    })

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    fig = px.bar(
        comparison[comparison["Metric"] != "R² Score"].melt(
            id_vars="Metric",
            var_name="Model",
            value_name="Error (dB)",
        ),
        x="Metric",
        y="Error (dB)",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#E8B923", "#8C929C"],
    )

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature ablation")

    ablation = pd.DataFrame({
        "Metric": ["MAE (dB)", "RMSE (dB)", "R² Score"],
        "All Features": [3.2516, 4.1365, 0.3945],
        "Without Network Type": [3.5902, 4.5145, 0.2789],
    })

    st.dataframe(ablation, use_container_width=True, hide_index=True)

    st.info(
        "Removing Network Type worsened all three recorded test-set "
        "metrics. This suggests that it provides useful predictive "
        "information in this experiment."
    )

    st.markdown("### Interpretation")

    st.write(
        "The Random Forest model improved on the mean-prediction baseline. "
        "Its R² score indicates that a substantial proportion of signal "
        "strength variation remains unexplained. Performance should "
        "therefore be interpreted cautiously, particularly for new "
        "geographical areas."
    )




# ============================================================
# ML PREDICTIONS — PREDICTION & ACTUAL-VALUE COMPARISON
# ============================================================

elif page == "ML Predictions":

    st.title("ML Prediction Lab")
    st.caption(
        "Predict signal strength for new inputs or evaluate predictions "
        "against actual measurements from the held-out test set."
    )

    model_path = ROOT / "models" / "rf_signal_model_compressed.joblib"
    test_path = ROOT / "models" / "rf_test_records.csv"

    if not model_path.exists():
        st.error("Trained model not found. Run Step 38 first.")
        st.stop()

    @st.cache_resource
    def load_trained_model(path, modified_time):
        return joblib.load(path)

    trained_model = load_trained_model(
        str(model_path),
        model_path.stat().st_mtime,
    )

    feature_columns = [
        "Latitude",
        "Longitude",
        "Data Throughput (Mbps)",
        "Latency (ms)",
        "Network Type",
    ]

    mode = st.radio(
        "Choose prediction mode",
        [
            "Predict New Input",
            "Compare With Actual Measurement",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # MODE 1: PREDICT NEW INPUT
    # --------------------------------------------------------

    if mode == "Predict New Input":

        st.info(
            "For a genuinely new location, the actual signal strength "
            "is unknown until a real measurement is collected."
        )

        with st.form("new_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                latitude = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(df["Latitude"].median()),
                    format="%.6f",
                )

                longitude = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=float(df["Longitude"].median()),
                    format="%.6f",
                )

                throughput = st.number_input(
                    "Throughput (Mbps)",
                    min_value=0.0,
                    value=float(max(
                        0, df["Data Throughput (Mbps)"].median()
                    )),
                )

            with col2:
                latency = st.number_input(
                    "Latency (ms)",
                    min_value=0.0,
                    value=float(max(
                        0, df["Latency (ms)"].median()
                    )),
                )

                network_types = sorted(
                    df["Network Type"].dropna().astype(str).unique()
                )

                network_type = st.selectbox(
                    "Network Type",
                    network_types,
                )

            submitted = st.form_submit_button(
                "Predict Signal Strength",
                use_container_width=True,
            )

        if submitted:

            input_data = pd.DataFrame([{
                "Latitude": latitude,
                "Longitude": longitude,
                "Data Throughput (Mbps)": throughput,
                "Latency (ms)": latency,
                "Network Type": network_type,
            }])

            prediction = float(
                trained_model.predict(input_data)[0]
            )

            st.markdown("### Prediction result")

            st.metric(
                "Predicted Signal Strength",
                f"{prediction:.2f} dBm",
            )

            st.caption(
                "Individual prediction error is unknown without an "
                "actual measurement. Recorded test-set MAE: 3.2516 dB."
            )

    # --------------------------------------------------------
    # MODE 2: COMPARE AGAINST ACTUAL MEASUREMENT
    # --------------------------------------------------------

    else:

        if not test_path.exists():
            st.error(
                "Test records not found. Run Step 39 in the notebook "
                "to create models/rf_test_records.csv."
            )
            st.stop()

        test_df = pd.read_csv(test_path)

        required_columns = feature_columns + [
            "Actual Signal Strength (dBm)"
        ]

        if not all(
            column in test_df.columns for column in required_columns
        ):
            st.error(
                "The test CSV is missing required columns. "
                "Please rerun Step 39."
            )
            st.stop()

        st.markdown("### Select a held-out test record")

        record_index = st.selectbox(
            "Test record",
            options=list(range(len(test_df))),
            format_func=lambda i: f"Test record {i + 1}",
        )

        record = test_df.iloc[record_index]

        # Only input features are passed to the model.
        input_data = pd.DataFrame([{
            column: record[column]
            for column in feature_columns
        }])

        st.markdown("#### Input features")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

        if st.button(
            "Compare Actual vs Predicted",
            use_container_width=True,
        ):

            actual = float(record["Actual Signal Strength (dBm)"])
            predicted = float(trained_model.predict(input_data)[0])

            signed_error = predicted - actual
            absolute_error = abs(signed_error)

            st.markdown("---")
            st.markdown("### Validation result")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Actual Signal",
                f"{actual:.2f} dBm",
            )

            c2.metric(
                "Predicted Signal",
                f"{predicted:.2f} dBm",
            )

            c3.metric(
                "Absolute Error",
                f"{absolute_error:.2f} dB",
            )

            comparison_df = pd.DataFrame({
                "Value": ["Actual", "Predicted"],
                "Signal Strength (dBm)": [actual, predicted],
            })

            fig = px.bar(
                comparison_df,
                x="Value",
                y="Signal Strength (dBm)",
                color="Value",
                color_discrete_map={
                    "Actual": "#E8B923",
                    "Predicted": "#8C929C",
                },
                text_auto=".2f",
                title="Actual vs Predicted Signal Strength",
            )

            fig.update_layout(
                template="plotly_white",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.write(
                f"Signed error (predicted − actual): "
                f"{signed_error:+.2f} dB"
            )

            st.caption(
                "Absolute error measures the difference for this record. "
                "The overall test-set MAE of 3.2516 dB is calculated "
                "across all held-out test records."
            )


# ============================================================
# 11. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("About RF Insight")

    st.markdown(
        """
        ## RF Signal Strength Analysis and Coverage Visualization

        ### Objective
        Analyze RF measurement data, discover statistical patterns,
        visualize geographical coverage, and evaluate a machine-learning
        model for signal-strength prediction.

        ### Workflow
        1. Dataset understanding and preprocessing
        2. Statistical analysis and outlier investigation
        3. Geographic analysis and coverage visualization
        4. Random Forest regression
        5. Model evaluation and feature importance
        6. Interactive web dashboard

        ### Recorded findings
        - 16,829 records in the analyzed dataset.
        - Random Forest MAE: 3.2516 dB.
        - Random Forest RMSE: 4.1365 dB.
        - Random Forest R²: 0.3945.
        - Network Type was the most important feature in the recorded
          permutation-importance analysis.

        ### Limitations
        - Interpolated values are estimates, not direct measurements.
        - Model quality depends on dataset coverage and quality.
        - Generalization to unseen areas requires appropriate spatial
          validation.
        - Throughput and latency may not be available in all future
          prediction scenarios.
        """
    )

# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "RF Insight • RF Signal Strength Analysis and Coverage Visualization"
)
