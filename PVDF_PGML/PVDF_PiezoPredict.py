import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="PVDF Composite Piezoelectric Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Paths ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
materials_properties_path = os.path.join(current_dir, 'materials_properties.py')
checkpoint_path = os.path.join(current_dir, 'best_phys_resid_monotonic_improved_v2.pt')
predictor_path = os.path.join(current_dir, 'piezoelectric_tensor_predictor.py')

# ---------- Constants ----------
FOOTNOTE_TEXT = """
**Reference:** This work is based on the following paper (yet to be published):
"Phase Characterization, Enhanced Piezoelectric Performance, and Device Potential of Electrospun PVDF/SnO2 Nanofibers via Physics-Guided Machine Learning"
Sachin Poudel,∗, Weronika Smok, Rubi Thapa, Anna Timofiejczuk, Nele Moelans and Anil Kunwar
"""

# ---------- CSS (Minimal & Elegant) ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1a2980, #26d0ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a2980;
        margin: 1rem 0 0.8rem 0;
        border-left: 4px solid #26d0ce;
        padding-left: 0.8rem;
    }
    .metric-card {
        background: white;
        border-radius: 1rem;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    .property-card {
        background: #f8fafc;
        border-radius: 0.8rem;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #26d0ce;
    }
    .property-name {
        font-size: 0.85rem;
        color: #4b5563;
        letter-spacing: 0.5px;
    }
    .property-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a2980;
    }
    .stButton > button {
        background: linear-gradient(100deg, #1a2980, #26d0ce);
        color: white;
        font-weight: 500;
        border-radius: 2rem;
        padding: 0.5rem 1.2rem;
        border: none;
        transition: 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        opacity: 0.95;
    }
    .welcome-card {
        background: linear-gradient(145deg, #f9fafb, #ffffff);
        border-radius: 1.5rem;
        padding: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    .file-status {
        background: #f1f5f9;
        border-radius: 1rem;
        padding: 0.8rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
@st.cache_resource
def load_materials():
    """Load materials properties once and cache."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("materials_properties", materials_properties_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.properties

def display_property_card(name, value):
    st.markdown(f"""
    <div class="property-card">
        <div class="property-name">{name}</div>
        <div class="property-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def display_metric(label, value, unit="pC/N"):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1rem; font-weight:500; color:#1a2980;">{label}</div>
        <div style="font-size:2rem; font-weight:700; color:#26d0ce;">{value:.2f} {unit}</div>
    </div>
    """, unsafe_allow_html=True)

def show_footer():
    st.markdown(f'<div class="footer"><p>{FOOTNOTE_TEXT.replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")

    # Load properties
    try:
        properties = load_materials()
        fillers = [f for f in properties.keys() if f != 'PVDF']
    except Exception as e:
        st.error(f"Failed to load materials: {e}")
        st.stop()

    selected_filler = st.selectbox("Filler", fillers, index=fillers.index('SnO2') if 'SnO2' in fillers else 0)
    dopant_fraction = st.slider("Dopant Fraction (%)", 0.02, 10.0, 1.5, 0.1)
    fabrication_method = st.selectbox("Fabrication Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
    beta_fraction = st.number_input("Beta Fraction (optional override)", 0.0, 1.0, 0.5725, 0.01,
                                    help="Leave as 0.0 to use calculated value. Values >0.6 are damped.")

    # Damping logic
    DAMPING_FACTOR = 0.3
    effective_beta = beta_fraction if beta_fraction <= 0.6 else 0.6 + (beta_fraction - 0.6) * DAMPING_FACTOR

    # Filler properties preview
    if selected_filler in properties:
        st.markdown("### 🧪 Filler Properties")
        filler_df = pd.DataFrame([{k:v for k,v in properties[selected_filler].items() if not k.startswith('_')}])
        st.dataframe(filler_df.T, use_container_width=True)  # Fixed: removed header=False

    predict_clicked = st.button("🔮 Predict Properties", type="primary", use_container_width=True)

    # File status
    with st.expander("📁 File Status", expanded=False):
        for name, path in [("materials_properties.py", materials_properties_path),
                           ("model checkpoint", checkpoint_path),
                           ("predictor module", predictor_path)]:
            if os.path.exists(path):
                st.success(f"✓ {name}")
            else:
                st.error(f"✗ {name} missing")

# ---------- Main Area ----------
st.markdown('<h1 class="main-header">PVDF Composite Piezoelectric Predictor</h1>', unsafe_allow_html=True)
st.caption("Physics‑informed machine learning for PVDF‑based composites")

if not predict_clicked:
    # Welcome screen
    st.markdown(f"""
    <div class="welcome-card">
        <h2 style="color:#1a2980;">✨ Welcome</h2>
        <p>Predict piezoelectric coefficients (<i>d</i><sub>33</sub>, <i>d</i><sub>31</sub>, <i>d</i><sub>15</sub>, ...) for PVDF composites using a physics‑guided neural network.</p>
        <ul>
            <li>Choose a filler and its concentration</li>
            <li>Select fabrication method</li>
            <li>Optionally override β‑phase fraction</li>
            <li>Click <strong>Predict Properties</strong></li>
        </ul>
        <p><strong>Available fillers:</strong> {', '.join(fillers)}</p>
    </div>
    """, unsafe_allow_html=True)
    show_footer()
    st.stop()

# ---------- Prediction ----------
with st.spinner("Calculating piezoelectric response..."):
    try:
        sys.path.insert(0, current_dir)
        from piezoelectric_tensor_predictor import predict_sample

        df = predict_sample(
            checkpoint_path=checkpoint_path,
            dopant=selected_filler,
            frac=dopant_fraction,
            method=fabrication_method,
            beta_fraction=effective_beta if effective_beta > 0 else None,
            device='cpu'
        )
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([df])

        # Extract coefficients
        d33 = float(df['predicted_d33'].iloc[0])
        d31 = float(df['phys_d31'].iloc[0])
        d32 = float(df['phys_d32'].iloc[0])
        d15 = float(df['phys_d15'].iloc[0])
        d24 = float(df['phys_d24'].iloc[0])

        # Build feature dictionary
        features = {
            "Dopant": selected_filler,
            "Dopant fraction": f"{dopant_fraction:.1f} %",
            "Fabrication": fabrication_method,
            "Input β": f"{beta_fraction:.4f}",
            "Effective β (damped)": f"{effective_beta:.4f}",
            "Model β used": f"{df['PVDF_Beta_Fraction_used'].iloc[0]:.4f}",
            "Dielectric constant εᵣ": f"{df['Effective Dielectric Constant'].iloc[0]:.2f}",
            "Young's modulus": f"{df['Effective Youngs Modulus'].iloc[0]:.1f} GPa",
            "Poisson's ratio": f"{df['Effective Poissons Ratio'].iloc[0]:.3f}",
            "Physics base d33": f"{df['physics_base_d33'].iloc[0]:.2f} pC/N",
            "Learned Δd33": f"{df['learned_delta_d33'].iloc[0]:.2f} pC/N",
            "Filler category": properties.get(selected_filler, {}).get('Filler_Category', '—')
        }

        # Piezoelectric tensor (Voigt notation)
        tensor = np.array([
            [0, 0, 0, 0, d15, 0],
            [0, 0, 0, d24, 0, 0],
            [d31, d32, -d33, 0, 0, 0]
        ])
        tensor_df = pd.DataFrame(tensor, index=["d₁ (X)", "d₂ (Y)", "d₃ (Z)"],
                                 columns=["1", "2", "3", "4", "5", "6"])

        st.success("✅ Prediction successful")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

# ---------- Results Display ----------
st.markdown('<div class="sub-header">📊 Piezoelectric Coefficients</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    display_metric("d₃₃ (longitudinal)", d33)
    display_metric("d₃₁ (transverse)", d31)
with col2:
    display_metric("d₁₅ (shear)", d15)
    display_metric("d₂₄ (shear)", d24)

# Tensor visualization
st.markdown('<div class="sub-header">🔢 Piezoelectric Tensor (Voigt notation)</div>', unsafe_allow_html=True)

def style_tensor(val):
    if abs(val) < 1e-6:
        return 'background-color: #eef2ff; color: #1e3a8a; font-weight:500'
    return 'background-color: #fffbeb; color: #b45309; font-weight:500' if val > 0 else 'background-color: #fee2e2; color: #b91c1c; font-weight:500'

styled_tensor = tensor_df.style.map(style_tensor).format("{:.2f}")
st.dataframe(styled_tensor, use_container_width=True)

# Additional properties
st.markdown('<div class="sub-header">📋 Material Properties</div>', unsafe_allow_html=True)
prop_cols = st.columns(2)
items = list(features.items())
mid = len(items)//2 + (len(items)%2)
for i, (key, val) in enumerate(items[:mid]):
    with prop_cols[0]:
        display_property_card(key, val)
for i, (key, val) in enumerate(items[mid:]):
    with prop_cols[1]:
        display_property_card(key, val)

# Download
result_data = pd.DataFrame({
    "Parameter": ["d₃₃", "d₃₁", "d₃₂", "d₁₅", "d₂₄", "Filler", "Fraction (%)", "Method", "β input", "β effective"],
    "Value": [f"{d33:.4f}", f"{d31:.4f}", f"{d32:.4f}", f"{d15:.4f}", f"{d24:.4f}",
              selected_filler, dopant_fraction, fabrication_method, beta_fraction, effective_beta]
})
csv = result_data.to_csv(index=False)
st.download_button("📥 Download Results (CSV)", csv, f"piezo_{selected_filler}_{dopant_fraction}%.csv", "text/csv")

show_footer()
