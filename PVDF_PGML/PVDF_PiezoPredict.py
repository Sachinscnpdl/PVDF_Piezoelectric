import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
import plotly.express as px
import plotly.graph_objects as go

# ---------- Page config ----------
st.set_page_config(
    page_title="PVDF Piezoelectric Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Paths ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
materials_properties_path = os.path.join(current_dir, 'materials_properties.py')
checkpoint_path = os.path.join(current_dir, 'best_phys_resid_monotonic_improved_v2.pt')
predictor_path = os.path.join(current_dir, 'piezoelectric_tensor_predictor.py')

# ---------- Minimal CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        text-align: center;
        background: linear-gradient(120deg, #1a2980, #26d0ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 0.8rem;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .small-metric {
        font-size: 0.85rem;
        color: #555;
        text-align: center;
        background: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">PVDF Composite Piezoelectric Predictor</h1>', unsafe_allow_html=True)

# ---------- Load materials data ----------
missing_files = []
properties_data = None
try:
    spec = importlib.util.spec_from_file_location("materials_properties", materials_properties_path)
    materials_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materials_module)
    properties_data = materials_module.properties
except Exception:
    missing_files.append("materials_properties.py")

try:
    sys.path.insert(0, current_dir)
    from piezoelectric_tensor_predictor import predict_sample
except Exception:
    missing_files.append("predictor module")

if missing_files:
    st.error(f"Missing files: {', '.join(missing_files)}")
    st.stop()

# ---------- Sidebar (clean) ----------
with st.sidebar:
    st.markdown("### ⚙️ Input Parameters")
    
    fillers = [f for f in properties_data.keys() if f != 'PVDF']
    selected_filler = st.selectbox("Filler", fillers, index=fillers.index('SnO2') if 'SnO2' in fillers else 0)
    dopant_fraction = st.slider("Dopant fraction (%)", 0.02, 10.0, 1.5, 0.1)
    fabrication_method = st.selectbox("Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
    beta_fraction = st.number_input("β fraction override", 0.0, 1.0, 0.5725, 0.01,
                                    help="0.0 = auto-calculated")
    
    damping_factor = 0.3
    effective_beta = 0.6 + (beta_fraction - 0.6) * damping_factor if beta_fraction > 0.6 else beta_fraction
    
    predict_button = st.button("⚡ Predict", type="primary", use_container_width=True)
    
    # Optional expanders for extra info (collapsed by default)
    with st.expander("📄 Filler properties"):
        filler_props = {k: v for k, v in properties_data[selected_filler].items() 
                       if not k.startswith('_') and not k.startswith('comment')}
        if filler_props:
            st.dataframe(pd.DataFrame(list(filler_props.items()), columns=["Property", "Value"]),
                        use_container_width=True, hide_index=True, height=200)
    
    with st.expander("🔧 System status"):
        for name, path in [("materials_properties.py", materials_properties_path), 
                           ("Model checkpoint", checkpoint_path), 
                           ("Predictor module", predictor_path)]:
            if os.path.exists(path):
                st.success(f"✓ {name}")
            else:
                st.error(f"✗ {name}")

# ---------- Prediction & Results ----------
if predict_button and properties_data:
    with st.spinner("Predicting..."):
        try:
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
            
            coeffs = {
                'd₃₃': float(df['predicted_d33'].iloc[0]),
                'd₃₁': float(df['phys_d31'].iloc[0]),
                'd₃₂': float(df['phys_d32'].iloc[0]),
                'd₁₅': float(df['phys_d15'].iloc[0]),
                'd₂₄': float(df['phys_d24'].iloc[0])
            }
            
            # Tensor in Voigt notation (3x6)
            tensor = np.array([
                [0, 0, 0, 0, coeffs['d₁₅'], 0],
                [0, 0, 0, coeffs['d₂₄'], 0, 0],
                [coeffs['d₃₁'], coeffs['d₃₂'], -coeffs['d₃₃'], 0, 0, 0]
            ])
            
            # Additional metrics
            extra = {
                "β fraction": effective_beta,
                "ε_r": df['Effective Dielectric Constant'].iloc[0],
                "E (GPa)": df['Effective Youngs Modulus'].iloc[0],
                "ν": df['Effective Poissons Ratio'].iloc[0],
                "Physics d₃₃": df['physics_base_d33'].iloc[0],
                "Δ d₃₃": df['learned_delta_d33'].iloc[0]
            }
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    
    # ---------- Display results ----------
    # Row 1: Metric cards for coefficients
    st.markdown("### Piezoelectric coefficients (pC/N)")
    cols = st.columns(len(coeffs))
    for col, (name, val) in zip(cols, coeffs.items()):
        col.metric(name, f"{val:.2f}")
    
    # Row 2: Bar chart + Tensor heatmap side by side
    col_chart, col_heat = st.columns([1, 1.2])
    
    with col_chart:
        # Bar chart of coefficients
        df_coeff = pd.DataFrame(list(coeffs.items()), columns=['Coefficient', 'Value'])
        fig_bar = px.bar(df_coeff, x='Coefficient', y='Value', color='Value',
                         color_continuous_scale='RdBu_r', text='Value',
                         title='Coefficient comparison')
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_heat:
        # Heatmap of piezoelectric tensor
        fig_heat = go.Figure(data=go.Heatmap(
            z=tensor,
            x=['σ₁', 'σ₂', 'σ₃', 'σ₄', 'σ₅', 'σ₆'],
            y=['Polarization X', 'Polarization Y', 'Polarization Z'],
            colorscale='Viridis',
            text=np.round(tensor, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar_title="pC/N"
        ))
        fig_heat.update_layout(title='Piezoelectric tensor (Voigt notation)', height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
    
    # Row 3: Small metrics row (compact)
    st.markdown("#### Additional properties")
    subcols = st.columns(len(extra))
    for col, (key, val) in zip(subcols, extra.items()):
        if key in ["β fraction", "ε_r", "ν", "Physics d₃₃", "Δ d₃₃"]:
            col.markdown(f'<div class="small-metric">{key}<br><b>{val:.3f}</b></div>', unsafe_allow_html=True)
        else:
            col.markdown(f'<div class="small-metric">{key}<br><b>{val:.2f}</b></div>', unsafe_allow_html=True)
    
    # Download button (discreet)
    results_df = pd.DataFrame({
        'Parameter': ['Filler', 'Fraction (%)', 'Method', 'β fraction'] + list(coeffs.keys()),
        'Value': [selected_filler, dopant_fraction, fabrication_method, effective_beta] + 
                 [f"{v:.4f}" for v in coeffs.values()]
    })
    st.download_button("📥 Export CSV", results_df.to_csv(index=False),
                      f"pvdf_{selected_filler}_{dopant_fraction}%.csv", "text/csv", use_container_width=False)

else:
    # Minimal empty state
    st.info("👈 Select filler, fraction, and method in the sidebar, then click **Predict**.")

# ---------- Footer (short) ----------
st.markdown("""
<div style="text-align:center;color:#aaa;font-size:0.7rem;margin-top:2rem;">
    Physics-informed ML · PVDF composites
</div>
""", unsafe_allow_html=True)
