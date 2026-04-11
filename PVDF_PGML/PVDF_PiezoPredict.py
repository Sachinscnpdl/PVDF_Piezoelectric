import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib.util

st.set_page_config(
    page_title="PGMLpiezo PVDF",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
materials_properties_path = os.path.join(current_dir, 'materials_properties.py')
checkpoint_path = os.path.join(current_dir, 'best_phys_resid_monotonic_improved_v2.pt')
predictor_path = os.path.join(current_dir, 'piezoelectric_tensor_predictor.py')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 20% 30%, #f8faff, #eef2ff);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #0f2b7c 0%, #1e88e5 40%, #00acc1 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .subhead {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 400;
        color: #1e3a8a;
        opacity: 0.8;
        margin-bottom: 1.5rem;
        letter-spacing: 0.3px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #0f2b7c, #00838f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-left: 4px solid #00acc1;
        padding-left: 0.75rem;
        margin-bottom: 1.2rem;
        margin-top: 0.2rem;
    }
    
    .card-glow {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(2px);
        border-radius: 1.5rem;
        padding: 1rem 1rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(255,255,255,0.5) inset;
        transition: all 0.25s ease;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.6);
    }
    
    .card-glow:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 40px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.95);
    }
    
    .coeff-label {
        font-size: 1rem;
        font-weight: 600;
        color: #0f2b7c;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    
    .coeff-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(145deg, #0f2b7c, #0097a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .unit {
        font-size: 0.8rem;
        font-weight: 500;
        color: #4b6e8a;
        margin-left: 0.2rem;
    }
    
    .sidebar-panel {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        border-radius: 1.5rem;
        padding: 1.2rem 1rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    .stButton > button {
        background: linear-gradient(95deg, #0f2b7c, #1e88e5);
        color: white;
        font-weight: 700;
        border-radius: 2rem;
        border: none;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.15);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(95deg, #1e88e5, #00acc1);
        transform: scale(1.01);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.2);
    }
    
    .property-card {
        background: #ffffffcc;
        border-radius: 1.2rem;
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #00acc1;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    
    .prop-label {
        font-weight: 600;
        color: #1e3a8a;
    }
    
    .prop-value {
        font-weight: 700;
        color: #00838f;
        float: right;
    }
    
    .tensor-cell {
        text-align: center;
        font-weight: 700;
        border-radius: 0.8rem;
    }
    
    .footer-note {
        text-align: center;
        font-size: 0.7rem;
        color: #5a6e8a;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    hr {
        margin: 0.5rem 0;
        background: linear-gradient(90deg, transparent, #00acc1, transparent);
        height: 1px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">PGMLpiezo  ⚡ PVDF</div>', unsafe_allow_html=True)
st.markdown('<div class="subhead">Physics‑Guided Machine Learning · composite intelligence</div>', unsafe_allow_html=True)

missing_files = []
properties_data = None

try:
    spec = importlib.util.spec_from_file_location("materials_properties", materials_properties_path)
    materials_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materials_module)
    properties_data = materials_module.properties
except Exception as e:
    missing_files.append("materials_properties.py")

try:
    sys.path.insert(0, current_dir)
    from piezoelectric_tensor_predictor import predict_sample
except Exception:
    missing_files.append("predictor module")

with st.sidebar:
    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-weight:700; color:#0f2b7c; margin-bottom:1rem;">⚙️ design matrix</h2>', unsafe_allow_html=True)
    
    if properties_data:
        fillers = [f for f in properties_data.keys() if f != 'PVDF']
        selected_filler = st.selectbox("Filler Material", fillers, index=fillers.index('SnO2') if 'SnO2' in fillers else 0)
        dopant_fraction = st.slider("Dopant Fraction (%)", 0.02, 10.0, 1.5, 0.1)
        fabrication_method = st.selectbox("Fabrication Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
        beta_fraction = st.number_input("Beta Fraction Override", 0.0, 1.0, 0.5725, 0.01, help="0.0 = calculated. >0.6 applies damping.")
        
        damping_factor = 0.3
        effective_beta = 0.6 + (beta_fraction - 0.6) * damping_factor if beta_fraction > 0.6 else beta_fraction
        
        st.markdown('<div style="margin: 1rem 0 0.5rem 0;"><span style="font-weight:600; color:#0f2b7c;">🔬 filler fingerprint</span></div>', unsafe_allow_html=True)
        filler_props = {k: v for k, v in properties_data[selected_filler].items() 
                       if not k.startswith('_') and not k.startswith('comment')}
        if filler_props:
            for prop, val in list(filler_props.items())[:5]:
                st.markdown(f'<div style="display:flex; justify-content:space-between; font-size:0.85rem; padding:0.2rem 0;"><span style="color:#1e3a8a;">{prop}</span><span style="font-weight:600; color:#00838f;">{val}</span></div>', unsafe_allow_html=True)
        
        predict_button = st.button("⚡ PREDICT", type="primary", use_container_width=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#5e6f8d;">system status</div>', unsafe_allow_html=True)
    for name, path in [("materials_properties.py", materials_properties_path), 
                       ("checkpoint", checkpoint_path), 
                       ("predictor", predictor_path)]:
        if os.path.exists(path):
            st.markdown(f'<div style="font-size:0.7rem; color:#2e7d64;">✓ {name}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:0.7rem; color:#c62828;">✗ {name}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if missing_files:
    st.error(f"Missing: {', '.join(missing_files)}")
    st.stop()

if properties_data and predict_button:
    with st.spinner("calculating tensor response ..."):
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
                'd33': float(df['predicted_d33'].iloc[0]),
                'd31': float(df['phys_d31'].iloc[0]),
                'd32': float(df['phys_d32'].iloc[0]),
                'd15': float(df['phys_d15'].iloc[0]),
                'd24': float(df['phys_d24'].iloc[0])
            }
            tensor = np.array([
                [0, 0, 0, 0, coeffs['d15'], 0],
                [0, 0, 0, coeffs['d24'], 0, 0],
                [coeffs['d31'], coeffs['d32'], -coeffs['d33'], 0, 0, 0]
            ])
            features = {
                "Filler Category": properties_data[selected_filler].get('Filler_Category', '-'),
                "Effective β": f"{effective_beta:.4f}",
                "Dielectric ε": f"{df['Effective Dielectric Constant'].iloc[0]:.2f}",
                "Young's Modulus": f"{df['Effective Youngs Modulus'].iloc[0]:.1f} GPa",
                "Poisson's Ratio": f"{df['Effective Poissons Ratio'].iloc[0]:.3f}",
                "Physics d33 base": f"{df['physics_base_d33'].iloc[0]:.2f} pC/N",
                "Learned Δ d33": f"{df['learned_delta_d33'].iloc[0]:.2f} pC/N"
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    col_res, col_ten = st.columns([1, 1], gap="large")
    
    with col_res:
        st.markdown('<div class="section-header">piezoelectric coefficients</div>', unsafe_allow_html=True)
        cols_cards = st.columns(2)
        coeff_items = list(coeffs.items())
        for idx, (name, val) in enumerate(coeff_items):
            with cols_cards[idx % 2]:
                st.markdown(f'''
                <div class="card-glow" style="margin-bottom:0.9rem;">
                    <div class="coeff-label">{name}</div>
                    <div class="coeff-value">{val:.2f}<span class="unit">pC/N</span></div>
                </div>
                ''', unsafe_allow_html=True)
    
    with col_ten:
        st.markdown('<div class="section-header">piezoelectric tensor [d]</div>', unsafe_allow_html=True)
        
        def style_tensor(val):
            if abs(val) < 1e-10:
                return 'background: #e0f2f1; color: #00695c; font-weight: 700; text-align: center; border-radius: 12px;'
            elif val < 0:
                return 'background: #ffebee; color: #c62828; font-weight: 700; text-align: center; border-radius: 12px;'
            else:
                return 'background: #fff8e1; color: #ef6c00; font-weight: 700; text-align: center; border-radius: 12px;'
        
        tensor_df = pd.DataFrame(tensor, index=["d₁", "d₂", "d₃"], columns=["1", "2", "3", "4", "5", "6"])
        styled = tensor_df.style.map(style_tensor).set_properties(**{'padding': '8px', 'font-size': '0.9rem'})
        st.dataframe(styled, use_container_width=True, height=160)
    
    st.markdown('<div class="section-header" style="margin-top:0.8rem;">derived properties</div>', unsafe_allow_html=True)
    prop_cols = st.columns(2)
    prop_items = list(features.items())
    for i, (k, v) in enumerate(prop_items):
        with prop_cols[i % 2]:
            st.markdown(f'''
            <div class="property-card">
                <span class="prop-label">{k}</span>
                <span class="prop-value">{v}</span>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    results_df = pd.DataFrame({
        'Parameter': ['Filler', 'Fraction (%)', 'Method', 'β Fraction'] + list(coeffs.keys()),
        'Value': [selected_filler, dopant_fraction, fabrication_method, effective_beta] + 
                 [f"{v:.4f}" for v in coeffs.values()]
    })
    st.download_button("📎 EXPORT CSV", results_df.to_csv(index=False),
                      f"pvdf_{selected_filler}_{dopant_fraction}%.csv", "text/csv")

else:
    st.markdown('''
    <div class="card-glow" style="text-align:left; padding:1.5rem; margin:1rem 0;">
        <div style="font-size:1.8rem; font-weight:800; background:linear-gradient(135deg,#0f2b7c,#00acc1); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">⚡ ready</div>
        <p style="margin-top:0.8rem; color:#1f3a6b;">configure filler, fraction & method → hit <strong>PREDICT</strong></p>
        <hr style="margin:1rem 0;">
        <div style="font-size:0.85rem; color:#2c5282;">physics‑informed model | piezoelectric tensor output</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if properties_data:
        fillers = [f for f in properties_data.keys() if f != 'PVDF']
        st.markdown(f'<div style="background:#eef2ff; border-radius:1rem; padding:0.5rem 1rem; font-size:0.8rem; color:#1e3a8a;">📌 available fillers: {", ".join(fillers)}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">PVDF composite predictor · physics‑guided learning · Poudel et al.</div>', unsafe_allow_html=True)
