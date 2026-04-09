import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib.util

st.set_page_config(
    page_title="PVDF Piezoelectric Predictor",
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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1a2980, #26d0ce, #1a2980);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1a2980;
        border-bottom: 2px solid #26d0ce;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 0.8rem;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .card:hover { transform: translateY(-3px); }
    .card-label { color: #1a2980; font-size: 1rem; margin-bottom: 0.3rem; }
    .card-value { color: #26d0ce; font-size: 1.3rem; font-weight: bold; }
    .sidebar-panel {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
    }
    .stButton>button {
        background: linear-gradient(120deg, #1a2980, #26d0ce);
        color: white;
        font-weight: bold;
        border-radius: 0.8rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(120deg, #26d0ce, #1a2980);
        transform: translateY(-2px);
    }
    .info-box {
        background: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin-top: 0.8rem;
        font-size: 0.85rem;
        color: #666;
    }
    .legend-dot {
        width: 16px;
        height: 16px;
        display: inline-block;
        margin-right: 4px;
        vertical-align: middle;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">PVDF Composite Piezoelectric Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#555;margin-bottom:1.5rem;">Physics-informed ML prediction of piezoelectric coefficients</p>', unsafe_allow_html=True)

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
    st.markdown('<h2 class="section-header">Input Parameters</h2>', unsafe_allow_html=True)
    
    if properties_data:
        fillers = [f for f in properties_data.keys() if f != 'PVDF']
        selected_filler = st.selectbox("Filler Material", fillers, index=fillers.index('SnO2') if 'SnO2' in fillers else 0)
        dopant_fraction = st.slider("Dopant Fraction (%)", 0.02, 10.0, 1.5, 0.1)
        fabrication_method = st.selectbox("Fabrication Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
        beta_fraction = st.number_input("Beta Fraction Override", 0.0, 1.0, 0.5725, 0.01, help="0.0 = calculated. >0.6 applies damping.")
        
        damping_factor = 0.3
        effective_beta = 0.6 + (beta_fraction - 0.6) * damping_factor if beta_fraction > 0.6 else beta_fraction
        
        st.markdown('<h3 class="section-header">Filler Properties</h3>', unsafe_allow_html=True)
        filler_props = {k: v for k, v in properties_data[selected_filler].items() 
                       if not k.startswith('_') and not k.startswith('comment')}
        if filler_props:
            st.dataframe(pd.DataFrame(list(filler_props.items()), columns=["Property", "Value"]), 
                        use_container_width=True, hide_index=True, height=200)
        
        predict_button = st.button("⚡ Predict", type="primary", use_container_width=True)
    
    st.markdown('<div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e0e0e0;">', unsafe_allow_html=True)
    for name, path in [("materials_properties.py", materials_properties_path), 
                       ("Model checkpoint", checkpoint_path), 
                       ("Predictor module", predictor_path)]:
        (st.success if os.path.exists(path) else st.error)(f"{'✓' if os.path.exists(path) else '✗'} {name}")
    st.markdown('</div></div>', unsafe_allow_html=True)

if missing_files:
    st.error(f"Missing: {', '.join(missing_files)}")
    st.stop()

if properties_data and predict_button:
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
                "Effective β Fraction": f"{effective_beta:.4f}",
                "Dielectric Constant": f"{df['Effective Dielectric Constant'].iloc[0]:.4f}",
                "Young's Modulus": f"{df['Effective Youngs Modulus'].iloc[0]:.4f} GPa",
                "Poisson's Ratio": f"{df['Effective Poissons Ratio'].iloc[0]:.4f}",
                "Physics Base d33": f"{df['physics_base_d33'].iloc[0]:.4f} pC/N",
                "Learned Δ d33": f"{df['learned_delta_d33'].iloc[0]:.4f} pC/N"
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    st.success("✅ Prediction complete")
    
    col_res, col_ten = st.columns([1, 1])
    
    with col_res:
        st.markdown('<h2 class="section-header">Piezoelectric Coefficients</h2>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, (name, val) in enumerate(coeffs.items()):
            with c1 if i < 3 else c2:
                st.markdown(f'''<div class="card" style="margin-bottom:0.8rem;">
                    <div class="card-label">{name}</div>
                    <div class="card-value">{val:.2f} pC/N</div>
                </div>''', unsafe_allow_html=True)
    
    with col_ten:
        st.markdown('<h2 class="section-header">Piezoelectric Tensor</h2>', unsafe_allow_html=True)
        
        def style_tensor(val):
            if abs(val) < 1e-10:
                return 'background-color: #e8f4f8; color: #1976d2; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); text-align: center;'
            elif val < 0:
                return 'background-color: #ffebee; color: #d32f2f; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); text-align: center;'
            else:
                return 'background-color: #fff3e0; color: #f57c00; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); text-align: center;'
        
        tensor_df = pd.DataFrame(tensor, index=["d₁", "d₂", "d₃"], columns=["1", "2", "3", "4", "5", "6"])
        styled = tensor_df.style.map(style_tensor)
        st.dataframe(styled, use_container_width=True)
        
        st.markdown('''<div style="display:flex;justify-content:center;gap:12px;margin-top:8px;">
            <div style="display:flex;align-items:center;"><span class="legend-dot" style="background:#e8f4f8;border:1px solid #90caf9;"></span><span style="font-size:0.8rem;color:#666;">Zero</span></div>
            <div style="display:flex;align-items:center;"><span class="legend-dot" style="background:#fff3e0;border:1px solid #ffb74d;"></span><span style="font-size:0.8rem;color:#666;">Positive</span></div>
            <div style="display:flex;align-items:center;"><span class="legend-dot" style="background:#ffebee;border:1px solid #ef9a9a;"></span><span style="font-size:0.8rem;color:#666;">Negative</span></div>
        </div>''', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Rows: polarization (X,Y,Z) • Columns: stress directions (Voigt notation)</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Computed Properties</h2>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    items = list(features.items())
    for i, (k, v) in enumerate(items):
        with fc1 if i < len(items)//2 else fc2:
            st.markdown(f'''<div class="card" style="margin-bottom:0.6rem;text-align:left;padding:0.8rem 1rem;">
                <span style="color:#1a2980;font-weight:600;">{k}:</span> 
                <span style="color:#26d0ce;font-weight:bold;">{v}</span>
            </div>''', unsafe_allow_html=True)
    
    st.markdown("---")
    results_df = pd.DataFrame({
        'Parameter': ['Filler', 'Fraction (%)', 'Method', 'β Fraction'] + list(coeffs.keys()),
        'Value': [selected_filler, dopant_fraction, fabrication_method, effective_beta] + 
                 [f"{v:.4f}" for v in coeffs.values()]
    })
    st.download_button("📥 Download CSV", results_df.to_csv(index=False),
                      f"pvdf_{selected_filler}_{dopant_fraction}%.csv", "text/csv")

else:
    st.markdown('''<div class="card" style="text-align:left;padding:2rem;">
        <h2 style="color:#1a2980;margin-bottom:1rem;">Welcome</h2>
        <p style="line-height:1.8;color:#444;">
            Predict piezoelectric coefficients for PVDF composites using physics-informed ML.<br><br>
            <strong>Steps:</strong><br>
            → Select filler material (sidebar)<br>
            → Adjust dopant fraction & fabrication method<br>
            → Optionally override β fraction<br>
            → Click <strong>"Predict"</strong> to view results
        </p>
    </div>''', unsafe_allow_html=True)
    
    if properties_data:
        fillers = [f for f in properties_data.keys() if f != 'PVDF']
        st.markdown(f'<div class="info-box">Available fillers: {", ".join(fillers)}</div>', unsafe_allow_html=True)

st.markdown("""---
<div style="text-align:center;color:#888;font-size:0.85rem;padding:1rem;">
    PVDF Composite Piezoelectric Predictor | Physics-Informed ML<br>
    <em>Poudel et al. - "Phase Characterization, Enhanced Piezoelectric Performance, and Device Potential of Electrospun PVDF/SnO2 Nanofibers via Physics-Guided Machine Learning" (Unpublished)</em>
</div>""", unsafe_allow_html=True)
