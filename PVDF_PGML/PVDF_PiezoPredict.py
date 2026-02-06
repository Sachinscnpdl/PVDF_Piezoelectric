import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="PVDF Composite Piezoelectric Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to required files
materials_properties_path = os.path.join(current_dir, 'materials_properties.py')
checkpoint_path = os.path.join(current_dir, 'best_phys_resid_monotonic_improved_v2.pt')
predictor_path = os.path.join(current_dir, 'piezoelectric_tensor_predictor.py')

# Define custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1a2980, #26d0ce, #1a2980);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    .sub-header {
        font-size: 1.7rem;
        font-weight: bold;
        color: #1a2980;
        margin-bottom: 1rem;
        border-bottom: 2px solid #26d0ce;
        padding-bottom: 0.3rem;
    }
    .highlight {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1a2980;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .data-table {
        background-color: white;
        border-radius: 0.8rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 0.8rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    .tensor-visualization {
        background-color: white;
        border-radius: 0.8rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(120deg, #1a2980, #26d0ce);
        color: white;
        font-weight: bold;
        border-radius: 0.8rem;
        padding: 0.7rem 1.2rem;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(120deg, #26d0ce, #1a2980);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .tensor-cell {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 0.8rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .tensor-cell:hover {
        transform: scale(1.05);
    }
    .tensor-header {
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
        background-color: #e4edf5;
        border-radius: 0.5rem;
    }
    .property-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 0.8rem;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .property-card:hover {
        transform: translateY(-3px);
    }
    .property-name {
        font-weight: bold;
        color: #1a2980;
        font-size: 1.1rem;
    }
    .property-value {
        color: #26d0ce;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .file-status {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        border-radius: 0.8rem;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .welcome-title {
        font-size: 2rem;
        font-weight: bold;
        color: #1a2980;
        margin-bottom: 1rem;
    }
    .welcome-text {
        font-size: 1.1rem;
        color: #333;
        line-height: 1.6;
    }
    .step {
        margin-bottom: 0.8rem;
        padding-left: 1.5rem;
        position: relative;
    }
    .step:before {
        content: "→";
        position: absolute;
        left: 0;
        color: #26d0ce;
        font-weight: bold;
    }
    .tensor-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .tensor-row {
        display: flex;
        gap: 0.5rem;
        justify-content: space-between;
    }
    .tensor-label {
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #1a2980;
        background-color: #e4edf5;
        border-radius: 0.5rem;
        padding: 0.5rem;
        min-width: 40px;
    }
    .tensor-value {
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem;
        min-width: 80px;
        transition: all 0.3s ease;
    }
    .tensor-value:hover {
        transform: scale(1.05);
    }
    .tensor-zero {
        background-color: #f0f0f0;
        color: #888;
    }
    .tensor-nonzero {
        background-color: #e4edf5;
        color: #1a2980;
    }
    .tensor-negative {
        background-color: #ffebee;
        color: #c62828;
    }
    .footnote {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">PVDF Composite Piezoelectric Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem; color: #555;">
Advanced prediction of piezoelectric coefficients for PVDF-based composites using physics-informed machine learning
</p>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Input Parameters</h2>', unsafe_allow_html=True)
    
    # Check if all required files exist
    missing_files = []
    
    # Load properties for the UI
    try:
        # Import materials_properties module
        import importlib.util
        spec = importlib.util.spec_from_file_location("materials_properties", materials_properties_path)
        materials_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materials_module)
        properties_data = materials_module.properties
    except Exception as e:
        st.error(f"Error loading materials_properties.py: {str(e)}")
        missing_files.append("materials_properties.py")
    
    # Import the predictor functions
    try:
        sys.path.insert(0, current_dir)
        from piezoelectric_tensor_predictor import predict_sample, predict_dataframe
    except Exception as e:
        st.error(f"Error importing predictor: {str(e)}")
        missing_files.append("predictor module")
    
    # Available fillers from properties
    if 'properties_data' in locals():
        fillers = list(properties_data.keys())
        fillers.remove('PVDF')  # Remove PVDF from the list of fillers
        
        # Input fields with default values as requested
        selected_filler = st.selectbox("Select Filler", fillers, index=fillers.index('SnO2') if 'SnO2' in fillers else 0)
        dopant_fraction = st.slider("Dopant Fraction (%)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        fabrication_method = st.selectbox("Fabrication Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
        beta_fraction = st.number_input("Beta Fraction (optional override)", min_value=0.0, max_value=1.0, value=0.5725, step=0.01, 
                                       help="Leave as 0.0 to use calculated value based on dopant")
        
        # Display filler properties
        st.markdown('<h3 class="sub-header">Filler Properties</h3>', unsafe_allow_html=True)
        if selected_filler in properties_data:
            filler_props = properties_data[selected_filler]
            prop_data = []
            for key, value in filler_props.items():
                if not key.startswith('_') and not key.startswith('comment'):
                    prop_data.append({"Property": key, "Value": value})
            if prop_data:
                st.dataframe(pd.DataFrame(prop_data), use_container_width=True, hide_index=True)
        
        # Predict button
        predict_button = st.button("Predict Piezoelectric Properties", type="primary", use_container_width=True)
    
    # File status at the bottom of the input panel
    st.markdown('<div class="file-status">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">File Status</h3>', unsafe_allow_html=True)
    
    required_files = {
        "materials_properties.py": materials_properties_path,
        "checkpoint file": checkpoint_path,
        "predictor module": predictor_path
    }
    
    for file_name, file_path in required_files.items():
        if os.path.exists(file_path):
            st.success(f"✓ {file_name}")
        else:
            st.error(f"✗ {file_name} not found")
            missing_files.append(file_name)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.stop()

# Main content area
if predict_button:
    # Show loading spinner
    with st.spinner("Calculating piezoelectric properties..."):
        try:
            # Make prediction using the real predictor
            df = predict_sample(
                checkpoint_path=checkpoint_path,
                dopant=selected_filler,
                frac=dopant_fraction,
                method=fabrication_method,
                beta_fraction=beta_fraction if beta_fraction > 0 else None,
                device='cpu'
            )
            
            # Ensure df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame([df])
            
            # Extract results
            predicted_d33 = float(df['predicted_d33'].iloc[0])
            phys_d31 = float(df['phys_d31'].iloc[0])
            phys_d32 = float(df['phys_d32'].iloc[0])
            phys_d15 = float(df['phys_d15'].iloc[0])
            phys_d24 = float(df['phys_d24'].iloc[0])
            
            # Extract additional features from the prediction
            features = {
                "Dopant": selected_filler,
                "Dopant Fraction (%)": f"{dopant_fraction:.1f}",
                "Fabrication Method": fabrication_method,
                "PVDF Beta Fraction": f"{df['PVDF_Beta_Fraction_used'].iloc[0]:.4f}",
                "Effective Dielectric Constant": f"{df['Effective Dielectric Constant'].iloc[0]:.4f}",
                "Effective Young's Modulus": f"{df['Effective Youngs Modulus'].iloc[0]:.4f} GPa",
                "Effective Poisson's Ratio": f"{df['Effective Poissons Ratio'].iloc[0]:.4f}",
                "Physics Base d33": f"{df['physics_base_d33'].iloc[0]:.4f} pC/N",
                "Learned Delta d33": f"{df['learned_delta_d33'].iloc[0]:.4f} pC/N",
                "Filler Category": properties_data.get(selected_filler, {}).get('Filler_Category', 'Not specified')
            }
            
            # Create tensor matrix
            tensor_matrix = np.array([
                [0, 0, 0, 0, phys_d15, 0],
                [0, 0, 0, phys_d24, 0, 0],
                [phys_d31, phys_d32, -predicted_d33, 0, 0, 0]
            ])
            
            st.success("✅ Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Display results
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    # Create side-by-side layout for results and tensor
    col_results, col_tensor = st.columns([1, 1])
    
    with col_results:
        # Key metrics in cards
        st.markdown('<h3 style="color: #1a2980; margin-bottom: 1rem;">Piezoelectric Coefficients</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1a2980;">d33</h3>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-weight: bold; color: #26d0ce;">{predicted_d33:.2f} pC/N</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1a2980;">d31</h3>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-weight: bold; color: #26d0ce;">{phys_d31:.2f} pC/N</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1a2980;">d32</h3>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-weight: bold; color: #26d0ce;">{phys_d32:.2f} pC/N</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1a2980;">d15</h3>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-weight: bold; color: #26d0ce;">{phys_d15:.2f} pC/N</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1a2980;">d24</h3>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-weight: bold; color: #26d0ce;">{phys_d24:.2f} pC/N</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add some additional info in the second column
            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1a2980;">Filler</h3>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-weight: bold; color: #26d0ce;">{selected_filler}</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_tensor:
        # Improved tensor visualization
        st.markdown('<h3 style="color: #1a2980; margin-bottom: 1rem;">Piezoelectric Tensor Matrix</h3>', unsafe_allow_html=True)
        st.markdown('<div class="tensor-visualization">', unsafe_allow_html=True)
        
        # Create a more visually appealing tensor representation
        st.markdown('<div class="tensor-container">', unsafe_allow_html=True)
        
        # Row labels
        st.markdown('<div class="tensor-row">', unsafe_allow_html=True)
        st.markdown('<div class="tensor-label"></div>', unsafe_allow_html=True)
        for j in range(6):
            st.markdown(f'<div class="tensor-label">d{j+1}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tensor values
        for i in range(3):
            st.markdown('<div class="tensor-row">', unsafe_allow_html=True)
            st.markdown(f'<div class="tensor-label">d{i+1}</div>', unsafe_allow_html=True)
            for j in range(6):
                value = tensor_matrix[i, j]
                if abs(value) < 1e-10:
                    css_class = "tensor-zero"
                elif value < 0:
                    css_class = "tensor-negative"
                else:
                    css_class = "tensor-nonzero"
                
                st.markdown(f'<div class="tensor-value {css_class}">{value:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Material properties
    st.markdown('<h2 class="sub-header">Material Properties</h2>', unsafe_allow_html=True)
    st.markdown('<div class="data-table">', unsafe_allow_html=True)
    
    # Create two columns for features
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        for i, (key, value) in enumerate(list(features.items())[:len(features)//2]):
            st.markdown(f"""
            <div class="property-card">
                <div class="property-name">{key}</div>
                <div class="property-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with feature_col2:
        for i, (key, value) in enumerate(list(features.items())[len(features)//2:]):
            st.markdown(f"""
            <div class="property-card">
                <div class="property-name">{key}</div>
                <div class="property-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download results button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create a DataFrame with all results
        results_df = pd.DataFrame({
            'Parameter': ['Dopant', 'Dopant Fraction (%)', 'Fabrication Method', 'PVDF Beta Fraction', 
                          'd33 (pC/N)', 'd31 (pC/N)', 'd32 (pC/N)', 'd15 (pC/N)', 'd24 (pC/N)'],
            'Value': [selected_filler, dopant_fraction, fabrication_method, beta_fraction,
                      f"{predicted_d33:.4f}", f"{phys_d31:.4f}", f"{phys_d32:.4f}", 
                      f"{phys_d15:.4f}", f"{phys_d24:.4f}"]
        })
        
        # Convert to CSV
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"piezoelectric_prediction_{selected_filler}_{dopant_fraction}%.csv",
            mime="text/csv"
        )
    
    # Add footnote with paper reference
    st.markdown("""
    <div class="footnote">
        <p><strong>Reference:</strong> This work is based on the following paper (yet to be published):</p>
        <p>"Phase Characterization, Enhanced Piezoelectric Performance, and Device Potential of Electrospun PVDF/SnO2 Nanofibers via Physics-Guided Machine Learning"</p>
        <p>Sachin Poudela,∗, Weronika Smoka, Rubi Thapab, Anna Timofiejczuka, Nele Moelansc and Anil Kunwar</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Welcome screen when no prediction has been made
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="welcome-title">Welcome to the PVDF Composite Piezoelectric Predictor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-text">
        <p>This advanced tool predicts the piezoelectric coefficients of PVDF-based composites using state-of-the-art physics-informed machine learning.</p>
        <p style="margin-top: 1.5rem;"><strong>To get started:</strong></p>
        <div class="step">Select a filler material from the sidebar (default: SnO2)</div>
        <div class="step">Adjust the dopant fraction (default: 1.5%) and fabrication method</div>
        <div class="step">Optionally override the beta fraction (default: 0.5725)</div>
        <div class="step">Click the <strong>"Predict Piezoelectric Properties"</strong> button</div>
        <div class="step">Explore the results including the full piezoelectric tensor</div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'properties_data' in locals():
        st.markdown(f"""
        <div class="welcome-text" style="margin-top: 1.5rem;">
            <p><strong>Available fillers:</strong> {", ".join(fillers)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add footnote with paper reference
    st.markdown("""
    <div class="footnote">
        <p><strong>Reference:</strong> This work is based on the following paper (yet to be published):</p>
        <p>"Phase Characterization, Enhanced Piezoelectric Performance, and Device Potential of Electrospun PVDF/SnO2 Nanofibers via Physics-Guided Machine Learning"</p>
        <p>Sachin Poudel∗, Weronika Smok, Rubi Thapa, Anna Timofiejczuk, Nele Moelans and Anil Kunwar</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>PVDF Composite Piezoelectric Predictor | Powered by Physics-Informed Machine Learning</p>
</div>
""", unsafe_allow_html=True)
