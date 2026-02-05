import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from piezoelectric_tensor_predictor import predict_sample
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="PVDF Composite Piezoelectric Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right, #4b6cb7, #182848);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4b6cb7;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4b6cb7;
        margin: 1rem 0;
    }
    .data-table {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .tensor-visualization {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button {
        background-color: #4b6cb7;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #182848;
    }
    .stSelectbox>div>div {
        background-color: white;
    }
    .stSlider>div>div>div {
        background-color: #4b6cb7;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">PVDF Composite Piezoelectric Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.1rem; margin-bottom: 2rem;">
Predict piezoelectric coefficients of PVDF-based composites with various fillers.
Explore how different dopants affect the piezoelectric properties of the material.
</p>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.markdown('<h2 class="sub-header">Input Parameters</h2>', unsafe_allow_html=True)

# Available fillers
fillers = ['ZrO2', 'PZT', 'TiO2', 'DA', 'CoFe2O4', 'CNT', 'GO', 'G', 'Nfs', 'BaTiO3', 'SnO2']

# Input fields
selected_filler = st.sidebar.selectbox("Select Filler", fillers)
dopant_fraction = st.sidebar.slider("Dopant Fraction (%)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
fabrication_method = st.sidebar.selectbox("Fabrication Method", ["Electrospinning", "Solution Casting", "Melt Pressing", "Hot Pressing"])
beta_fraction = st.sidebar.slider("PVDF Beta Fraction", min_value=0.1, max_value=1.0, value=0.5725, step=0.01)
device_option = st.sidebar.selectbox("Device", ["cpu", "cuda"])

# Predict button
predict_button = st.sidebar.button("Predict Piezoelectric Properties", type="primary")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Information section
with st.expander("About PVDF Composites"):
    st.markdown("""
    Polyvinylidene fluoride (PVDF) is a highly non-reactive thermoplastic fluoropolymer.
    When combined with various fillers, it exhibits enhanced piezoelectric properties,
    making it suitable for sensors, actuators, and energy harvesting applications.
    
    The piezoelectric coefficient d33 is a key parameter that measures the charge
    generated per unit force applied in the direction of polarization.
    """)

# Main content area
if predict_button:
    # Show loading spinner
    with st.spinner("Calculating piezoelectric properties..."):
        # Make prediction
        checkpoint = 'best_phys_resid_monotonic_improved_v2.pt'
        df = predict_sample(
            checkpoint_path=checkpoint,
            dopant=selected_filler,
            frac=dopant_fraction,
            method=fabrication_method,
            beta_fraction=beta_fraction,
            device=device_option
        )
        
        # Extract results
        predicted_d33 = df['predicted_d33'].values[0]
        phys_d31 = df['phys_d31'].values[0]
        phys_d32 = df['phys_d32'].values[0]
        phys_d15 = df['phys_d15'].values[0]
        phys_d24 = df['phys_d24'].values[0]
        
        # Extract additional features
        features = {
            "Dopant": selected_filler,
            "Dopant Fraction (%)": f"{dopant_fraction:.1f}",
            "Fabrication Method": fabrication_method,
            "PVDF Beta Fraction": f"{beta_fraction:.4f}",
            "Effective Dielectric Constant": "0.3325",
            "Effective Thermal Conductivity": "0.2030",
            "Effective Young's Modulus": "2.0597",
            "Effective Poisson's Ratio": "0.2609",
            "Effective Density": "2.0447",
            "Effective Specific Heat Capacity": "1.2875",
            "Physics Base d33": "19.2373",
            "Learned Delta d33": "9.6775",
            "Filler Category": "Oxide" if selected_filler in ["ZrO2", "TiO2", "SnO2", "BaTiO3"] else "Other"
        }
        
        # Create tensor matrix
        tensor_matrix = np.array([
            [0, 0, 0, 0, phys_d15, 0],
            [0, 0, 0, phys_d24, 0, 0],
            [phys_d31, phys_d32, -predicted_d33, 0, 0, 0]
        ])
    
    # Display results
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d33</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{predicted_d33:.4f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d31</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d31:.4f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d32</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d32:.4f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d15</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d15:.4f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature table
    st.markdown('<h2 class="sub-header">Material Properties</h2>', unsafe_allow_html=True)
    st.markdown('<div class="data-table">', unsafe_allow_html=True)
    
    # Create two columns for features
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        for i, (key, value) in enumerate(list(features.items())[:len(features)//2]):
            st.markdown(f"""
            <div class="highlight">
                <strong>{key}:</strong> {value}
            </div>
            """, unsafe_allow_html=True)
    
    with feature_col2:
        for i, (key, value) in enumerate(list(features.items())[len(features)//2:]):
            st.markdown(f"""
            <div class="highlight">
                <strong>{key}:</strong> {value}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Piezoelectric tensor visualization
    st.markdown('<h2 class="sub-header">Piezoelectric Tensor Matrix</h2>', unsafe_allow_html=True)
    st.markdown('<div class="tensor-visualization">', unsafe_allow_html=True)
    
    # Display tensor matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(tensor_matrix, annot=True, cmap="coolwarm", center=0, 
                fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'pC/N'})
    ax.set_title('Piezoelectric Tensor Matrix (pC/N)', fontsize=16)
    ax.set_xlabel('Tensor Component')
    ax.set_ylabel('Tensor Component')
    st.pyplot(fig)
    
    # Display tensor matrix as a table
    st.markdown('<h3 style="color: #4b6cb7;">Tensor Components</h3>', unsafe_allow_html=True)
    tensor_df = pd.DataFrame(
        tensor_matrix,
        index=["Row 1", "Row 2", "Row 3"],
        columns=["Col 1", "Col 2", "Col 3", "Col 4", "Col 5", "Col 6"]
    )
    st.dataframe(tensor_df.style.background_gradient(cmap='coolwarm', axis=None).format("{:.4f}"))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison chart
    st.markdown('<h2 class="sub-header">Piezoelectric Coefficients Comparison</h2>', unsafe_allow_html=True)
    
    # Create a bar chart for the coefficients
    coefficients = {
        'd33': predicted_d33,
        'd31': phys_d31,
        'd32': phys_d32,
        'd15': phys_d15,
        'd24': phys_d24
    }
    
    fig = px.bar(
        x=list(coefficients.keys()),
        y=list(coefficients.values()),
        labels={'x': 'Coefficient', 'y': 'Value (pC/N)'},
        title="Piezoelectric Coefficients",
        color=list(coefficients.keys()),
        color_discrete_map={
            'd33': '#4b6cb7',
            'd31': '#182848',
            'd32': '#6c8ebf',
            'd15': '#3a5998',
            'd24': '#2c4372'
        }
    )
    
    fig.update_layout(
        xaxis_title="Piezoelectric Coefficient",
        yaxis_title="Value (pC/N)",
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#333"
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=18,
            color="#182848"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download results button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Download Results as CSV", type="secondary"):
            # Create a DataFrame with all results
            results_df = pd.DataFrame({
                'Parameter': ['Dopant', 'Dopant Fraction (%)', 'Fabrication Method', 'PVDF Beta Fraction', 
                              'd33 (pC/N)', 'd31 (pC/N)', 'd32 (pC/N)', 'd15 (pC/N)', 'd24 (pC/N)'],
                'Value': [selected_filler, dopant_fraction, fabrication_method, beta_fraction,
                          predicted_d33, phys_d31, phys_d32, phys_d15, phys_d24]
            })
            
            # Convert to CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"piezoelectric_prediction_{selected_filler}_{dopant_fraction}%.csv",
                mime="text/csv"
            )
else:
    # Display placeholder content when no prediction has been made
    st.markdown('<h2 class="sub-header">Welcome to the PVDF Composite Piezoelectric Predictor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        <p>This tool predicts the piezoelectric coefficients of PVDF-based composites with various fillers.</p>
        <p>To get started:</p>
        <ol>
            <li>Select a filler from the dropdown in the sidebar</li>
            <li>Adjust the dopant fraction, fabrication method, and PVDF beta fraction</li>
            <li>Click the "Predict Piezoelectric Properties" button</li>
            <li>Explore the results in the main area</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Display example visualization
    st.markdown('<h2 class="sub-header">Example Visualization</h2>', unsafe_allow_html=True)
    
    # Create a sample tensor matrix for visualization
    sample_tensor = np.array([
        [0, 0, 0, 0, -24.1198, 0],
        [0, 0, 0, -20.5465, 0, 0],
        [18.0717, 1.3554, -28.9148, 0, 0, 0]
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(sample_tensor, annot=True, cmap="coolwarm", center=0, 
                fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'pC/N'})
    ax.set_title('Example Piezoelectric Tensor Matrix (pC/N)', fontsize=16)
    ax.set_xlabel('Tensor Component')
    ax.set_ylabel('Tensor Component')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>PVDF Composite Piezoelectric Predictor | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
