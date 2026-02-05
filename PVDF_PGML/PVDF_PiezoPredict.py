import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current directory to path to import the predictor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the actual predictor
try:
    from piezoelectric_tensor_predictor import predict_sample
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing predictor: {e}")
    PREDICTOR_AVAILABLE = False

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
    .prediction-highlight {
        background: linear-gradient(120deg, #f0f7ff 0%, #e6f0ff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
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

# Check if predictor is available
if not PREDICTOR_AVAILABLE:
    st.error("The piezoelectric predictor module is not available. Please ensure materials_properties.py is in the correct directory.")
    st.stop()

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

# Check if checkpoint file exists
checkpoint_path = os.path.join(current_dir, 'best_phys_resid_monotonic_improved_v2.pt')
checkpoint_exists = os.path.exists(checkpoint_path)

if not checkpoint_exists:
    st.sidebar.warning("⚠️ Checkpoint file not found. Using mock predictions.")
    use_mock = st.sidebar.checkbox("Use Mock Predictions", value=True)
else:
    use_mock = st.sidebar.checkbox("Use Mock Predictions", value=False)

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

# Mock prediction function (fallback)
def mock_predict_sample(dopant, frac, method, beta_fraction):
    """
    Mock prediction function that simulates the behavior of the original predict_sample function
    based on the sample output provided.
    """
    # Base values for different dopants (these are mock values based on the sample)
    dopant_base_values = {
        'SnO2': {'d33': 28.9148, 'd31': 18.0717, 'd32': 1.3554, 'd15': -24.1198, 'd24': -20.5465},
        'ZrO2': {'d33': 26.5, 'd31': 16.8, 'd32': 1.2, 'd15': -22.5, 'd24': -19.0},
        'PZT': {'d33': 35.2, 'd31': 22.1, 'd32': 1.8, 'd15': -29.3, 'd24': -24.8},
        'TiO2': {'d33': 24.8, 'd31': 15.6, 'd32': 1.1, 'd15': -20.9, 'd24': -17.6},
        'DA': {'d33': 22.3, 'd31': 14.2, 'd32': 0.9, 'd15': -18.7, 'd24': -15.8},
        'CoFe2O4': {'d33': 27.6, 'd31': 17.4, 'd32': 1.3, 'd15': -23.1, 'd24': -19.5},
        'CNT': {'d33': 31.4, 'd31': 19.8, 'd32': 1.5, 'd15': -26.2, 'd24': -22.1},
        'GO': {'d33': 30.1, 'd31': 19.0, 'd32': 1.4, 'd15': -25.0, 'd24': -21.2},
        'G': {'d33': 32.7, 'd31': 20.6, 'd32': 1.6, 'd15': -27.3, 'd24': -23.0},
        'Nfs': {'d33': 23.5, 'd31': 14.8, 'd32': 1.0, 'd15': -19.8, 'd24': -16.7},
        'BaTiO3': {'d33': 33.9, 'd31': 21.4, 'd32': 1.7, 'd15': -28.4, 'd24': -23.9}
    }
    
    # Get base values for the selected dopant
    base_values = dopant_base_values.get(dopant, dopant_base_values['SnO2'])
    
    # Adjust values based on fraction and beta fraction
    frac_factor = 1.0 + (frac - 1.5) * 0.1  # Linear adjustment based on fraction
    beta_factor = beta_fraction / 0.5725  # Adjustment based on beta fraction
    
    # Apply adjustments
    adjusted_values = {
        'd33': base_values['d33'] * frac_factor * beta_factor,
        'd31': base_values['d31'] * frac_factor * beta_factor,
        'd32': base_values['d32'] * frac_factor * beta_factor,
        'd15': base_values['d15'] * frac_factor * beta_factor,
        'd24': base_values['d24'] * frac_factor * beta_factor
    }
    
    # Create a DataFrame similar to the original
    df = pd.DataFrame({
        'predicted_d33': [adjusted_values['d33']],
        'phys_d31': [adjusted_values['d31']],
        'phys_d32': [adjusted_values['d32']],
        'phys_d15': [adjusted_values['d15']],
        'phys_d24': [adjusted_values['d24']]
    })
    
    return df

# Main content area
if predict_button:
    # Show loading spinner
    with st.spinner("Calculating piezoelectric properties..."):
        try:
            if use_mock or not checkpoint_exists:
                # Use mock prediction
                df = mock_predict_sample(
                    dopant=selected_filler,
                    frac=dopant_fraction,
                    method=fabrication_method,
                    beta_fraction=beta_fraction
                )
                prediction_source = "Mock Prediction"
            else:
                # Use actual prediction
                df = predict_sample(
                    checkpoint_path=checkpoint_path,
                    dopant=selected_filler,
                    frac=dopant_fraction,
                    method=fabrication_method,
                    beta_fraction=beta_fraction,
                    device=device_option
                )
                prediction_source = "Model Prediction"
            
            # Extract results
            predicted_d33 = df['predicted_d33'].values[0]
            phys_d31 = df['phys_d31'].values[0]
            phys_d32 = df['phys_d32'].values[0]
            phys_d15 = df['phys_d15'].values[0]
            phys_d24 = df['phys_d24'].values[0]
            
            # Create tensor matrix
            tensor_matrix = np.array([
                [0, 0, 0, 0, phys_d15, 0],
                [0, 0, 0, phys_d24, 0, 0],
                [phys_d31, phys_d32, -predicted_d33, 0, 0, 0]
            ])
            
            # Create features dictionary
            features = {
                "Dopant": selected_filler,
                "Dopant Fraction (%)": f"{dopant_fraction:.1f}",
                "Fabrication Method": fabrication_method,
                "PVDF Beta Fraction": f"{beta_fraction:.4f}",
                "Prediction Source": prediction_source,
                "Filler Category": "Oxide" if selected_filler in ["ZrO2", "TiO2", "SnO2", "BaTiO3"] else "Other"
            }
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Falling back to mock prediction...")
            df = mock_predict_sample(
                dopant=selected_filler,
                frac=dopant_fraction,
                method=fabrication_method,
                beta_fraction=beta_fraction
            )
            prediction_source = "Mock Prediction (Fallback)"
            
            # Extract results
            predicted_d33 = df['predicted_d33'].values[0]
            phys_d31 = df['phys_d31'].values[0]
            phys_d32 = df['phys_d32'].values[0]
            phys_d15 = df['phys_d15'].values[0]
            phys_d24 = df['phys_d24'].values[0]
            
            # Create tensor matrix
            tensor_matrix = np.array([
                [0, 0, 0, 0, phys_d15, 0],
                [0, 0, 0, phys_d24, 0, 0],
                [phys_d31, phys_d32, -predicted_d33, 0, 0, 0]
            ])
            
            # Create features dictionary
            features = {
                "Dopant": selected_filler,
                "Dopant Fraction (%)": f"{dopant_fraction:.1f}",
                "Fabrication Method": fabrication_method,
                "PVDF Beta Fraction": f"{beta_fraction:.4f}",
                "Prediction Source": prediction_source,
                "Filler Category": "Oxide" if selected_filler in ["ZrO2", "TiO2", "SnO2", "BaTiO3"] else "Other"
            }
    
    # Display results
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    # Key metrics with enhanced highlighting
    st.markdown('<div class="prediction-highlight">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature table with enhanced styling
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
    
    # Piezoelectric tensor visualization with enhanced styling
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
    
    # Comparison chart with enhanced styling
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
                              'd33 (pC/N)', 'd31 (pC/N)', 'd32 (pC/N)', 'd15 (pC/N)', 'd24 (pC/N)', 'Prediction Source'],
                'Value': [selected_filler, dopant_fraction, fabrication_method, beta_fraction,
                          predicted_d33, phys_d31, phys_d32, phys_d15, phys_d24, prediction_source]
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
