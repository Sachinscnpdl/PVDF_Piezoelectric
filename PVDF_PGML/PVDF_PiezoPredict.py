# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
import os
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the predictor functions from the modified module
try:
    from piezoelectric_tensor_predictor import PiezoelectricTensorPredictor
except ImportError:
    st.error("Error importing piezoelectric_tensor_predictor. Make sure the module is in the same directory.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Piezoelectric Tensor Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F0F9FF;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3B82F6;
    }
    .property-table {
        font-size: 0.9rem;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Load properties from JSON file
@st.cache_data
def load_properties_from_json():
    """Load material properties from JSON file"""
    try:
        with open('properties.json', 'r') as f:
            properties = json.load(f)
        return properties
    except FileNotFoundError:
        st.error("properties.json file not found! Please make sure it's in the same directory.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Error parsing properties.json. Please check the file format.")
        st.stop()

# Initialize the predictor
@st.cache_resource
def initialize_predictor(checkpoint_path):
    """Initialize the piezoelectric predictor"""
    try:
        predictor = PiezoelectricTensorPredictor(checkpoint_path=checkpoint_path, device='cpu')
        return predictor
    except Exception as e:
        st.error(f"Error initializing predictor: {str(e)}")
        st.stop()

# Load properties and predictor
properties = load_properties_from_json()
predictor = initialize_predictor('best_phys_resid_monotonic_improved_v2.pt')

# Extract dopant names from properties
available_dopants = [dopant for dopant in properties.keys() if dopant != 'PVDF']

# App title and description
st.markdown('<h1 class="main-header">⚡ Piezoelectric Tensor Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
This app predicts the piezoelectric coefficients for PVDF-based composites with various dopants.
Enter the composite parameters below to get predictions for d33, d31, d32, d15, and d24 coefficients.
""")

# Sidebar for input parameters
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Composite Parameters</h2>', unsafe_allow_html=True)
    
    # Dopant selection
    dopant = st.selectbox(
        "Select Dopant",
        available_dopants,
        help="Choose the filler material to add to PVDF"
    )
    
    # Volume fraction input
    fraction = st.slider(
        "Volume Fraction (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1,
        help="Volume percentage of dopant in the composite"
    )
    
    # Fabrication method selection
    method = st.selectbox(
        "Fabrication Method",
        ["Electrospinning", "Solution casting", "Poling", "Sol-gel"],
        help="Manufacturing method for the composite"
    )
    
    # Beta fraction override (optional)
    beta_fraction = st.number_input(
        "Beta Fraction (optional override)",
        min_value=0.1,
        max_value=1.0,
        value=0.55,
        step=0.01,
        help="Override the calculated beta fraction (leave to use calculated value)"
    )
    
    # Display dopant properties
    st.markdown('<h3 class="sub-header">📊 Dopant Properties</h3>', unsafe_allow_html=True)
    
    if dopant in properties:
        dopant_props = properties[dopant]
        prop_df = pd.DataFrame({
            'Property': list(dopant_props.keys()),
            'Value': list(dopant_props.values())
        })
        st.dataframe(prop_df, use_container_width=True, hide_index=True)
    
    # Prediction button
    predict_button = st.button("Predict Piezoelectric Coefficients", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction results section
    if predict_button:
        with st.spinner('Predicting piezoelectric coefficients...'):
            try:
                # Make prediction
                result = predictor.predict_sample(
                    dopant=dopant,
                    frac=fraction,
                    method=method,
                    beta_fraction=beta_fraction,
                    verbose=False
                )
                
                st.markdown('<h2 class="sub-header">📈 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Create a results DataFrame
                results_data = {
                    'Parameter': ['Dopant', 'Volume Fraction', 'Fabrication Method', 'Beta Fraction'],
                    'Value': [dopant, f"{fraction}%", method, f"{result.get('PVDF_Beta_Fraction_used', 'N/A')}"]
                }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Piezoelectric coefficients
                st.markdown('<h3 class="sub-header">⚡ Piezoelectric Coefficients (pC/N)</h3>', unsafe_allow_html=True)
                
                coeff_data = []
                for key, value in result.items():
                    if key.startswith('phys_') or key == 'predicted_d33':
                        if key == 'predicted_d33':
                            coeff_name = 'd33 (predicted)'
                        else:
                            coeff_name = key.replace('phys_', '')
                        coeff_data.append({'Coefficient': coeff_name, 'Value': f"{value:.4f}"})
                
                coeff_df = pd.DataFrame(coeff_data)
                st.dataframe(coeff_df, use_container_width=True, hide_index=True)
                
                # Display tensor matrix
                st.markdown('<h3 class="sub-header">🧮 Piezoelectric Tensor Matrix</h3>', unsafe_allow_html=True)
                
                # Parse the tensor matrix if available
                if 'piezoelectric_tensor' in result:
                    try:
                        tensor_matrix = eval(result['piezoelectric_tensor'])
                        
                        # Create a nice table
                        tensor_df = pd.DataFrame(
                            tensor_matrix,
                            columns=['d11', 'd12', 'd13', 'd14', 'd15', 'd16']
                        )
                        tensor_df.index = ['Row 1', 'Row 2', 'Row 3']
                        
                        st.dataframe(tensor_df.style.format("{:.4f}"), use_container_width=True)
                        
                        # Visual representation
                        st.markdown('<h4 class="sub-header">📊 Tensor Visualization</h4>', unsafe_allow_html=True)
                        
                        # Create heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=tensor_matrix,
                            colorscale='RdBu',
                            text=np.round(tensor_matrix, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            colorbar_title="pC/N"
                        ))
                        
                        fig.update_layout(
                            title="Piezoelectric Tensor Heatmap",
                            xaxis_title="Tensor Component",
                            yaxis_title="Tensor Row",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except:
                        st.write("Tensor matrix:", result['piezoelectric_tensor'])
                
                # Additional metrics
                st.markdown('<h3 class="sub-header">📊 Additional Metrics</h3>', unsafe_allow_html=True)
                
                metrics_data = {
                    'Physics Base d33': f"{result.get('physics_base_d33', 'N/A'):.4f} pC/N",
                    'Learned Delta': f"{result.get('learned_delta_d33', 'N/A'):.4f} pC/N",
                    'Effective Dielectric Constant': f"{result.get('Effective Dielectric Constant', 'N/A'):.4f}",
                    'Effective Young\'s Modulus': f"{result.get('Effective Youngs Modulus', 'N/A'):.4f} GPa"
                }
                
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics_data.keys()),
                    'Value': list(metrics_data.values())
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Download results
                st.markdown('<h3 class="sub-header">💾 Download Results</h3>', unsafe_allow_html=True)
                
                # Prepare CSV for download
                download_data = {
                    'Parameter': [],
                    'Value': []
                }
                
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        download_data['Parameter'].append(key)
                        download_data['Value'].append(value)
                
                download_df = pd.DataFrame(download_data)
                csv = download_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"piezoelectric_prediction_{dopant}_{fraction}percent.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Show help text if no prediction yet
    else:
        st.markdown("""
        ### Welcome to the Piezoelectric Tensor Predictor!
        
        To get started:
        1. Select a dopant material from the sidebar
        2. Adjust the volume fraction using the slider
        3. Choose the fabrication method
        4. Optionally adjust the beta fraction
        5. Click the **"Predict Piezoelectric Coefficients"** button
        
        The app will then predict the full piezoelectric tensor for your composite!
        """)
        
        # Show available dopants
        st.markdown('<h3 class="sub-header">📋 Available Dopants</h3>', unsafe_allow_html=True)
        
        dopant_list = []
        for dopant in available_dopants:
            if dopant in properties:
                filler_category = properties[dopant].get('Filler_Category', 'Not specified')
                dopant_list.append({
                    'Dopant': dopant,
                    'Category': filler_category
                })
        
        dopant_df = pd.DataFrame(dopant_list)
        st.dataframe(dopant_df, use_container_width=True, hide_index=True)

with col2:
    # Information panel
    st.markdown('<h2 class="sub-header">ℹ️ About the Model</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Model Features:**
    
    ✅ **Physics-informed ML**: Combines material physics with neural networks
    ✅ **Full tensor prediction**: Estimates d33, d31, d32, d15, d24 coefficients
    ✅ **Composite properties**: Calculates effective material properties
    ✅ **Manufacturing-aware**: Accounts for different fabrication methods
    ✅ **Beta-phase modeling**: Includes PVDF β-phase fraction effects
    
    **Key Assumptions:**
    
    • Base PVDF properties are fixed
    • Composite properties follow mixing rules
    • Tensor components follow standard PVDF conventions
    • Manufacturing methods affect baseline d33
    
    **Tensor Convention:**
    
    ```
    [0   0   0   0   d15   0]
    [0   0   0   d24  0   0]
    [d31 d32 d33 0   0   0]
    ```
    
    where:
    - d33: Thickness mode coefficient
    - d31, d32: Transverse coefficients
    - d15, d24: Shear coefficients
    """)
    
    # Quick reference
    st.markdown('<h3 class="sub-header">📚 Quick Reference</h3>', unsafe_allow_html=True)
    
    ref_data = {
        'Fabrication Method': ['Electrospinning', 'Solution casting', 'Poling', 'Sol-gel'],
        'Baseline d33 (pC/N)': [18.0, 6.0, 6.0, 6.0],
        'Beta Ref': [0.5496, 0.5, 0.5, 0.5]
    }
    
    ref_df = pd.DataFrame(ref_data)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Piezoelectric Tensor Predictor • Powered by Physics-Informed Machine Learning</p>
    <p>For research use only. Results may vary based on material properties and conditions.</p>
</div>
""", unsafe_allow_html=True)

# Batch prediction section
with st.expander("📁 Batch Prediction (Multiple Samples)"):
    st.markdown("""
    Upload a CSV file with multiple samples for batch prediction.
    
    **Required columns:**
    - `Dopants`: Name of the dopant material
    - `Dopants fr`: Volume fraction (percentage)
    - `Fabrication Method`: Manufacturing method (optional, defaults to Electrospinning)
    
    **Example CSV format:**
    ```csv
    Dopants,Dopants fr,Fabrication Method
    SnO2,1.5,Electrospinning
    CNT,0.5,Solution casting
    BaTiO3,2.0,Poling
    ```
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            batch_df = pd.read_csv(uploaded_file)
            
            if st.button("Run Batch Prediction"):
                with st.spinner('Processing batch prediction...'):
                    results = predictor.predict_batch(batch_df.to_dict('records'), verbose=False)
                    
                    st.success(f"Batch prediction completed for {len(results)} samples!")
                    
                    # Show results
                    st.dataframe(results, use_container_width=True)
                    
                    # Download batch results
                    batch_csv = results.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Batch Results",
                        data=batch_csv,
                        file_name="batch_piezoelectric_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")

# Make sure properties.json is accessible
try:
    with open('properties.json', 'r') as f:
        properties_loaded = json.load(f)
    st.sidebar.success(f"Loaded {len(properties_loaded)} materials from properties.json")
except:
    st.sidebar.error("Could not load properties.json")
