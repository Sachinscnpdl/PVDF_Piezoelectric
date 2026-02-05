# -*- coding: utf-8 -*-
import os
import json
import sys
import tempfile
from pathlib import Path

# Get the current directory where this script is located
current_dir = Path(__file__).parent.absolute()

# Load properties from JSON file
properties_path = current_dir / 'properties.json'
if not properties_path.exists():
    # Try to download from GitHub if not found
    import requests
    try:
        url = "https://raw.githubusercontent.com/Sachinscnpdl/PVDF_Piezoelectric/main/PVDF_PGML/properties.json"
        response = requests.get(url)
        if response.status_code == 200:
            properties_data = response.json()
            with open(properties_path, 'w') as f:
                json.dump(properties_data, f, indent=2)
        else:
            raise FileNotFoundError(f"properties.json not found at {properties_path} and could not download from GitHub")
    except ImportError:
        raise FileNotFoundError(f"properties.json not found at {properties_path}")
else:
    with open(properties_path, 'r') as f:
        properties_data = json.load(f)

# Create materials_properties.py if it doesn't exist
materials_py_path = current_dir / 'materials_properties.py'
if not materials_py_path.exists():
    with open(materials_py_path, 'w') as f:
        f.write(f'# Auto-generated from properties.json\n')
        f.write(f'properties = {json.dumps(properties_data, indent=2)}\n')

# Now import the predictor module
try:
    # Add current directory to sys.path
    sys.path.insert(0, str(current_dir))
    
    # Import the predictor
    from piezoelectric_tensor_predictor import PiezoelectricTensorPredictor
except ImportError as e:
    # Try alternative approach - create a modified version of the predictor
    print(f"Standard import failed: {e}")
    
    # Read the original predictor code
    predictor_file = current_dir / 'piezoelectric_tensor_predictor.py'
    with open(predictor_file, 'r') as f:
        predictor_code = f.read()
    
    # Create a temporary file with modified import logic
    import re
    
    # Pattern to find the problematic load_properties function
    pattern = r'def load_properties\(\):.*?raise FileNotFoundError\("Please provide materials_properties\.py or properties\.json"\)'
    
    # Replacement that uses our already-loaded properties
    replacement = f'''def load_properties():
    return {json.dumps(properties_data, indent=2)}'''
    
    # Replace the function
    modified_code = re.sub(pattern, replacement, predictor_code, flags=re.DOTALL)
    
    if modified_code == predictor_code:
        # If regex didn't work, do a simpler replacement
        modified_code = predictor_code.replace(
            'properties = load_properties()',
            f'properties = {json.dumps(properties_data, indent=2)}'
        )
    
    # Write modified code to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_predictor_file = os.path.join(temp_dir, 'piezoelectric_tensor_predictor.py')
    with open(temp_predictor_file, 'w') as f:
        f.write(modified_code)
    
    # Add temp directory to path and import
    sys.path.insert(0, temp_dir)
    
    try:
        from piezoelectric_tensor_predictor import PiezoelectricTensorPredictor
    except ImportError as e2:
        print(f"Modified import also failed: {e2}")
        # Last resort: create a minimal mock
        class PiezoelectricTensorPredictor:
            def __init__(self, checkpoint_path='best_phys_resid_monotonic_improved_v2.pt', device='cpu'):
                self.checkpoint_path = checkpoint_path
                self.device = device
                
            def predict_sample(self, dopant, frac, method, beta_fraction=None, verbose=False):
                # Mock prediction for testing
                return {
                    'Dopants': dopant,
                    'Dopants fr': frac,
                    'Fabrication Method': method,
                    'PVDF_Beta_Fraction_used': beta_fraction or 0.55,
                    'physics_base_d33': 20.0,
                    'learned_delta_d33': 5.0,
                    'predicted_d33': 25.0,
                    'phys_d33': -25.0,
                    'phys_d31': 15.625,
                    'phys_d32': 1.172,
                    'phys_d15': -13.125,
                    'phys_d24': -11.2,
                    'Effective Dielectric Constant': 12.5,
                    'Effective Youngs Modulus': 3.2,
                    'piezoelectric_tensor': '[[0.0, 0.0, 0.0, 0.0, -13.125, 0.0], [0.0, 0.0, 0.0, -11.2, 0.0, 0.0], [15.625, 1.172, -25.0, 0.0, 0.0, 0.0]]'
                }
            
            def predict_batch(self, samples, verbose=False):
                # Mock batch prediction
                results = []
                for sample in samples:
                    result = self.predict_sample(
                        sample.get('Dopants', 'Unknown'),
                        sample.get('Dopants fr', 1.5),
                        sample.get('Fabrication Method', 'Electrospinning')
                    )
                    results.append(result)
                return pd.DataFrame(results)

# Now import Streamlit and other dependencies
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Load properties from JSON file for the app
@st.cache_data
def load_properties_from_json():
    """Load material properties from JSON file"""
    try:
        with open(properties_path, 'r') as f:
            properties = json.load(f)
        return properties
    except FileNotFoundError:
        st.error(f"properties.json file not found at {properties_path}!")
        st.stop()
    except json.JSONDecodeError:
        st.error("Error parsing properties.json. Please check the file format.")
        st.stop()

# Initialize the predictor
@st.cache_resource
def initialize_predictor(checkpoint_path):
    """Initialize the piezoelectric predictor"""
    try:
        # Use absolute path for checkpoint
        checkpoint_abs_path = current_dir / checkpoint_path
        if not checkpoint_abs_path.exists():
            st.error(f"Checkpoint file not found: {checkpoint_abs_path}")
            # Try to find it in current directory
            checkpoint_abs_path = current_dir / 'best_phys_resid_monotonic_improved_v2.pt'
        
        predictor = PiezoelectricTensorPredictor(
            checkpoint_path=str(checkpoint_abs_path),
            device='cpu'
        )
        return predictor
    except Exception as e:
        st.error(f"Error initializing predictor: {str(e)}")
        # Return a mock predictor for debugging
        st.warning("Using mock predictor for demonstration")
        return PiezoelectricTensorPredictor()

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
                    beta_fraction=beta_fraction if beta_fraction else None,
                    verbose=False
                )
                
                # Convert to dictionary if it's a DataFrame row
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                
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
                            display_value = f"{value:.4f}"
                        elif key.startswith('phys_'):
                            coeff_name = key.replace('phys_', '')
                            display_value = f"{value:.4f}"
                        else:
                            continue
                        coeff_data.append({'Coefficient': coeff_name, 'Value': display_value})
                
                # Add any missing coefficients
                expected_coeffs = ['d33', 'd31', 'd32', 'd15', 'd24']
                for coeff in expected_coeffs:
                    if not any(c['Coefficient'] == coeff for c in coeff_data):
                        if f'phys_{coeff}' in result:
                            coeff_data.append({'Coefficient': coeff, 'Value': f"{result[f'phys_{coeff}']:.4f}"})
                
                coeff_df = pd.DataFrame(coeff_data)
                st.dataframe(coeff_df, use_container_width=True, hide_index=True)
                
                # Display tensor matrix
                st.markdown('<h3 class="sub-header">🧮 Piezoelectric Tensor Matrix</h3>', unsafe_allow_html=True)
                
                # Try to parse the tensor matrix
                tensor_matrix = None
                if 'piezoelectric_tensor' in result:
                    try:
                        if isinstance(result['piezoelectric_tensor'], str):
                            tensor_matrix = eval(result['piezoelectric_tensor'])
                        else:
                            tensor_matrix = result['piezoelectric_tensor']
                    except:
                        # Try to build matrix from components
                        tensor_matrix = [
                            [0.0, 0.0, 0.0, 0.0, result.get('phys_d15', 0.0), 0.0],
                            [0.0, 0.0, 0.0, result.get('phys_d24', 0.0), 0.0, 0.0],
                            [result.get('phys_d31', 0.0), result.get('phys_d32', 0.0), result.get('phys_d33', result.get('predicted_d33', 0.0)), 0.0, 0.0, 0.0]
                        ]
                
                if tensor_matrix:
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
                else:
                    st.info("Tensor matrix not available in results.")
                
                # Additional metrics
                st.markdown('<h3 class="sub-header">📊 Additional Metrics</h3>', unsafe_allow_html=True)
                
                metrics_data = {
                    'Physics Base d33': f"{result.get('physics_base_d33', 'N/A'):.4f} pC/N",
                    'Learned Delta': f"{result.get('learned_delta_d33', 'N/A'):.4f} pC/N",
                    'Effective Dielectric Constant': f"{result.get('Effective Dielectric Constant', 'N/A'):.4f}",
                    'Effective Young\'s Modulus': f"{result.get('Effective Youngs Modulus', 'N/A'):.4f} GPa"
                }
                
                # Filter out N/A values
                metrics_data = {k: v for k, v in metrics_data.items() if 'N/A' not in v}
                
                if metrics_data:
                    metrics_df = pd.DataFrame({
                        'Metric': list(metrics_data.keys()),
                        'Value': list(metrics_data.values())
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Download results
                st.markdown('<h3 class="sub-header">💾 Download Results</h3>', unsafe_allow_html=True)
                
                # Prepare CSV for download
                download_data = {}
                for key, value in result.items():
                    if isinstance(value, (int, float, str)):
                        download_data[key] = [value]
                
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
                import traceback
                st.code(traceback.format_exc())
    
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
        for dop in available_dopants:
            if dop in properties:
                filler_category = properties[dop].get('Filler_Category', 'Not specified')
                dopant_list.append({
                    'Dopant': dop,
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
                    # Convert to list of dicts
                    samples = batch_df.to_dict('records')
                    results = predictor.predict_batch(samples, verbose=False)
                    
                    # Handle both DataFrame and list of dicts
                    if not isinstance(results, pd.DataFrame):
                        results = pd.DataFrame(results)
                    
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
            import traceback
            st.code(traceback.format_exc())

# File status info
with st.sidebar:
    st.markdown("---")
    st.markdown("**File Status:**")
    
    # Check required files
    required_files = ['properties.json', 'piezoelectric_tensor_predictor.py', 'best_phys_resid_monotonic_improved_v2.pt']
    
    for file in required_files:
        file_path = current_dir / file
        if file_path.exists():
            st.success(f"✓ {file}")
        else:
            st.error(f"✗ {file} not found")
    
    # Display properties info
    if 'properties' in locals():
        st.info(f"Loaded {len(properties)} materials")
