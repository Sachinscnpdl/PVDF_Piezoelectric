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
import tempfile

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
json_path = os.path.join(current_dir, 'properties.json')
py_path = os.path.join(current_dir, 'materials_properties.py')

# Check if properties.json exists
if not os.path.exists(json_path):
    st.error(f"❌ properties.json not found at {json_path}")
    st.stop()

# Load properties from JSON
with open(json_path, 'r') as f:
    properties_data = json.load(f)

# Create materials_properties.py with proper formatting
with open(py_path, 'w') as f:
    f.write('# materials_properties.py\n')
    f.write('# Auto-generated from properties.json\n\n')
    f.write('properties = {\n')
    
    # Write each dopant
    for i, (dopant, props) in enumerate(properties_data.items()):
        f.write(f'  "{dopant}": {{\n')
        
        # Write each property
        for j, (key, value) in enumerate(props.items()):
            # Format value properly
            if isinstance(value, str):
                formatted_value = f'"{value}"'
            elif isinstance(value, (int, float)):
                formatted_value = str(value)
            else:
                formatted_value = f'"{str(value)}"'
            
            # Check if it's the last property
            comma = ',' if j < len(props) - 1 else ''
            f.write(f'    "{key}": {formatted_value}{comma}\n')
        
        # Check if it's the last dopant
        comma = ',' if i < len(properties_data) - 1 else ''
        f.write(f'  }}{comma}\n')
    
    f.write('}\n')

print(f"Created {py_path}")

# Now import the predictor
try:
    # First, let's read and modify the predictor code to handle the properties loading
    predictor_file = os.path.join(current_dir, 'piezoelectric_tensor_predictor.py')
    with open(predictor_file, 'r') as f:
        predictor_code = f.read()
    
    # Modify the code to use our properties directly
    # Replace the load_properties function
    import re
    
    # Pattern to find the load_properties function
    pattern = r'def load_properties\(\):.*?raise FileNotFoundError\("Please provide materials_properties\.py or properties\.json"\)'
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
    
    # Now import the predictor functions
    from piezoelectric_tensor_predictor import predict_sample
    
except Exception as e:
    st.error(f"Error loading predictor: {str(e)}")
    # Create a mock predict_sample function for testing
    def predict_sample(checkpoint_path, dopant, frac, method, beta_fraction=None, device='cpu'):
        # Mock prediction for testing
        return pd.DataFrame({
            'Dopants': [dopant],
            'Dopants fr': [frac],
            'Fabrication Method': [method],
            'PVDF_Beta_Fraction_used': [beta_fraction or 0.55],
            'physics_base_d33': [20.0],
            'learned_delta_d33': [5.0],
            'predicted_d33': [25.0],
            'phys_d33': [-25.0],
            'phys_d31': [15.625],
            'phys_d32': [1.172],
            'phys_d15': [-13.125],
            'phys_d24': [-11.2],
            'Effective Dielectric Constant': [12.5],
            'Effective Thermal Conductivity': [1.2],
            'Effective Youngs Modulus': [3.2],
            'Effective Poissons Ratio': [0.29],
            'Effective Density': [2.0],
            'Effective Specific Heat Capacity': [1.3]
        })
    
    st.warning("⚠️ Using mock predictor for demonstration")

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

# Available fillers from properties.json
fillers = list(properties_data.keys())
fillers.remove('PVDF')  # Remove PVDF from the list of fillers

# Input fields
selected_filler = st.sidebar.selectbox("Select Filler", fillers)
dopant_fraction = st.sidebar.slider("Dopant Fraction (%)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
fabrication_method = st.sidebar.selectbox("Fabrication Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
beta_fraction = st.sidebar.slider("PVDF Beta Fraction (optional override)", min_value=0.1, max_value=1.0, value=0.5725, step=0.01)
device_option = st.sidebar.selectbox("Device", ["cpu"])

# Display filler properties
st.sidebar.markdown('<h3 class="sub-header">Filler Properties</h3>', unsafe_allow_html=True)
if selected_filler in properties_data:
    filler_props = properties_data[selected_filler]
    for key, value in filler_props.items():
        if not key.startswith('_'):  # Skip comment fields
            st.sidebar.text(f"{key}: {value}")

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
        try:
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
            
            # Convert to DataFrame if it's not already
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame([df])
            
            # Extract results
            predicted_d33 = float(df['predicted_d33'].iloc[0])
            phys_d31 = float(df.get('phys_d31', [15.625])[0])
            phys_d32 = float(df.get('phys_d32', [1.172])[0])
            phys_d15 = float(df.get('phys_d15', [-13.125])[0])
            phys_d24 = float(df.get('phys_d24', [-11.2])[0])
            
            # Extract additional features from the prediction
            features = {
                "Dopant": selected_filler,
                "Dopant Fraction (%)": f"{dopant_fraction:.1f}",
                "Fabrication Method": fabrication_method,
                "PVDF Beta Fraction": f"{df.get('PVDF_Beta_Fraction_used', [beta_fraction])[0]:.4f}",
                "Effective Dielectric Constant": f"{df.get('Effective Dielectric Constant', [12.5])[0]:.4f}",
                "Effective Thermal Conductivity": f"{df.get('Effective Thermal Conductivity', [1.2])[0]:.4f}",
                "Effective Young's Modulus": f"{df.get('Effective Youngs Modulus', [3.2])[0]:.4f} GPa",
                "Effective Poisson's Ratio": f"{df.get('Effective Poissons Ratio', [0.29])[0]:.4f}",
                "Effective Density": f"{df.get('Effective Density', [2.0])[0]:.4f} g/cm³",
                "Effective Specific Heat Capacity": f"{df.get('Effective Specific Heat Capacity', [1.3])[0]:.4f} J/g·K",
                "Physics Base d33": f"{df.get('physics_base_d33', [20.0])[0]:.4f} pC/N",
                "Learned Delta d33": f"{df.get('learned_delta_d33', [5.0])[0]:.4f} pC/N",
                "Filler Category": properties_data.get(selected_filler, {}).get('Filler_Category', 'Not specified')
            }
            
            # Create tensor matrix
            tensor_matrix = np.array([
                [0, 0, 0, 0, phys_d15, 0],
                [0, 0, 0, phys_d24, 0, 0],
                [phys_d31, phys_d32, -predicted_d33, 0, 0, 0]
            ])
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            # Use mock data for demonstration
            predicted_d33 = 25.0
            phys_d31 = 15.625
            phys_d32 = 1.172
            phys_d15 = -13.125
            phys_d24 = -11.2
            
            features = {
                "Dopant": selected_filler,
                "Dopant Fraction (%)": f"{dopant_fraction:.1f}",
                "Fabrication Method": fabrication_method,
                "PVDF Beta Fraction": f"{beta_fraction:.4f}",
                "Effective Dielectric Constant": "12.5",
                "Effective Thermal Conductivity": "1.2",
                "Effective Young's Modulus": "3.2 GPa",
                "Effective Poisson's Ratio": "0.29",
                "Effective Density": "2.0 g/cm³",
                "Effective Specific Heat Capacity": "1.3 J/g·K",
                "Physics Base d33": "20.0 pC/N",
                "Learned Delta d33": "5.0 pC/N",
                "Filler Category": properties_data.get(selected_filler, {}).get('Filler_Category', 'Not specified')
            }
            
            tensor_matrix = np.array([
                [0, 0, 0, 0, phys_d15, 0],
                [0, 0, 0, phys_d24, 0, 0],
                [phys_d31, phys_d32, -predicted_d33, 0, 0, 0]
            ])
    
    # Display results
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d33</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{predicted_d33:.2f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d31</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d31:.2f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d32</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d32:.2f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d15</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d15:.2f} pC/N</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #4b6cb7;">d24</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="font-weight: bold; color: #182848;">{phys_d24:.2f} pC/N</h2>', unsafe_allow_html=True)
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
    st.dataframe(tensor_df.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison chart
    st.markdown('<h2 class="sub-header">Piezoelectric Coefficients Comparison</h2>', unsafe_allow_html=True)
    
    # Create a bar chart for the coefficients
    coefficients = {
        'd33': predicted_d33,
        'd31': phys_d31,
        'd32': phys_d32,
        'd15': abs(phys_d15),
        'd24': abs(phys_d24)
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
        [0, 0, 0, 0, -24.12, 0],
        [0, 0, 0, -20.55, 0, 0],
        [18.07, 1.36, -28.91, 0, 0, 0]
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
