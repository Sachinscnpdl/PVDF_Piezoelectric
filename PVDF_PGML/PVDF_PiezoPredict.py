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

# Check if all required files exist
st.sidebar.markdown("### File Status")

required_files = {
    "materials_properties.py": materials_properties_path,
    "checkpoint file": checkpoint_path,
    "predictor module": predictor_path
}

missing_files = []
for file_name, file_path in required_files.items():
    if os.path.exists(file_path):
        st.sidebar.success(f"✓ {file_name}")
    else:
        st.sidebar.error(f"✗ {file_name} not found")
        missing_files.append(file_name)

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.stop()

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
    st.stop()

# Import the predictor functions
try:
    sys.path.insert(0, current_dir)
    from piezoelectric_tensor_predictor import predict_sample, predict_dataframe
    st.sidebar.success("✓ Predictor module loaded")
except Exception as e:
    st.error(f"Error importing predictor: {str(e)}")
    st.stop()

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
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">PVDF Composite Piezoelectric Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.1rem; margin-bottom: 2rem;">
Predict piezoelectric coefficients of PVDF-based composites with various fillers using physics-informed machine learning.
</p>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Input Parameters</h2>', unsafe_allow_html=True)
    
    # Available fillers from properties
    fillers = list(properties_data.keys())
    fillers.remove('PVDF')  # Remove PVDF from the list of fillers
    
    # Input fields
    selected_filler = st.selectbox("Select Filler", fillers)
    dopant_fraction = st.slider("Dopant Fraction (%)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    fabrication_method = st.selectbox("Fabrication Method", ["Electrospinning", "Solution casting", "Poling", "Sol-gel"])
    beta_fraction = st.number_input("Beta Fraction (optional override)", min_value=0.0, max_value=1.0, value=0.55, step=0.01, 
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
    
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Key metrics in cards
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
    
    # Material properties
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
    
    # Show full results in expander
    with st.expander("View Detailed Prediction Results"):
        st.dataframe(df)
    
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
    # Welcome screen when no prediction has been made
    st.markdown('<h2 class="sub-header">Welcome to the PVDF Composite Piezoelectric Predictor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        <p>This tool predicts the piezoelectric coefficients of PVDF-based composites using physics-informed machine learning.</p>
        <p>To get started:</p>
        <ol>
            <li>Select a filler material from the sidebar</li>
            <li>Adjust the dopant fraction and fabrication method</li>
            <li>Optionally override the beta fraction (or leave as 0.0 for calculated value)</li>
            <li>Click the <strong>"Predict Piezoelectric Properties"</strong> button</li>
            <li>Explore the results including the full piezoelectric tensor</li>
        </ol>
        
        <p><strong>Available fillers:</strong> {}</p>
    </div>
    """.format(", ".join(fillers)), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>PVDF Composite Piezoelectric Predictor | Powered by Physics-Informed Machine Learning</p>
</div>
""", unsafe_allow_html=True)
