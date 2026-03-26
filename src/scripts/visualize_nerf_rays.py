import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from src.data.loader import SherwoodLoader

def generate_ray_visualizations():
    print("--- Support Researcher: Generating NeRF Ray Visualizations ---")
    data_root = "Sherwood"
    
    # Placeholder for ray data
    nbins = 2048
    if not os.path.exists(data_root):
        print(f"Data root {data_root} missing. Creating dummy continuous field for visualization.")
        np.random.seed(42)
        pos_axis = np.linspace(0, 60000, nbins)
        density = np.abs(np.sin(pos_axis / 2000)) * np.exponential(1.5, nbins)
        temp = np.random.normal(20000, 5000, nbins)
        h1_frac = np.random.beta(0.5, 5, nbins)
        tau_target = density ** 1.5 * np.exp(-temp / 1e5)
    else:
        # Load real data
        loader = SherwoodLoader(data_root)
        try:
            sightlines = loader.load_sightlines(1, 0.3)
            # Pick a single highly varying ray for clear demonstration
            idx = 10 
            
            pos_axis = sightlines['pos_axis']
            density = sightlines['density'][idx]
            temp = sightlines['temp'][idx]
            h1_frac = sightlines['h1_frac'][idx]
            tau_target = sightlines['tau_h1'][idx]
            
        except FileNotFoundError as e:
            print("Data files not found, exiting.", e)
            return

    df = pd.DataFrame({
        'Position_kpch': pos_axis,
        'Density': density,
        'Temperature': temp,
        'H1_Fraction': h1_frac,
        'Tau_GroundTruth': tau_target
    })
    
    # Generate Multi-axis subplot to display physical relations on the sightline
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Density trace
    fig.add_trace(
        go.Scatter(x=df['Position_kpch'], y=df['Density'], name="Density ρ/⟨ρ⟩", line=dict(color='blue', width=1)),
        secondary_y=False,
    )
    
    # Ground Truth Tau trace
    fig.add_trace(
        go.Scatter(x=df['Position_kpch'], y=df['Tau_GroundTruth'], name="Ground Truth τ (Absorption)", line=dict(color='red', width=1.5)),
        secondary_y=True,
    )
    
    # H1_Fraction trace (using a third proxy axis, plotted on density scale mathematically scaled for visibility)
    fig.add_trace(
        go.Scatter(x=df['Position_kpch'], y=df['H1_Fraction']*1e4, name="H1 Frac (x10^4)", line=dict(color='green', dash='dot', width=1)),
        secondary_y=False,
    )

    fig.update_layout(
        title="NeRF Stage 2a Project Validation: Cosmological Physics mapping vs Target Optical Depth",
        xaxis_title="Comoving Position along Ray (kpc/h)",
        plot_bgcolor='rgb(10, 10, 10)',
        paper_bgcolor='rgb(10, 10, 10)',
        font=dict(color='white')
    )
    
    fig.update_yaxes(title_text="Physical Fields (Density & scaled H1)", secondary_y=False)
    fig.update_yaxes(title_text="Target Flux / Optical Depth (τ)", secondary_y=True)

    # Manage Artifact Location natively respecting isolation
    out_dir = "artifacts/nerf/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    html_out = os.path.join(out_dir, "ray_integration_fields.html")
    # Write interactive dynamic plot to local directory
    fig.write_html(html_out)
    
    # Generate Stats Table of this specific Ray
    stats_df = df[['Density', 'Temperature', 'H1_Fraction', 'Tau_GroundTruth']].describe()
    stats_csv = os.path.join(out_dir, "ray_field_statistics.csv")
    stats_df.to_csv(stats_csv)

    # MLflow Logging targeting specific NeRF branch
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://44.201.176.18:5000")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("exp/nerf")
        print(f"Connected to MLflow at {mlflow_uri}")
    except Exception as e:
        print(f"MLflow connection issue: {e}")

    try:
        with mlflow.start_run(run_name="stage2a_ray_visualization"):
            # Record base metrics
            mlflow.log_metric("max_density_ray", float(df['Density'].max()))
            mlflow.log_metric("max_tau_ray", float(df['Tau_GroundTruth'].max()))
            
            # Log the interactive HTML widget and stats table securely to tracking server
            mlflow.log_artifact(html_out, artifact_path="interactive_plots")
            mlflow.log_artifact(stats_csv, artifact_path="statistics")
            
            print(f"Artifacts successfully logged directly to MLflow Run ID: {mlflow.active_run().info.run_id}")
            
    except Exception as e:
        print("MLflow execution failed:", e)

if __name__ == "__main__":
    generate_ray_visualizations()
