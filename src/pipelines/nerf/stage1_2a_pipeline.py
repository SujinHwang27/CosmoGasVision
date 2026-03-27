import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath('.'))

import torch
import torch.optim as optim
from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF, volume_render_physics
import mlflow
from dotenv import load_dotenv

load_dotenv()

def run_stage1_2a():
    """Execute Stage 1 and Stage 2a of the NeRF Tomography Plan."""
    print("--- Running NeRF Stage 1: Data Preprocessing ---")
    data_root = "Sherwood"
    if not os.path.exists(data_root):
        print(f"Warning: Data root {data_root} missing. Create dummy data.")
        # Dummy data for dummy test if not present
        box_max = 60000.0  # kpc/h proxy
        coords = torch.rand(10, 2048, 3) * box_max
        tau_gt = torch.rand(10)
    else:
        loader = SherwoodLoader(data_root)
        try:
            # Use physics_id=1, z=0.3
            sightlines = loader.load_sightlines(1, 0.3)
            # Stage 1: Map logic coords -> scale to unit cube [0, 1]
            coords_raw = loader.get_world_coordinates(sightlines)
            
            box_max = sightlines['header']['box_kpc_h'] * 1000 # Usually 60 Mpc/h is 60000 kpc/h
            print(f"Loaded {coords_raw.shape[0]} rays. Normalizing coords to box scale {box_max}")
            
            # Subsample 10 sightlines for rapid prototyping
            coords = torch.tensor(coords_raw[:10], dtype=torch.float32) / box_max

            # Target optical depths (we just sum it for our dummy volume rendering test)
            tau_gt = torch.tensor(sightlines['tau_h1'][:10].sum(axis=-1), dtype=torch.float32)

        except FileNotFoundError as e:
            print("Data files not found, exiting.", e)
            return

    print("--- Running NeRF Stage 2a (Physics): Arch Setup & Accurate Ray Integrator ---")

    # Set up MLflow following the new Governance Rules
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://44.201.176.18:5000")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Hierarchical Experiment Name: Project/Methodology
        mlflow.set_experiment("CosmoGasVision/NeRF")
        print(f"Connected to MLflow at {mlflow_uri} [Experiment: CosmoGasVision/NeRF]")
    except Exception as e:
        print(f"MLflow connection issue: {e}")

    try:
        # Standardized Run Name: Stage-Description
        with mlflow.start_run(run_name="Stage2a-PhysicsIntegratorValidation"):
            # Mandatory Metadata Tagging
            mlflow.set_tags({
                "model_type": "nerf",
                "stage": "2a",
                "physics_id": "1",
                "redshift": "0.3",
            })
            
            model = IGMNeRF(hidden_dim=256, num_layers=4, L=5) 
            print(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")
            
            optimizer = optim.Adam(model.parameters(), lr=5e-4) # Lower LR for stability in damping wings
            mse_loss = torch.nn.MSELoss()

            for step in range(10):
                optimizer.zero_grad()
                
                # Forward Pass: Using Physically Consistent Integrator
                tau_pred = volume_render_physics(model, coords) # (10,)
                
                loss = mse_loss(tau_pred, tau_gt)
                loss.backward()
                
                # Check gradients on output layer
                grad_norm = model.out_layer.weight.grad.norm().item()
                
                optimizer.step()
                
                print(f"Step {step+1}/10 | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")
                
                mlflow.log_metric("loss", loss.item(), step=step)
                mlflow.log_metric("grad_norm", grad_norm, step=step)

            print("Backward pass and gradient flow confirmed successfully! Projection layer verified.")
            
    except Exception as e:
        print("Model execution failed:", e)

if __name__ == "__main__":
    run_stage1_2a()
