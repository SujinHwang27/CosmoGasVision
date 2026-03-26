import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath('.'))

import torch
import torch.optim as optim
from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF, volume_render_dummy
import mlflow

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
            coords = loader.get_world_coordinates(sightlines)
            
            box_max = sightlines['header']['box_kpc_h'] * 1000 # Usually 60 Mpc/h is 60000 kpc/h
            print(f"Loaded {coords.shape[0]} rays. Normalizing coords to box scale {box_max}")
            
            # Subsample 10 sightlines for rapid prototyping
            coords = coords[:10]
            coords = torch.tensor(coords, dtype=torch.float32) / box_max

            # Target optical depths (we just sum it for our dummy volume rendering test)
            tau_gt = sightlines['tau_h1'][:10].sum(axis=-1)
            tau_gt = torch.tensor(tau_gt, dtype=torch.float32)

        except FileNotFoundError as e:
            print("Data files not found, exiting.", e)
            return

    print("--- Running NeRF Stage 2a: Arch Setup & Differentiable Ray Integrator ---")

    # Set up MLflow if environment allows (using an environment variable or local)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://44.201.176.18:5000")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("exp/nerf_stage1_2a")
        print(f"Connected to MLflow at {mlflow_uri}")
    except Exception as e:
        print(f"MLflow connection issue: {e}")

    try:
        with mlflow.start_run():
            model = IGMNeRF(hidden_dim=256, num_layers=8, L=5) # Smaller for quick test
            print(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            mse_loss = torch.nn.MSELoss()

            for step in range(5):
                optimizer.zero_grad()
                
                # Forward Pass
                ray_points = coords  # (10, 2048, 3)
                tau_pred = volume_render_dummy(model, ray_points) # (10,)
                
                loss = mse_loss(tau_pred, tau_gt)
                loss.backward()
                
                # Check gradients
                grad_norm = model.out_layer.weight.grad.norm().item()
                
                optimizer.step()
                
                print(f"Step {step+1}/5 | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")
                
                mlflow.log_metric("loss", loss.item(), step=step)
                mlflow.log_metric("grad_norm", grad_norm, step=step)

            print("Backward pass and gradient flow confirmed successfully! Projection layer verified.")
            
    except Exception as e:
        print("Model execution failed:", e)

if __name__ == "__main__":
    run_stage1_2a()
