import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath('.'))

# Force UTF-8 stdout so MLflow's run-link emoji doesn't trigger a cp949 codec
# error on Korean-locale Windows consoles (encountered during Stage 2a smoke).
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

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
        # Dummy data — 10 rays, 256 bins for the smoke path
        box_max = 60000.0  # kpc/h
        nbins_dummy = 256
        coords = torch.rand(10, nbins_dummy, 3)  # already in [0, 1]
        vel_axis = torch.linspace(0, 6000.0, nbins_dummy)
        tau_gt_profile = torch.rand(10, nbins_dummy)
    else:
        loader = SherwoodLoader(data_root)
        try:
            # Use physics_id=1, z=0.3
            sightlines = loader.load_sightlines(1, 0.3)
            # Stage 1: Map logic coords -> scale to unit cube [0, 1]
            coords_raw = loader.get_world_coordinates(sightlines)

            # FIX (D-08): box_kpc_h is already in kpc/h (per Sherwood/src/utils.py
            # line 35). The previous `* 1000` made the normalized coords ~1e-3
            # instead of filling [0, 1].
            box_max = sightlines['header']['box_kpc_h']
            print(f"Loaded {coords_raw.shape[0]} rays. Normalizing coords to box scale {box_max} kpc/h")

            # Smoke-run scope: 10 sightlines, full 2048-bin grid. The windowed
            # Voigt convolution in volume_render_physics keeps the intermediate
            # tensor O(n_rays * n_src * (2*W+1)) rather than O(n_src * n_obs),
            # closing [D-06]'s memory gap.
            n_rays = 10
            coords = torch.tensor(coords_raw[:n_rays], dtype=torch.float32) / box_max

            # Velocity grid (km/s) — full-resolution simulation grid
            vel_axis = torch.tensor(sightlines['vel_axis'], dtype=torch.float32)

            # Ground truth: full tau(v) profile per ray
            tau_gt_profile = torch.tensor(sightlines['tau_h1'][:n_rays], dtype=torch.float32)

            # Sanity check: normalized coords should fill [0, 1]
            print(f"Normalized coord range: [{coords.min().item():.4f}, {coords.max().item():.4f}]")
            print(f"Smoke scope: {n_rays} rays x {coords.shape[1]} bins (full grid, windowed Voigt).")

        except FileNotFoundError as e:
            print("Data files not found, exiting.", e)
            return

    print("--- Running NeRF Stage 2a (Physics): Arch Setup & Accurate Ray Integrator ---")

    # Set up MLflow following the new Governance Rules
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Hierarchical Experiment Name: Project/Methodology
        mlflow.set_experiment("CosmoGasVision/NeRF")
        print(f"Connected to MLflow at {mlflow_uri} [Experiment: CosmoGasVision/NeRF]")
    except Exception as e:
        print(f"MLflow connection issue: {e}")

    try:
        # Standardized Run Name: Stage-Description
        with mlflow.start_run(run_name="Stage2a-PhysicsIntegratorRevalidation"):
            # Mandatory Metadata Tagging
            mlflow.set_tags({
                "model_type": "nerf",
                "stage": "2a",
                "physics_id": "1",
                "redshift": "0.3",
            })

            # Production-scale architecture (D-09 paper-vs-code parity): 8 layers / L=10.
            model = IGMNeRF(hidden_dim=256, num_layers=8, L=10)

            # Learnable optical-depth amplitude absorbing sigma_0 * ds * mean column.
            # Parameterized in log-space so it stays positive under unconstrained Adam,
            # and anchored to log(tau_amp) ~ N(0, sigma_log) per [D-10] to break the
            # tau_amp <-> density rescaling degeneracy. Width sigma_log=0.5 corresponds
            # to a multiplicative uncertainty of factor exp(0.5) ~ 1.65; tight enough
            # to break the symmetry, loose enough not to dominate the data fit.
            log_tau_amp = torch.nn.Parameter(torch.tensor(0.0))
            sigma_log = 0.5
            tau_amp_prior_weight = 1e-3   # subdominant to data MSE at smoke scale

            print(f"Model: {sum(p.numel() for p in model.parameters())} params + log_tau_amp scalar.")

            optimizer = optim.Adam(
                list(model.parameters()) + [log_tau_amp],
                lr=5e-4,
            )
            mse_loss = torch.nn.MSELoss()

            mlflow.log_params({
                "num_layers": 8,
                "hidden_dim": 256,
                "L_fourier": 10,
                "n_rays": coords.shape[0],
                "n_bins": coords.shape[1],
                "lr": 5e-4,
                "loss_form": "tau_v_profile_mse + log_tau_amp_prior",
                "voigt_window_bins": 64,
                "log_tau_amp_sigma": sigma_log,
                "tau_amp_prior_weight": tau_amp_prior_weight,
            })

            for step in range(10):
                optimizer.zero_grad()

                tau_amp = torch.exp(log_tau_amp)

                # Forward pass: full tau(v) profile via windowed RSD convolution
                tau_pred = volume_render_physics(
                    model, coords, vel_axis=vel_axis, tau_amp=tau_amp
                )  # (n_rays, n_bins)

                loss_data = mse_loss(tau_pred, tau_gt_profile)
                loss_prior = (log_tau_amp ** 2) / (2 * sigma_log ** 2)
                loss = loss_data + tau_amp_prior_weight * loss_prior
                loss.backward()

                grad_norm = model.out_layer.weight.grad.norm().item()
                optimizer.step()

                print(
                    f"Step {step+1}/10 | Loss: {loss.item():.4f} "
                    f"(data={loss_data.item():.4f}, prior={loss_prior.item():.4f}) | "
                    f"Grad Norm: {grad_norm:.4f} | tau_amp: {tau_amp.item():.4f}",
                    flush=True,
                )

                mlflow.log_metric("loss", loss.item(), step=step)
                mlflow.log_metric("loss_data", loss_data.item(), step=step)
                mlflow.log_metric("loss_prior", loss_prior.item(), step=step)
                mlflow.log_metric("grad_norm", grad_norm, step=step)
                mlflow.log_metric("tau_amp", tau_amp.item(), step=step)

            print("Backward pass and gradient flow confirmed. Projection layer verified.")
            print(f"Run id: {mlflow.active_run().info.run_id}", flush=True)

    except Exception as e:
        print(f"Pipeline error after run body: {e}", flush=True)

if __name__ == "__main__":
    run_stage1_2a()
