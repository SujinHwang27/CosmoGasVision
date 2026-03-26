import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.signal import find_peaks
from src.data.loader import SherwoodLoader

def extract_seeds(physics_id: int, redshift: float, data_root: str, threshold: float = 10.0):
    """
    Stage 1: Seed Extraction.
    Find peaks in density and H1 fraction to initialize Gaussian centers.
    """
    loader = SherwoodLoader(data_root)
    print(f"Loading data for Physics {physics_id}, Redshift {redshift:.3f}...")
    data = loader.load_sightlines(physics_id, redshift)
    
    num_los = data['header']['num_los']
    nbins = data['header']['nbins']
    
    # Store seeds
    # Format: (x, y, z, density, h1_frac, temp, v_pec, tau)
    all_seeds = []
    
    print(f"Extracting peaks with threshold {threshold} from {num_los} sightlines...")
    
    # Optimized extraction
    for i in tqdm(range(num_los)):
        density_profile = data['density'][i]
        
        # Peak finding in 1D density profile
        # rho/<rho>
        peaks, _ = find_peaks(density_profile, height=threshold)
        
        iaxis = data['iaxis'][i]
        xaxis, yaxis, zaxis = data['xaxis'][i], data['yaxis'][i], data['zaxis'][i]
        pos_axis = data['pos_axis']
        
        for p_idx in peaks:
            pos = pos_axis[p_idx]
            
            if iaxis == 1: # Runs along x
                coord = [pos, yaxis, zaxis]
            elif iaxis == 2: # Runs along y
                coord = [xaxis, pos, zaxis]
            else: # Runs along z
                coord = [xaxis, yaxis, pos]
            
            # Physical fields at peak
            seed = [
                coord[0], coord[1], coord[2],
                data['density'][i, p_idx],
                data['h1_frac'][i, p_idx],
                data['temp'][i, p_idx],
                data['v_pec'][i, p_idx],
                data['tau_h1'][i, p_idx]
            ]
            all_seeds.append(seed)
            
    return np.array(all_seeds), data['header']

if __name__ == "__main__":
    print("STARTING MAIN!")
    parser = argparse.ArgumentParser(description="Stage 1: Seed Extraction")
    # RATIONALE: Switching from Pilot P3 to P2 as the primary baseline for development. 
    # Physics 2 (Stellar winds) provides the standard baseline for subsequent stages.
    parser.add_argument("--physics", type=int, default=2, help="Physics model index (1-4)")
    parser.add_argument("--redshift", type=float, default=0.3, help="Redshift (0.1, 0.3, 2.2, 2.4)")
    # RATIONALE: Threshold 50.0 is high to ensure seeds are centered on well-defined physical structures.
    parser.add_argument("--threshold", type=float, default=50.0, help="Density peak threshold (rho/mean)")
    parser.add_argument("--data_root", type=str, default="/home/sujin/CosmoGasVision/Sherwood", help="Root data directory")
    args = parser.parse_args()
    print("ARGS PARSED!")

    try:
        from dotenv import load_dotenv
        load_dotenv()
        import mlflow
        MLFLOW_AVAILABLE = True
    except ImportError:
        print("Warning: mlflow not installed. Experiment tracking will be disabled.")
        MLFLOW_AVAILABLE = False

    use_mlflow = MLFLOW_AVAILABLE
    
    if use_mlflow:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("IGM-Tomography-Stage1")
        run_name = f"Seeds_P{args.physics}_z{args.redshift:.3f}"
        context_manager = mlflow.start_run(run_name=run_name)
    else:
        from contextlib import nullcontext
        context_manager = nullcontext()

    with context_manager as run:
        if use_mlflow:
            mlflow.log_params({
                "physics": args.physics,
                "redshift": args.redshift,
                "threshold": args.threshold
            })
        
        seed_array, header = extract_seeds(args.physics, args.redshift, args.data_root, args.threshold)
        
        num_seeds = len(seed_array)
        print(f"Extracted {num_seeds} seed points.")
        
        if use_mlflow:
            mlflow.log_metric("num_seeds", num_seeds)
        
        output_dir = "experiments/3dgs_baseline/stage1_seeds"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"seeds_P{args.physics}_z{args.redshift:.3f}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, seed_array)
        
        if use_mlflow:
            mlflow.log_artifact(filepath)
        print(f"Seeds saved to {filepath}")
