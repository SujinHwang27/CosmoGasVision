import os
import numpy as np
from typing import Tuple, Dict
import torch

class SherwoodLoader:
    """
    Loader for Sherwood simulation binary data.
    Ported and enhanced from Sherwood/src/utils.py
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        # Mapping physics IDs to directory names
        self.physics_models = {
            1: 'Physics1_nofeedback',
            2: 'Physics2_stellarwind',
            3: 'Physics3_windAGN',
            4: 'Physics4_windstrongAGN'
        }

    def load_sightlines(self, physics_id: int, redshift: float, nspec: int = 16384) -> Dict[str, np.ndarray]:
        """
        Loads sightline data from binary files.
        """
        if physics_id not in self.physics_models:
            raise ValueError(f"Invalid physics_id {physics_id}. Must be 1-4.")
        
        sim_name = self.physics_models[physics_id]
        base_path = os.path.join(self.data_root, sim_name)
        
        # File names as per Sherwood naming convention
        los_file = os.path.join(base_path, f"los2048_n{nspec}_z{redshift:.3f}.dat")
        tau_file = os.path.join(base_path, f"tauH1_2048_n{nspec}_z{redshift:.3f}.dat")
        
        if not os.path.exists(los_file):
            raise FileNotFoundError(f"LOS file not found: {los_file}")
        if not os.path.exists(tau_file):
            raise FileNotFoundError(f"Tau file not found: {tau_file}")

        # Read LOS file
        with open(los_file, "rb") as f:
            # Header
            header = {
                'redshift': np.fromfile(f, dtype=np.double, count=1)[0],
                'omega_m': np.fromfile(f, dtype=np.double, count=1)[0],
                'omega_l': np.fromfile(f, dtype=np.double, count=1)[0],
                'omega_b': np.fromfile(f, dtype=np.double, count=1)[0],
                'h100': np.fromfile(f, dtype=np.double, count=1)[0],
                'box_kpc_h': np.fromfile(f, dtype=np.double, count=1)[0],
                'Xh': np.fromfile(f, dtype=np.double, count=1)[0],
                'nbins': np.fromfile(f, dtype=np.int32, count=1)[0],
                'num_los': np.fromfile(f, dtype=np.int32, count=1)[0]
            }
            
            nbins = header['nbins']
            num_los = header['num_los']
            
            # Coordinates
            iaxis = np.fromfile(f, dtype=np.int32, count=num_los)      # 1=x, 2=y, 3=z
            xaxis = np.fromfile(f, dtype=np.double, count=num_los)     # kpc/h
            yaxis = np.fromfile(f, dtype=np.double, count=num_los)     # kpc/h
            zaxis = np.fromfile(f, dtype=np.double, count=num_los)     # kpc/h
            
            # Axes
            pos_axis = np.fromfile(f, dtype=np.double, count=nbins)  # kpc/h
            vel_axis = np.fromfile(f, dtype=np.double, count=nbins)  # km/s
            
            # Physical fields
            density = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
            h1_frac = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
            temp = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))
            v_pec = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))

        # Read Tau file
        with open(tau_file, "rb") as f:
            tau_h1 = np.fromfile(f, dtype=np.double, count=nbins * num_los).reshape((num_los, nbins))

        # Sanitize NaNs which may appear in early/strong AGN models
        density = np.nan_to_num(density, nan=0.0)
        h1_frac = np.nan_to_num(h1_frac, nan=0.0)
        temp = np.nan_to_num(temp, nan=1e4)  # Default warm IGM temp
        v_pec = np.nan_to_num(v_pec, nan=0.0)
        tau_h1 = np.nan_to_num(tau_h1, nan=0.0)

        # Sanity Checks
        self._validate_data(density, h1_frac, temp, tau_h1)

        return {
            'header': header,
            'iaxis': iaxis,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'zaxis': zaxis,
            'pos_axis': pos_axis,
            'vel_axis': vel_axis,
            'density': density,
            'h1_frac': h1_frac,
            'temp': temp,
            'v_pec': v_pec,
            'tau_h1': tau_h1
        }

    def _validate_data(self, density: np.ndarray, h1_frac: np.ndarray, temp: np.ndarray, tau: np.ndarray):
        """
        Validates astrophysical data ranges.
        """
        assert (density >= 0).all(), f"Negative density found: {density.min()}"
        assert (h1_frac >= 0).all() and (h1_frac <= 1.0001).all(), f"Invalid H1 fraction: {h1_frac.min()} - {h1_frac.max()}"
        assert (temp > 0).all(), f"Non-positive temperature found: {temp.min()}"
        assert (tau >= 0).all(), f"Negative optical depth found: {tau.min()}"
        print("Data sanity check passed.")

    def get_world_coordinates(self, data: Dict) -> np.ndarray:
        """
        Converts los indices and pos_axis into full 3D coordinates for every bin.
        Returns array of shape (num_los, nbins, 3)
        """
        num_los = data['header']['num_los']
        nbins = data['header']['nbins']
        
        coords = np.zeros((num_los, nbins, 3))
        
        # iaxis: 1=x, 2=y, 3=z. This is the axis ALONG which the sightline runs.
        # xaxis, yaxis, zaxis: the coordinates of the sightline in the other two axes.
        for i in range(num_los):
            axis = data['iaxis'][i]
            x, y, z = data['xaxis'][i], data['yaxis'][i], data['zaxis'][i]
            
            if axis == 1: # Runs along x
                coords[i, :, 0] = data['pos_axis']
                coords[i, :, 1] = y
                coords[i, :, 2] = z
            elif axis == 2: # Runs along y
                coords[i, :, 0] = x
                coords[i, :, 1] = data['pos_axis']
                coords[i, :, 2] = z
            elif axis == 3: # Runs along z
                coords[i, :, 0] = x
                coords[i, :, 1] = y
                coords[i, :, 2] = data['pos_axis']
                
        return coords
