import numpy as np
from classy import Class
import csv
from tqdm import tqdm


def generate_flat_cmb_dataset_streamed(omega_b_range=(0.01, 0.07),
                                       omega_cdm_range=(0.1, 0.6),
                                       num_points=500,
                                       l_max=800,
                                       output_file="cmb_flat_dataset.csv"):
    omega_b_vals = np.linspace(*omega_b_range, num_points)
    omega_cdm_vals = np.linspace(*omega_cdm_range, num_points)

    omega_b_grid, omega_cdm_grid = np.meshgrid(omega_b_vals, omega_cdm_vals)
    omega_b_flat = omega_b_grid.flatten()
    omega_cdm_flat = omega_cdm_grid.flatten()

    mask = (omega_b_flat + omega_cdm_flat) <= 1.0
    omega_b_valid = omega_b_flat[mask]
    omega_cdm_valid = omega_cdm_flat[mask]
    omega_lambda_valid = 1.0 - (omega_b_valid + omega_cdm_valid)

    T_CMB_uK = 2.7255e6  # μK
    ells = np.arange(2, l_max + 1)
    ell_factor = ells * (ells + 1) / (2 * np.pi)

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["omega_b", "omega_cdm", "omega_lambda"] + [f"D_ell_{l}" for l in ells]
        writer.writerow(header)

        for omega_b, omega_cdm, omega_lambda in tqdm(
            zip(omega_b_valid, omega_cdm_valid, omega_lambda_valid),
            total=len(omega_b_valid),
            desc="Computing D_ell^TT"
        ):
            try:
                cosmo = Class()
                cosmo.set({
                    "omega_b": omega_b,
                    "omega_cdm": omega_cdm,
                    "h": 0.67810,
                    "A_s": 2.1e-9,
                    "n_s": 0.966,
                    "tau_reio": 0.054,
                    "Omega_Lambda": omega_lambda,
                    "output": "tCl",
                    "lensing": "no",
                    "l_max_scalars": l_max,
                })
                cosmo.compute()
                cls = cosmo.raw_cl(l_max)
                cl_tt = cls['tt'][2:l_max + 1]
                d_ell = ell_factor * cl_tt * T_CMB_uK**2
                row = [omega_b, omega_cdm, omega_lambda] + d_ell.tolist()
                writer.writerow(row)
            except Exception as e:
                print(f"Skipped point ω_b={omega_b:.4f}, ω_cdm={omega_cdm:.4f}: {e}")
            finally:
                try:
                    cosmo.struct_cleanup()
                    cosmo.empty()
                    del cosmo  # ✅ ensure full memory release
                except:
                    pass


# Run the function
generate_flat_cmb_dataset_streamed()

