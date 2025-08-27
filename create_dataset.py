# build_datasets.py
# Unified CAMB-only dataset pipeline + cleaning + merge + PCA + validation
# Uses: clean_dataset.clean_dataset / merge_dataset / pca_component

import os, sys, textwrap
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
from smt.sampling_methods import LHS

import camb

from clean_dataset import clean_dataset, merge_dataset, pca_component  # :contentReference[oaicite:2]{index=2}

# ──────────────────────────────────────────────────────────────────────
# Constants / defaults
# ──────────────────────────────────────────────────────────────────────

T_CMB      = 2.7255
OMEGA_B0   = 0.02237
OMEGA_CDM0 = 0.12000
OMEGA_G0   = 2.471e-5
OMEGA_UR0  = 1.709e-5

# Base cosmology (overridden per-sample for the ω's only)
BASE = dict(
    omega_b = OMEGA_B0,
    omega_c = OMEGA_CDM0,
    omega_g = OMEGA_G0,
    omega_ur= OMEGA_UR0,
    omega_k = 0.0,
    h       = 0.6736,
    H0      = 67.36,
    YHe     = 0.2471,
    tau     = 0.0544,
    As      = 2.1e-9,
    n_s     = 0.9649,
)

# Planck validation point (exact values you gave)
PLANCK_VAL = dict(
    omega_b = 0.02237,
    omega_c = 0.12000,
    omega_g = 2.471e-5,
    omega_ur= 1.709e-5,
    omega_k = 0.0,
    h       = 0.6736,
    H0      = 67.36,
    YHe     = 0.2471,
    tau     = 0.0544,
    As      = 2.1e-9,
    n_s     = 0.9649,
)

# ──────────────────────────────────────────────────────────────────────
# Helpers (same conventions as your scripts)  :contentReference[oaicite:3]{index=3}
# ──────────────────────────────────────────────────────────────────────

def tcmb_omega_g(omega_g, omega_g_ref=OMEGA_G0, T_ref=T_CMB):
    return T_ref * (omega_g / omega_g_ref)**0.25

def neff_omega_ur(omega_ur, t_cmb, omega_ur_ref=OMEGA_UR0, T_ref=T_CMB):
    scale = (t_cmb / T_ref)**4
    return 3.046 * omega_ur / (omega_ur_ref * scale)

def camb_dl(params, lmax):
    # params contains ω's, H0/h, tau, YHe, As, ns, etc.
    t_cmb = tcmb_omega_g(params["omega_g"])
    Neff  = neff_omega_ur(params["omega_ur"], t_cmb)

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"],
                       ombh2=params["omega_b"],
                       omch2=params["omega_c"],
                       omk=params["omega_k"],
                       YHe=params["YHe"],
                       tau=params["tau"],
                       TCMB=t_cmb)
    pars.Neff = Neff
    pars.num_massive_neutrinos = 0
    pars.InitPower.set_params(As=params["As"], ns=params["n_s"])
    pars.set_accuracy(AccuracyBoost=1, lAccuracyBoost=1, lSampleBoost=1)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    # "unlensed_scalar" → columns: TT, EE, BB, TE in µK^2 when CMB_unit="muK"
    Dl_camb = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=lmax)["unlensed_scalar"][:, 0]
    return Dl_camb[2:lmax + 1]  # D_ell for ℓ=2..lmax

# ──────────────────────────────────────────────────────────────────────
# Sampling (Planck core, full box, and edge-band near ΩΛ≈0)
# ──────────────────────────────────────────────────────────────────────

def sample_box_lhs(n, ranges, seed=42):
    limits = np.array([ranges['ob'], ranges['oc'], ranges['og'], ranges['our']], float)
    lhs = LHS(xlimits=limits, random_state=seed)
    return lhs(n)

def flat_mask(X, h2):
    return (X.sum(axis=1) <= h2 + 1e-15)  # allow tiny FP slack

def draw_core(n, ranges, h2, seed=1001):
    """Planck-like box with flatness filter; oversample then trim to n."""
    X = sample_box_lhs(int(max(1.2*n, n+2000)), ranges, seed=seed)
    X = X[flat_mask(X, h2)]
    if len(X) < n:
        Y = sample_box_lhs((n-len(X))*2, ranges, seed=seed+1)
        Y = Y[flat_mask(Y, h2)]
        X = np.vstack([X, Y])
    return X[:n]

def draw_full(n, ranges, h2, seed=2001):
    """Full coverage box with flatness filter."""
    X = sample_box_lhs(int(max(1.2*n, n+2000)), ranges, seed=seed)
    X = X[flat_mask(X, h2)]
    if len(X) < n:
        Y = sample_box_lhs((n-len(X))*2, ranges, seed=seed+1)
        Y = Y[flat_mask(Y, h2)]
        X = np.vstack([X, Y])
    return X[:n]

def draw_edgeband(n, ranges, h2, seed=3001, quantile=0.10):
    """Prefer samples with the *lowest* ΩΛ h² = h² - (ωb+ωc+ωg+ωur)."""
    rng = np.random.RandomState(seed)
    out = []
    need = n
    while sum(len(a) for a in out) < n:
        X = sample_box_lhs(int(max(need*6, 3000)), ranges, seed=rng.randint(1, 10_000))
        lam = h2 - X.sum(axis=1)
        mask = lam >= 0.0
        X, lam = X[mask], lam[mask]
        if len(X) == 0:
            continue
        thr = np.quantile(lam, quantile)  # near boundary
        out.append(X[lam <= thr])
        need = n - sum(len(a) for a in out)
    return np.vstack(out)[:n]

# ──────────────────────────────────────────────────────────────────────
# NEW: Adaptive writer that tops up until we hit n_target exactly
# ──────────────────────────────────────────────────────────────────────

def make_dataset_exact(out_file, n_target, proposal_fn, base_params, lmax, prepend=None):
    """
    Create exactly n_target rows. Any time a candidate is skipped (ΩΛ<0 or CAMB error),
    we immediately draw a replacement from the same proposal_fn distribution.
    If 'prepend' is given, those rows (array-like of [ob,oc,og,our]) are attempted first.
    """
    ell = np.arange(2, lmax+1)
    h2 = base_params["h"]**2

    n_camb = 0
    n_skip_lambda = 0
    n_skip_error  = 0
    n_proposed    = 0

    with h5py.File(out_file, "w") as f:
        d_omega_b      = f.create_dataset("omega_b",      (0,), maxshape=(None,), dtype="f4")
        d_omega_cdm    = f.create_dataset("omega_c",      (0,), maxshape=(None,), dtype="f4")
        d_omega_g      = f.create_dataset("omega_g",      (0,), maxshape=(None,), dtype="f4")
        d_omega_ur     = f.create_dataset("omega_ur",     (0,), maxshape=(None,), dtype="f4")
        d_omega_lambda = f.create_dataset("omega_lambda", (0,), maxshape=(None,), dtype="f4")
        d_log_d_ell    = f.create_dataset("d_ell",        (0, len(ell)), maxshape=(None, len(ell)), dtype="f4")
        f.create_dataset("ell", data=ell)

        kept = 0
        buf  = np.empty((0,4), dtype=float)

        # Helper to append one successful row
        def append_row(omega_b, omega_c, omega_g, omega_ur, D_ell, omega_lambda):
            nonlocal kept
            k = kept
            for ds in (d_omega_b, d_omega_cdm, d_omega_g, d_omega_ur, d_omega_lambda, d_log_d_ell):
                ds.resize((k+1,) + ds.shape[1:])
            d_omega_b[k]      = omega_b
            d_omega_cdm[k]    = omega_c
            d_omega_g[k]      = omega_g
            d_omega_ur[k]     = omega_ur
            d_omega_lambda[k] = omega_lambda
            d_log_d_ell[k]    = np.log(np.clip(D_ell, 1e-30, None)).astype("f4")
            kept += 1

        # Try prepend rows (e.g., exact Planck point first)
        if prepend is not None and len(prepend):
            for (omega_b, omega_c, omega_g, omega_ur) in prepend:
                if kept >= n_target:
                    break
                omega_lambda = h2 - (omega_b + omega_c + omega_g + omega_ur)
                n_proposed += 1
                if omega_lambda < 0.0:
                    n_skip_lambda += 1
                    continue
                P = base_params.copy()
                P.update(dict(omega_b=float(omega_b), omega_c=float(omega_c),
                              omega_g=float(omega_g), omega_ur=float(omega_ur)))
                try:
                    D_ell = camb_dl(P, lmax)
                    n_camb += 1
                    append_row(omega_b, omega_c, omega_g, omega_ur, D_ell, omega_lambda)
                except Exception:
                    n_skip_error += 1
                    continue

        # Main loop: keep topping up until we hit n_target
        pbar = tqdm(total=n_target, initial=kept, desc="models", unit="mdl")
        BATCH = 4096
        while kept < n_target:
            if len(buf) == 0:
                # Refill buffer with fresh proposals from the same distribution
                need = max(BATCH, n_target - kept)
                buf = np.asarray(proposal_fn(need), dtype=float)

            # Pop one candidate
            omega_b, omega_c, omega_g, omega_ur = buf[0]
            buf = buf[1:]
            n_proposed += 1

            omega_lambda = h2 - (omega_b + omega_c + omega_g + omega_ur)
            if omega_lambda < 0.0:
                n_skip_lambda += 1
                continue

            P = base_params.copy()
            P.update(dict(omega_b=float(omega_b), omega_c=float(omega_c),
                          omega_g=float(omega_g), omega_ur=float(omega_ur)))
            try:
                D_ell = camb_dl(P, lmax)
                n_camb += 1
                append_row(omega_b, omega_c, omega_g, omega_ur, D_ell, omega_lambda)
                pbar.update(1)
            except Exception:
                n_skip_error += 1
                # immediately continue; the while loop will draw replacements as needed
                continue

        pbar.close()

    print(textwrap.dedent(f"""
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃        dataset finished      ┃
    ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
    ┃ target kept (CAMB) : {n_target:9d}
    ┃ proposals tried    : {n_proposed:9d}
    ┃ lambda_skipped     : {n_skip_lambda:9d}
    ┃ error_skipped      : {n_skip_error:9d}
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    → file: {out_file}
    """).strip())

# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    out_dir="dataset",
    lmax=1000,
    # ranges: full coverage + narrower Planck-like core
    FULL=dict(ob=(0.010, 0.030), oc=(0.090, 0.160), og=(2.40e-5, 2.60e-5), our=(1.20e-5, 1.90e-5)),
    CORE=dict(ob=(0.020, 0.025), oc=(0.105, 0.135), og=(2.45e-5, 2.50e-5), our=(1.45e-5, 1.85e-5)),
    N_CORE=30000, N_FULL=15000, N_EDGE=15000,  # heavier Planck sampling
    edge_quantile=0.10,
    seed=1234,
    build_validation=True, N_VAL=2000
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    h2 = BASE["h"]**2
    rng = np.random.RandomState(seed)

    # 1) Proposal samplers (fresh seed every refill to avoid repeats)
    def sampler_core(m): return draw_core(m, CORE, h2, seed=rng.randint(1, 1_000_000_000))
    def sampler_full(m): return draw_full(m, FULL, h2, seed=rng.randint(1, 1_000_000_000))
    def sampler_edge(m): return draw_edgeband(m, FULL, h2, seed=rng.randint(1, 1_000_000_000), quantile=edge_quantile)

    # 2) Write raw datasets (exact counts guaranteed)
    core_raw = f"{out_dir}/core_{N_CORE}_lmax{lmax}.h5"
    full_raw = f"{out_dir}/full_{N_FULL}_lmax{lmax}.h5"
    edge_raw = f"{out_dir}/edge_{N_EDGE}_lmax{lmax}.h5"

    make_dataset_exact(core_raw, N_CORE, sampler_core, BASE, lmax)
    make_dataset_exact(full_raw, N_FULL, sampler_full, BASE, lmax)
    make_dataset_exact(edge_raw, N_EDGE, sampler_edge, BASE, lmax)

    # 3) Clean each (truncate to lmax-1 multipoles, drop zero rows)
    ell_keep = lmax - 1
    core_clean = f"{out_dir}/core_clean.h5"
    full_clean = f"{out_dir}/full_clean.h5"
    edge_clean = f"{out_dir}/edge_clean.h5"
    param_names = ['omega_b', 'omega_c', 'omega_g', 'omega_ur']  # :contentReference[oaicite:5]{index=5}

    clean_dataset(core_raw, core_clean, param_names, ell_keep=ell_keep)  # :contentReference[oaicite:6]{index=6}
    clean_dataset(full_raw, full_clean, param_names, ell_keep=ell_keep)  # :contentReference[oaicite:7]{index=7}
    clean_dataset(edge_raw, edge_clean, param_names, ell_keep=ell_keep)  # :contentReference[oaicite:8]{index=8}

    # 4) Merge (pairwise using your helper)
    merged12 = f"{out_dir}/merged_tmp.h5"
    merged   = f"{out_dir}/merged_final.h5"
    merge_dataset(core_clean, full_clean, merged12)                 # :contentReference[oaicite:9]{index=9}
    merge_dataset(merged12,  edge_clean, merged)                    # :contentReference[oaicite:10]{index=10}
    os.remove(merged12)

    # 5) PCA packs (20 and 50 components)
    pca20 = f"{out_dir}/merged_20_pca.h5"
    pca50 = f"{out_dir}/merged_50_pca.h5"
    pca_component(merged, pca20, 20, param_names)                   # :contentReference[oaicite:11]{index=11}
    pca_component(merged, pca50, 50, param_names)                   # :contentReference[oaicite:12]{index=12}

    # 6) Validation set (cleaned, includes exact Planck point)
    if build_validation:
        val_raw   = f"{out_dir}/val_{N_VAL}_lmax{lmax}.h5"
        val_clean = f"{out_dir}/val_{N_VAL}_clean.h5"

        # Prepend the exact Planck point, then top up from FULL until we have N_VAL
        planck_row = np.array([[PLANCK_VAL["omega_b"], PLANCK_VAL["omega_c"],
                                PLANCK_VAL["omega_g"], PLANCK_VAL["omega_ur"]]], dtype=float)
        make_dataset_exact(
            val_raw, N_VAL, sampler_full, PLANCK_VAL, lmax,
            prepend=planck_row
        )
        clean_dataset(val_raw, val_clean, param_names, ell_keep=ell_keep)  # :contentReference[oaicite:13]{index=13}

    print("\n✓ Pipeline complete.")
    print(f"   Merged dataset : {merged}")
    print(f"   PCA(20)        : {pca20}")
    print(f"   PCA(50)        : {pca50}")
    if build_validation:
        print(f"   Validation     : {val_clean}")

# ──────────────────────────────────────────────────────────────────────

def extend_with_small_batch(
    out_dir="dataset",
    lmax=1000,
    # defaults mirror your earlier ranges (tweak as you like)
    FULL=dict(ob=(0.001, 0.40), oc=(0.001, 0.40), og=(2.30e-5, 2.60e-5), our=(1.30e-5, 2.00e-5)),
    CORE=dict(ob=(0.020, 0.025), oc=(0.105, 0.135), og=(2.45e-5, 2.50e-5), our=(1.65e-5, 1.85e-5)),
    N_CORE=2500, N_FULL=30000, N_EDGE=2500,
    edge_quantile=0.10,
    seed=5678,
    existing_merged_name="merged_final.h5",  # expected to live in out_dir
):
    """
    Build a small add-on dataset (core=2.5k, full=30k, edge=2.5k), clean it,
    merge with existing merged_final.h5, and produce PCA packs:
      - merged_complete_20_pca.h5
      - merged_complete_50_pca.h5

    NOTE: The existing merged file must have the same lmax (multipole length)
    as the new data (ell_keep = lmax-1), or merging will fail.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    param_names = ['omega_b', 'omega_c', 'omega_g', 'omega_ur']
    ell_keep = lmax - 1

    # Samplers (fresh seed per refill)
    h2 = BASE["h"]**2
    rng = np.random.RandomState(seed)
    def sampler_core(m): return draw_core(m, CORE, h2, seed=rng.randint(1, 1_000_000_000))
    def sampler_full(m): return draw_full(m, FULL, h2, seed=rng.randint(1, 1_000_000_000))
    def sampler_edge(m): return draw_edgeband(m, FULL, h2, seed=rng.randint(1, 1_000_000_000), quantile=edge_quantile)

    # Raw add-on datasets
    core_raw = f"{out_dir}/addon_core_{N_CORE}_lmax{lmax}.h5"
    full_raw = f"{out_dir}/addon_full_{N_FULL}_lmax{lmax}.h5"
    edge_raw = f"{out_dir}/addon_edge_{N_EDGE}_lmax{lmax}.h5"

    make_dataset_exact(core_raw, N_CORE, sampler_core, BASE, lmax)
    make_dataset_exact(full_raw, N_FULL, sampler_full, BASE, lmax)
    make_dataset_exact(edge_raw, N_EDGE, sampler_edge, BASE, lmax)

    # Clean each
    core_clean = f"{out_dir}/addon_core_clean.h5"
    full_clean = f"{out_dir}/addon_full_clean.h5"
    edge_clean = f"{out_dir}/addon_edge_clean.h5"
    clean_dataset(core_raw, core_clean, param_names, ell_keep=ell_keep)
    clean_dataset(full_raw, full_clean, param_names, ell_keep=ell_keep)
    clean_dataset(edge_raw, edge_clean, param_names, ell_keep=ell_keep)

    # Merge the add-on set into a single cleaned file
    addon_tmp   = f"{out_dir}/addon_merged_tmp.h5"
    addon_clean = f"{out_dir}/addon_merged_clean.h5"
    merge_dataset(core_clean, full_clean, addon_tmp)
    merge_dataset(addon_tmp,  edge_clean, addon_clean)
    os.remove(addon_tmp)

    # Merge with existing merged_final.h5
    existing_merged = f"{out_dir}/{existing_merged_name}"
    if not os.path.exists(existing_merged):
        raise FileNotFoundError(f"Expected existing merged file at: {existing_merged}")

    merged_complete = f"{out_dir}/merged_complete.h5"
    merge_dataset(existing_merged, addon_clean, merged_complete)

    # PCA packs from the completed merge
    pca20 = f"{out_dir}/merged_complete_20_pca.h5"
    pca50 = f"{out_dir}/merged_complete_50_pca.h5"
    pca_component(merged_complete, pca20, 20, param_names)
    pca_component(merged_complete, pca50, 50, param_names)

    print("\n✓ Add-on batch merged.")
    print(f"   Existing merged : {existing_merged}")
    print(f"   Add-on merged   : {addon_clean}")
    print(f"   Complete merge  : {merged_complete}")
    print(f"   PCA(20)         : {pca20}")
    print(f"   PCA(50)         : {pca50}")


if __name__ == "__main__":
    extend_with_small_batch(out_dir="dataset", lmax=1000)



"""
if __name__ == "__main__":
    # Example run with defaults. Adjust ranges/counts here ↓↓↓
    run_pipeline(
        out_dir="dataset",
        lmax=1000,
        FULL=dict(
            ob=(0.001, 0.40),
            oc=(0.001, 0.40),
            og=(2.30e-5, 2.60e-5),
            our=(1.30e-5, 2.00e-5),
        ),
        CORE=dict(
            ob=(0.020, 0.025),
            oc=(0.105, 0.135),
            og=(2.45e-5, 2.50e-5),
            our=(1.65e-5, 1.85e-5),
        ),
        N_CORE=25000, N_FULL=70000, N_EDGE=25000,
        edge_quantile=0.10,
        seed=1234,
        build_validation=True, N_VAL=2000,
   )
"""