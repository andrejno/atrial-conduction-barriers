"""Run the verification, graph-limit, and conduction-calibration studies.

All outputs are deterministic.  The script writes machine-readable CSV/NPZ
files beneath ``data`` and publication figures beneath ``figures``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import (
    EPParameters,
    PhaseParameters,
    SpectralGrid,
    activation_cv,
    apd90,
    classifier_arctan,
    conductivity_factor,
    solve_monodomain,
    solve_phase,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RECON_KX = 1.0
RECON_KY = 0.6
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def reconstruction_grid(n: int) -> SpectralGrid:
    """Periodic reconstruction grid with the reported diagonal anisotropy."""
    return SpectralGrid(n, n, kx_weight=RECON_KX, ky_weight=RECON_KY)


def exact_primitive(s: np.ndarray) -> np.ndarray:
    return (
        35.0 / (128.0 * np.pi) * np.sin(2.0 * np.pi * s)
        + 7.0 / (128.0 * np.pi) * np.sin(6.0 * np.pi * s)
        + 7.0 / (640.0 * np.pi) * np.sin(10.0 * np.pi * s)
        + 1.0 / (896.0 * np.pi) * np.sin(14.0 * np.pi * s)
    )


def phase_exact_fields(grid: SpectralGrid, t: float):
    Fx = exact_primitive(grid.X)
    Fy = exact_primitive(grid.Y)
    u = 0.10 + 0.05 * np.exp(-20.0 * t) + 7.5 * np.exp(-15.0 * t) * (Fx + 0.7 * Fy)
    ut = -1.0 * np.exp(-20.0 * t) - 112.5 * np.exp(-15.0 * t) * (Fx + 0.7 * Fy)
    uref = 0.10 + 5.0 * Fx - 6.0 * Fy
    lam = 25.0 * (1.0 + 0.2 * np.cos(2.0 * np.pi * grid.X) * np.sin(2.0 * np.pi * grid.Y))
    return u, ut, uref, lam


def phase_exact_forcing(grid: SpectralGrid, t: float, mu: float, nu: float, kappa: float) -> np.ndarray:
    # The manufactured residual is evaluated on a threefold periodic grid and
    # then restricted to the collocated nodes of ``grid``.  This reduces
    # nonlinear aliasing in the prescribed forcing; it is not an independent
    # solution calculation.
    fine = SpectralGrid(3 * grid.nx, 3 * grid.ny, grid.lx, grid.ly, grid.kx_weight, grid.ky_weight)
    u, ut, uref, lam = phase_exact_fields(fine, t)
    Lu = fine.anisotropic_laplacian(u)
    chemical = u**3 - u - mu * Lu - nu * classifier_arctan(Lu, kappa) * fine.grad_norm_k(u)
    rhs = fine.anisotropic_laplacian(chemical) + lam * (uref - u)
    f = ut - rhs
    return f[::3, ::3]


def _phase_solution_error(
    n: int,
    dt: float,
    T: float,
    mu: float,
    nu: float,
    kappa: float,
) -> tuple[dict[str, float], np.ndarray, list[np.ndarray]]:
    """Solve the prescribed phase problem and return all three solution errors."""
    grid = reconstruction_grid(n)
    u0, _, uref, lam = phase_exact_fields(grid, 0.0)
    par = PhaseParameters(
        mu=mu,
        nu=nu,
        dt=dt,
        classifier="smooth",
        N=kappa,
        admm_tol=1.0e-8,
        admm_maxiter=8000,
    )
    nsteps = round(T / dt)
    if not np.isclose(nsteps * dt, T, rtol=0.0, atol=1.0e-15):
        raise ValueError("T must be an integer multiple of dt")
    state, history = solve_phase(
        grid,
        u0,
        uref,
        lam,
        par,
        nsteps,
        forcing_callback=lambda step, t, old, g=grid: phase_exact_forcing(
            g, t, mu, nu, kappa
        ),
        keep_history=True,
    )
    assert history is not None
    exact_T, _, _, _ = phase_exact_fields(grid, T)
    error_T = state.u - exact_T
    lap_error_T = grid.anisotropic_laplacian(error_T)
    lap_exact_T = grid.anisotropic_laplacian(exact_T)
    numerator_h2 = 0.0
    denominator_h2 = 0.0
    for step, numerical in enumerate(history[1:], start=1):
        exact, _, _, _ = phase_exact_fields(grid, step * dt)
        numerator_h2 += dt * grid.h2k_norm(numerical - exact) ** 2
        denominator_h2 += dt * grid.h2k_norm(exact) ** 2
    metrics = {
        "n": int(n),
        "h": 1.0 / n,
        "dt": dt,
        "T": T,
        "E0": float(np.sqrt(np.mean(error_T**2)) / np.sqrt(np.mean(exact_T**2))),
        "EL": float(
            np.sqrt(np.mean(lap_error_T**2))
            / np.sqrt(np.mean(lap_exact_T**2))
        ),
        "EH2_time": math.sqrt(numerator_h2 / denominator_h2),
        "max_mass_defect": max(d["mass_defect"] for d in state.diagnostics),
        "max_state_residual": max(d["state_residual"] for d in state.diagnostics),
        "max_laplacian_tail": max(d["tail_laplacian"] for d in state.diagnostics),
        "max_admm_iterations": max(d["admm_iterations"] for d in state.diagnostics),
    }
    return metrics, state.u.copy(), history


def run_phase_exact_spatial_solution(
    T: float,
    mu: float,
    nu: float,
    kappa: float,
) -> pd.DataFrame:
    """Four-level computed-solution study with a finest-level time control."""
    dt = 2.5e-7
    resolutions = [15, 23, 31, 39]
    rows: list[dict[str, float]] = []
    finest_field: np.ndarray | None = None
    finest_history: list[np.ndarray] | None = None
    for n in resolutions:
        row, field, history = _phase_solution_error(n, dt, T, mu, nu, kappa)
        rows.append(row)
        if n == resolutions[-1]:
            finest_field = field
            finest_history = history

    spatial = pd.DataFrame(rows).sort_values("h", ascending=False)
    scales = spatial["h"].to_numpy(dtype=float)
    for column in ["E0", "EL", "EH2_time"]:
        values = spatial[column].to_numpy(dtype=float)
        order = np.full(values.shape, np.nan, dtype=float)
        order[1:] = np.log(values[:-1] / values[1:]) / np.log(
            scales[:-1] / scales[1:]
        )
        spatial[f"order_{column}"] = order
    spatial.to_csv(DATA / "phase_exact_solution_spatial.csv", index=False)

    # Doubling the step on the finest grid supplies a direct bound on the
    # residual time-discretisation contribution to this spatial curve.
    coarse_dt = 2.0 * dt
    coarse, coarse_field, coarse_history = _phase_solution_error(
        resolutions[-1], coarse_dt, T, mu, nu, kappa
    )
    assert finest_field is not None and finest_history is not None
    grid = reconstruction_grid(resolutions[-1])
    exact_T, _, _, _ = phase_exact_fields(grid, T)
    difference_T = finest_field - coarse_field
    lap_difference_T = grid.anisotropic_laplacian(difference_T)
    lap_exact_T = grid.anisotropic_laplacian(exact_T)
    control_h2_num = 0.0
    control_h2_den = 0.0
    for step, coarse_state in enumerate(coarse_history[1:], start=1):
        fine_state = finest_history[2 * step]
        exact, _, _, _ = phase_exact_fields(grid, step * coarse_dt)
        control_h2_num += coarse_dt * grid.h2k_norm(fine_state - coarse_state) ** 2
        control_h2_den += coarse_dt * grid.h2k_norm(exact) ** 2
    finest = spatial.iloc[-1]
    control = {
        "n": resolutions[-1],
        "fine_dt": dt,
        "coarse_dt": coarse_dt,
        "fine_E0": float(finest.E0),
        "coarse_E0": float(coarse["E0"]),
        "fine_EL": float(finest.EL),
        "coarse_EL": float(coarse["EL"]),
        "fine_EH2_time": float(finest.EH2_time),
        "coarse_EH2_time": float(coarse["EH2_time"]),
        "solution_change_E0": float(
            np.sqrt(np.mean(difference_T**2)) / np.sqrt(np.mean(exact_T**2))
        ),
        "solution_change_EL": float(
            np.sqrt(np.mean(lap_difference_T**2))
            / np.sqrt(np.mean(lap_exact_T**2))
        ),
        "solution_change_EH2_time": math.sqrt(control_h2_num / control_h2_den),
    }
    for metric in ["E0", "EL", "EH2_time"]:
        control[f"solution_change_fraction_of_{metric}"] = (
            control[f"solution_change_{metric}"] / control[f"fine_{metric}"]
        )
        control[f"scalar_error_change_fraction_{metric}"] = abs(
            control[f"coarse_{metric}"] - control[f"fine_{metric}"]
        ) / control[f"fine_{metric}"]
    (DATA / "phase_exact_solution_spatial_time_control.json").write_text(
        json.dumps(control, indent=2)
    )
    return spatial


def run_phase_exact() -> pd.DataFrame:
    T = 1.0e-3
    mu, nu, kappa = 3.0e-3, 5.0e-2, 1.0e-2
    rows = []
    final_field = None
    term_balance = None
    for dt in [2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5]:
        grid = reconstruction_grid(95)
        u0, _, uref, lam = phase_exact_fields(grid, 0.0)
        par = PhaseParameters(
            mu=mu,
            nu=nu,
            dt=dt,
            classifier="smooth",
            N=kappa,
            admm_tol=1.0e-8,
            admm_maxiter=8000,
        )
        nsteps = round(T / dt)
        state, hist = solve_phase(
            grid,
            u0,
            uref,
            lam,
            par,
            nsteps,
            forcing_callback=lambda n, t, old, g=grid: phase_exact_forcing(g, t, mu, nu, kappa),
            keep_history=True,
        )
        assert hist is not None
        uT, _, _, _ = phase_exact_fields(grid, T)
        err = state.u - uT
        e0 = np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(uT**2))
        le = grid.anisotropic_laplacian(err)
        lu = grid.anisotropic_laplacian(uT)
        ed = np.sqrt(np.mean(le**2)) / np.sqrt(np.mean(lu**2))
        num_h2 = 0.0
        den_h2 = 0.0
        for j, uj in enumerate(hist[1:], start=1):
            ex, _, _, _ = phase_exact_fields(grid, j * dt)
            ej = uj - ex
            num_h2 += dt * grid.h2k_norm(ej) ** 2
            den_h2 += dt * grid.h2k_norm(ex) ** 2
        eh2 = math.sqrt(num_h2 / den_h2)
        rows.append(
            {
                "dt": dt,
                "E0": e0,
                "EL": ed,
                "EH2_time": eh2,
                "max_mass_defect": max(d["mass_defect"] for d in state.diagnostics),
                "max_state_residual": max(d["state_residual"] for d in state.diagnostics),
                "max_laplacian_tail": max(d["tail_laplacian"] for d in state.diagnostics),
                "max_admm_iterations": max(d["admm_iterations"] for d in state.diagnostics),
            }
        )
        if dt == 2.5e-5:
            final_field = uT

    df = pd.DataFrame(rows).sort_values("dt", ascending=False)
    for col in ["E0", "EL", "EH2_time"]:
        vals = df[col].to_numpy()
        dts = df["dt"].to_numpy()
        order = np.full_like(vals, np.nan)
        order[1:] = np.log(vals[:-1] / vals[1:]) / np.log(dts[:-1] / dts[1:])
        df[f"order_{col}"] = order
    df.to_csv(DATA / "phase_exact_temporal.csv", index=False)

    grid0 = reconstruction_grid(189)
    u, ut, uref, lam = phase_exact_fields(grid0, 0.0)
    Lu = grid0.anisotropic_laplacian(u)
    terms = {
        "time derivative": ut,
        "double-well": grid0.anisotropic_laplacian(u**3 - u),
        "biharmonic": grid0.anisotropic_laplacian(-mu * Lu),
        "active": grid0.anisotropic_laplacian(-nu * classifier_arctan(Lu, kappa) * grid0.grad_norm_k(u)),
        "confidence": lam * (uref - u),
    }
    # Every entry in this balance is evaluated on the same 189-by-189 grid.
    # This reporting forcing is formed from those same discrete terms; the
    # independently oversampled-and-restricted forcing used in the time-step
    # calculation above is intentionally reported by the spatial residual
    # study instead.
    forcing = ut - sum(terms[name] for name in ["double-well", "biharmonic", "active", "confidence"])
    term_balance = {k: float(np.sqrt(np.mean(v**2))) for k, v in terms.items()}
    term_balance["forcing"] = float(np.sqrt(np.mean(forcing**2)))
    closure = ut - sum(terms[name] for name in ["double-well", "biharmonic", "active", "confidence"]) - forcing
    term_balance["closure_relative_residual"] = float(
        np.sqrt(np.mean(closure**2)) / max(np.sqrt(np.mean(ut**2)), 1.0e-30)
    )
    term_balance["grid_modes"] = int(grid0.nx)
    (DATA / "phase_exact_term_balance.json").write_text(json.dumps(term_balance, indent=2))

    residual_rows = []
    for n in [15, 23, 31, 47, 63, 95, 127]:
        g = reconstruction_grid(n)
        u, ut, uref, lam = phase_exact_fields(g, 0.0)
        f = phase_exact_forcing(g, 0.0, mu, nu, kappa)
        Lu = g.anisotropic_laplacian(u)
        rhs = g.anisotropic_laplacian(
            u**3 - u - mu * Lu - nu * classifier_arctan(Lu, kappa) * g.grad_norm_k(u)
        ) + lam * (uref - u) + f
        residual_rows.append(
            {
                "n": n,
                "forcing_grid_modes": 3 * n,
                "manufactured_relative_residual": np.sqrt(np.mean((ut-rhs)**2))
                / np.sqrt(np.mean(ut**2)),
            }
        )
    pd.DataFrame(residual_rows).to_csv(DATA / "phase_exact_spatial.csv", index=False)
    run_phase_exact_spatial_solution(T, mu, nu, kappa)
    return df


def ep_exact_fields(X: np.ndarray, Y: np.ndarray, t: float):
    v = 0.35 + 0.05 * np.exp(-2.0 * t) * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    vt = -0.10 * np.exp(-2.0 * t) * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    h = 0.60 + 0.04 * np.exp(-t) * np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    ht = -0.04 * np.exp(-t) * np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    return v, vt, h, ht


def ep_exact_error(
    n: int,
    dt: float,
    T: float = 0.04,
    d_long: float = 5.0e-4,
    d_trans: float = 3.0e-4,
) -> dict[str, float]:
    """Return manufactured-solution errors on one independently chosen mesh/time step."""
    dx = 1.0 / n
    par = EPParameters(
            dx=dx,
            dy=dx,
            dt=dt,
            d_long=d_long,
            d_trans=d_trans,
            tau_in=0.30,
            tau_out=6.0,
            tau_open=0.50,
            tau_close=0.80,
            v_gate=0.13,
            gate_slope=40.0,
    )
    # ``solve_monodomain`` is finite volume and therefore evaluates its
    # material and forcing fields at cell centres.
    x = (np.arange(n)[:, None] + 0.5) * dx
    y = (np.arange(n)[None, :] + 0.5) * dx
    substrate = -np.ones((n, n))
    eta = float(conductivity_factor(np.array([-1.0]), par.eta_min, par.conductivity_slope)[0])
    Dx, Dy = par.d_long * eta, par.d_trans * eta

    def vf(t, X, Y):
        v, vt, h, _ = ep_exact_fields(X, Y, t)
        vxx = -(2.0*np.pi)**2 * (v - 0.35)
        vyy = vxx
        reaction = h * v**2 * (1.0-v) / par.tau_in - v/par.tau_out
        return vt - (Dx*vxx + Dy*vyy) - reaction

    def hf(t, X, Y):
        v, _, h, ht = ep_exact_fields(X, Y, t)
        z = np.clip(par.gate_slope * (v - par.v_gate), -60.0, 60.0)
        H = 1.0 / (1.0 + np.exp(-z))
        A = (1.0-H)/par.tau_open
        B = A + H/par.tau_close
        return ht - (A-B*h)

    v0, _, h0, _ = ep_exact_fields(x, y, 0.0)
    sol = solve_monodomain(
        substrate,
        par,
        T,
        vf,
        v0=v0,
        h0=h0,
        boundary="periodic",
        gate_forcing=hf,
    )
    ve, _, he, _ = ep_exact_fields(x, y, T)
    ev = np.sqrt(np.mean((sol.v-ve)**2))/np.sqrt(np.mean(ve**2))
    eh = np.sqrt(np.mean((sol.h-he)**2))/np.sqrt(np.mean(he**2))
    # This smooth periodic manufactured field makes the spectral H1
    # diagnostic an accurate derivative norm, independently of the FV update.
    sg = SpectralGrid(n, n)
    dv = sol.v-ve
    dvx, dvy = sg.grad(dv)
    vex, vey = sg.grad(ve)
    eh1 = np.sqrt(np.mean(dv**2+dvx**2+dvy**2))/np.sqrt(np.mean(ve**2+vex**2+vey**2))
    return {"n": n, "dx": dx, "dt": dt, "EV_L2": ev, "EV_H1": eh1, "Eh_L2": eh, **sol.diagnostics}


def _add_orders(df: pd.DataFrame, scale: str, columns: list[str]) -> pd.DataFrame:
    """Add pairwise observed orders after sorting from coarse to fine."""
    out = df.sort_values(scale, ascending=False).copy()
    scales = out[scale].to_numpy(dtype=float)
    for col in columns:
        values = out[col].to_numpy(dtype=float)
        order = np.full(values.shape, np.nan, dtype=float)
        order[1:] = np.log(values[:-1] / values[1:]) / np.log(scales[:-1] / scales[1:])
        out[f"order_{col}"] = order
    return out


def run_ep_exact() -> pd.DataFrame:
    # Temporal and spatial studies are intentionally separate: their fixed
    # companion discretisation is recorded in the corresponding CSV.
    temporal = _add_orders(
        pd.DataFrame([ep_exact_error(96, dt) for dt in [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4]]),
        "dt",
        ["EV_L2", "EV_H1", "Eh_L2"],
    )
    temporal.to_csv(DATA / "ep_exact_temporal.csv", index=False)
    spatial = _add_orders(
        pd.DataFrame([ep_exact_error(n, 2.5e-6, T=0.004, d_long=3.0e-2, d_trans=2.0e-2) for n in [48, 64, 96, 128]]),
        "dx",
        ["EV_L2", "EV_H1", "Eh_L2"],
    )
    spatial.to_csv(DATA / "ep_exact_spatial.csv", index=False)
    return temporal


def q_initial(grid: SpectralGrid) -> np.ndarray:
    q = (
        np.cos(2*np.pi*grid.X)
        +0.70*np.cos(2*np.pi*grid.Y)
        +0.35*np.cos(2*np.pi*(grid.X+grid.Y))
        +0.20*np.sin(2*np.pi*(2*grid.X-grid.Y))
    )
    return 0.20*q/2.25


def low_mode_rms(grid: SpectralGrid, u: np.ndarray, cutoff: float=6.0) -> float:
    uh=grid.fft(u)/(grid.nx*grid.ny)
    ix=np.fft.fftfreq(grid.nx)*grid.nx; iy=np.fft.fftfreq(grid.ny)*grid.ny
    IX,IY=np.meshgrid(ix,iy,indexing="ij"); rad=np.sqrt(IX**2+IY**2)
    mask=(rad>0)&(rad<=cutoff)
    return float(np.sqrt(np.sum(np.abs(uh[mask])**2)))


def run_graph_limit() -> pd.DataFrame:
    rows=[]
    representative={}
    for n,dt,label in [(31,1e-4,"L0"),(47,5e-5,"L1"),(63,2.5e-5,"L2")]:
        g=reconstruction_grid(n)
        u0=q_initial(g); ref=u0.copy(); lam=np.full_like(u0,.05)
        pg=PhaseParameters(mu=3e-3,nu=1e-2,dt=dt,classifier="graph",admm_tol=1e-8)
        ns=round(5e-4/dt)
        sg,hg=solve_phase(g,u0,ref,lam,pg,ns,keep_history=True)
        assert hg is not None
        alpha=dt*pg.nu; rho=pg.rho_factor*alpha
        chi_g=(rho/alpha)*sg.y
        for N in [8,32,128,512]:
            ps=PhaseParameters(mu=3e-3,nu=1e-2,dt=dt,classifier="smooth",N=N,admm_tol=1e-8)
            ss,hs=solve_phase(g,u0,ref,lam,ps,ns,keep_history=True)
            assert hs is not None
            num=den=0.0
            for ug,us in zip(hg[1:],hs[1:]):
                num += dt*g.h2k_norm(us-ug)**2
                den += dt*g.h2k_norm(ug)**2
            eH=math.sqrt(num/den)
            einf=max(np.sqrt(np.mean((us-ug)**2)) for ug,us in zip(hg,hs))/max(np.sqrt(np.mean(ug**2)) for ug in hg)
            chi_s=(ps.rho_factor)*ss.y
            echi=low_mode_rms(g,chi_s-chi_g)/max(low_mode_rms(g,chi_g),1e-30)
            rows.append({
                "level":label,"n":n,"dt":dt,"N":N,"EH2_time":eH,"Einf_L2":einf,"Echi_low":echi,
                "graph_max_primal":max(d["primal_residual"] for d in sg.diagnostics),
                "graph_max_dual":max(d["dual_residual"] for d in sg.diagnostics),
                "graph_max_state":max(d["state_residual"] for d in sg.diagnostics),
                "graph_max_box":max(d["box_residual"] for d in sg.diagnostics),
                "graph_max_comp":max(d["complementarity_residual"] for d in sg.diagnostics),
                "graph_max_projection":max(d["graph_projection_residual"] for d in sg.diagnostics),
                "graph_max_iterations":max(d["admm_iterations"] for d in sg.diagnostics),
            })
            if label=="L2" and N in (8,32,128,512): representative[N]=ss.u.copy()
        if label=="L2":
            frozen_gradient = g.grad_norm_k(hg[-2])
            representative["graph"] = sg.u.copy()
            # z and y are tied to the final predictor c before the fidelity split.
            representative["laplacian"] = sg.z.copy()
            active_mask = frozen_gradient > 1.0e-10 * max(float(np.max(frozen_gradient)), 1.0e-30)
            representative["xi"] = np.divide(
                chi_g,
                frozen_gradient,
                out=np.zeros_like(chi_g),
                where=active_mask,
            )
            gamma = 1.0 / max(1.0, float(np.sqrt(np.mean(representative["laplacian"] ** 2))))
            representative["active_mask"] = active_mask
            representative["projection_residual"] = np.where(
                active_mask,
                representative["xi"] - np.clip(
                    representative["xi"] + gamma * representative["laplacian"], -1.0, 1.0
                ),
                0.0,
            )
    df=pd.DataFrame(rows)
    df.to_csv(DATA/"graph_limit.csv",index=False)
    np.savez_compressed(DATA/"graph_representative.npz",**{str(k):v for k,v in representative.items()})
    return df


def _stripe_stimulus(shape, axis: str, amplitude=1.2, duration=2.0, width=2.0):
    def fn(t,x,y):
        if t>=duration: return np.zeros(shape)
        mask=(x<width) if axis=="x" else (y<width)
        return amplitude*np.broadcast_to(mask,shape)
    return fn


def run_conduction_calibration() -> pd.DataFrame:
    """Separate mesh and time-step refinements for the prescribed EP scale.

    The mesh series holds ``dt=0.01 ms`` fixed.  The time series holds
    ``dx=0.5 mm`` fixed.  Thus neither reported trend is a coupled h--dt
    comparison.
    """
    cases = [
        ("space", "h=1.00 mm", 1.0, 0.01),
        ("space", "h=0.50 mm", 0.5, 0.01),
        ("space", "h=0.25 mm", 0.25, 0.01),
        ("time", "dt=0.040 ms", 0.5, 0.04),
        ("time", "dt=0.020 ms", 0.5, 0.02),
        ("time", "dt=0.010 ms", 0.5, 0.01),
    ]
    rows=[]
    for study, label, dx, dt in cases:
        for axis in ["x","y"]:
            Lprop=70.0; Lcross=20.0
            if axis=="x": nx,ny=round(Lprop/dx),round(Lcross/dx)
            else: nx,ny=round(Lcross/dx),round(Lprop/dx)
            sub=-np.ones((nx,ny))
            p=EPParameters(dx=dx,dy=dx,dt=dt,d_long=.32,d_trans=.0512)
            trace=(int((50.0 if axis=="x" else 10.0)/dx),int((10.0 if axis=="x" else 30.0)/dx))
            sol=solve_monodomain(sub,p,350.0,_stripe_stimulus(sub.shape,axis),trace_points={"probe":trace},fiber_axis="x")
            if axis=="x": cv,r2,nfit=activation_cv(sol.activation,dx,0,(10.0,55.0))
            else: cv,r2,nfit=activation_cv(sol.activation,dx,1,(10.0,55.0))
            rows.append({
                "study": study,
                "configuration":label,
                "dx_mm":dx,
                "dt_ms":dt,
                "direction":axis,
                "cv_m_per_s":cv,
                "regression_R2":r2,"n_fit":nfit,"APD90_ms":apd90(sol.traces["time"],sol.traces["probe"]),
                "is_production": bool(np.isclose(dx, 0.5) and np.isclose(dt, 0.02)),
                **sol.diagnostics,
            })
    df=pd.DataFrame(rows)
    for study, scale in [("space", "dx_mm"), ("time", "dt_ms")]:
        for direction in ["x", "y"]:
            idx = df.index[(df["study"] == study) & (df["direction"] == direction)]
            ordered = df.loc[idx].sort_values(scale, ascending=False)
            reference = ordered.iloc[-1]
            df.loc[ordered.index, "cv_relative_to_finest"] = np.abs(
                ordered["cv_m_per_s"] / reference["cv_m_per_s"] - 1.0
            )
            df.loc[ordered.index, "apd90_relative_to_finest"] = np.abs(
                ordered["APD90_ms"] / reference["APD90_ms"] - 1.0
            )
    df.to_csv(DATA/"conduction_calibration.csv",index=False)
    return df


def make_exact_figure(phase_df: pd.DataFrame, ep_df: pd.DataFrame) -> None:
    phase_spatial = pd.read_csv(DATA / "phase_exact_solution_spatial.csv")
    ep_spatial = pd.read_csv(DATA / "ep_exact_spatial.csv")
    terms=json.loads((DATA/"phase_exact_term_balance.json").read_text())
    g=reconstruction_grid(95); ue,_,_,_=phase_exact_fields(g,1e-3)
    fig,axs=plt.subplots(2,3,figsize=(7.15,5.05),constrained_layout=True)
    im=axs[0,0].imshow(ue.T,origin="lower",extent=(0,1,0,1),cmap="RdBu_r",aspect="equal")
    axs[0,0].set_title(r"(a) prescribed $u_\star(T)$",fontsize=8.2); axs[0,0].set_xlabel("x"); axs[0,0].set_ylabel("y"); fig.colorbar(im,ax=axs[0,0],shrink=.75)
    phase_ratio=phase_df.dt/phase_df.dt.min()
    for c,m,l in [("E0","o",r"$E_0$"),("EL","s",r"$E_{\mathcal{L}}$"),("EH2_time","^",r"$E_{H^2,\tau}$")]: axs[0,1].semilogy(phase_ratio,phase_df[c],marker=m,label=l)
    axs[0,1].invert_xaxis(); axs[0,1].set_xticks([1,2,4,8],["1","2","4","8"]); axs[0,1].set_title("(b) phase-field time error",fontsize=8.2); axs[0,1].set_xlabel(r"$\Delta\tau/(2.5\times10^{-5})$"); axs[0,1].set_ylabel("relative error"); axs[0,1].legend(frameon=False)
    for c,m,l in [("E0","o",r"$E_0$"),("EL","s",r"$E_{\mathcal{L}}$"),("EH2_time","^",r"$E_{H^2,\tau}$")]:
        axs[0,2].loglog(phase_spatial.n,phase_spatial[c],marker=m,label=l)
    axs[0,2].set_title("(c) phase-field spatial error",fontsize=8.2); axs[0,2].set_xlabel("modes per direction"); axs[0,2].set_ylabel("relative error"); axs[0,2].legend(frameon=False)
    ep_ratio=ep_df.dt/ep_df.dt.min()
    for c,m,l in [("EV_L2","o",r"$V:L^2$"),("EV_H1","s",r"$V:H^1$"),("Eh_L2","^",r"$r:L^2$")]: axs[1,0].semilogy(ep_ratio,ep_df[c],marker=m,label=l)
    axs[1,0].invert_xaxis(); axs[1,0].set_xticks([1,2,4,8],["1","2","4","8"]); axs[1,0].set_title("(d) monodomain time error",fontsize=8.2); axs[1,0].set_xlabel(r"$\Delta t/(2.5\times10^{-4}\,\mathrm{ms})$"); axs[1,0].set_ylabel("relative error"); axs[1,0].legend(frameon=False)
    names=["time derivative","double-well","biharmonic","active","confidence","forcing"]; vals=[terms[k] for k in names]; axs[1,1].barh(names,vals,color="#2a7da8"); axs[1,1].set_xscale("log"); axs[1,1].set_title("(e) phase-field term balance"); axs[1,1].set_xlabel("RMS magnitude")
    for c,m,l in [("EV_L2","o",r"$V:L^2$"),("EV_H1","s",r"$V:H^1$"),("Eh_L2","^",r"$r:L^2$")]:
        axs[1,2].loglog(ep_spatial.n,ep_spatial[c],marker=m,label=l)
    axs[1,2].set_title("(f) monodomain spatial error",fontsize=8.2); axs[1,2].set_xlabel("cells per direction"); axs[1,2].set_ylabel("relative error")
    axs[1,2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                    ncol=3, borderaxespad=0, handlelength=1.2,
                    handletextpad=0.35, columnspacing=0.7)
    fig.savefig(FIG/"fig1_exact_verification.pdf"); fig.savefig(FIG/"fig1_exact_verification.png"); plt.close(fig)


def make_core_figures(phase_df, ep_df, graph_df, cv_df):
    make_exact_figure(phase_df, ep_df)
    rep=np.load(DATA/"graph_representative.npz")
    fig,axs=plt.subplots(2,3,figsize=(7.15,4.70),constrained_layout=True)
    for ax,key,title,cmap in [(axs[0,0],"graph",r"(a) graph substrate $u(T)$","RdBu_r"),(axs[0,1],"laplacian",r"(b) predictor $\mathcal{L}_K c(T)$","RdBu_r"),(axs[0,2],"xi",r"(c) graph multiplier $\xi$","coolwarm")]:
        im=ax.imshow(rep[key].T,origin="lower",extent=(0,1,0,1),cmap=cmap,aspect="equal"); ax.set_title(title,fontsize=8.2); ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(im,ax=ax,shrink=.68,pad=.02)
    styles={"L0":":","L1":"--","L2":"-"}
    for lev,gdf in graph_df.groupby("level"):
        axs[1,0].loglog(gdf.N,gdf.EH2_time,"o"+styles[lev],label=lev)
        axs[1,1].loglog(gdf.N,gdf.Echi_low,"o"+styles[lev],label=lev)
    axs[1,0].set_title(r"(d) state error to graph"); axs[1,0].set_xlabel("classifier steepness N"); axs[1,0].set_ylabel(r"$E_N^{H^2}$"); axs[1,0].legend(frameon=False)
    axs[1,1].set_title(r"(e) active-product error"); axs[1,1].set_xlabel("classifier steepness N"); axs[1,1].set_ylabel(r"$E_N^\chi$")
    gd=graph_df[graph_df.level=="L2"].iloc[0]
    vals=[gd.graph_max_primal,gd.graph_max_dual,gd.graph_max_state,gd.graph_max_comp,gd.graph_max_box]
    axs[1,2].bar(range(5),vals,color="#159f74"); axs[1,2].set_yscale("log"); axs[1,2].set_xticks(range(5),["pri","dual","state","comp","box"],rotation=25); axs[1,2].set_title("(f) graph algebraic closure"); axs[1,2].set_ylabel("maximum residual")
    fig.savefig(FIG/"fig2_graph_limit.pdf"); fig.savefig(FIG/"fig2_graph_limit.png"); plt.close(fig)

    make_conduction_figure(cv_df)


def make_conduction_figure(cv_df):
    extension = DATA / "conduction_extension.csv"
    if extension.exists():
        fine = pd.read_csv(extension)
        fine = fine[(fine.dx_mm == .125) & (fine.dt_ms == .01)].copy()
        fine["study"] = "space"
        fine["is_production"] = False
        cv_df = pd.concat([cv_df, fine], ignore_index=True)
    fig,axs=plt.subplots(1,3,figsize=(7.15,2.45),constrained_layout=True)
    prod=cv_df[cv_df.is_production]
    axs[0].bar(["longitudinal","transverse"],prod.cv_m_per_s,color=["#2a7da8","#e07a3f"]); axs[0].set_ylabel("conduction velocity (m/s)"); axs[0].set_title("(a) Production conduction velocities")
    for direction,marker in [("x","o"),("y","s")]:
        d=cv_df[(cv_df.study=="space") & (cv_df.direction==direction)].sort_values("dx_mm",ascending=False)
        axs[1].plot(d.dx_mm,d.cv_m_per_s,marker=marker,label="longitudinal" if direction=="x" else "transverse")
    axs[1].invert_xaxis(); axs[1].set_xlabel(r"mesh spacing (mm), fixed $\Delta t=0.01$ ms"); axs[1].set_ylabel("conduction velocity (m/s)"); axs[1].set_title("(b) mesh refinement"); axs[1].legend(frameon=False)
    for direction,marker in [("x","o"),("y","s")]:
        d=cv_df[(cv_df.study=="time") & (cv_df.direction==direction)].sort_values("dt_ms",ascending=False)
        axs[2].plot(d.dt_ms,d.APD90_ms,marker=marker,label="longitudinal" if direction=="x" else "transverse")
    axs[2].invert_xaxis(); axs[2].set_xlabel("time step (ms), fixed $h=0.5$ mm"); axs[2].set_ylabel("APD90 (ms)"); axs[2].set_title("(c) time-step refinement",fontsize=8.0); axs[2].legend(frameon=False)
    fig.savefig(FIG/"fig3_conduction_calibration.pdf"); fig.savefig(FIG/"fig3_conduction_calibration.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-1-only",
        action="store_true",
        help="regenerate the exact-solution data and Figure 1 only",
    )
    parser.add_argument("--conduction-figure-only", action="store_true", help="redraw Figure 3 from the stored conduction results only")
    args = parser.parse_args()
    if args.conduction_figure_only:
        make_conduction_figure(pd.read_csv(DATA / "conduction_calibration.csv"))
        return
    phase=run_phase_exact(); print("phase exact\n",phase)
    if args.experiment_1_only:
        # Experiment 1 contains both manufactured solvers.  Regenerate the EP
        # temporal and spatial tables as well, so this focused command cannot
        # silently reuse missing or stale EP verification artifacts.
        ep = run_ep_exact(); print("EP exact\n",ep)
        make_exact_figure(phase, ep)
        return
    ep=run_ep_exact(); print("EP exact\n",ep)
    graph=run_graph_limit(); print("graph\n",graph)
    cv=run_conduction_calibration(); print("calibration\n",cv[["configuration","direction","cv_m_per_s","APD90_ms","regression_R2"]])
    make_core_figures(phase,ep,graph,cv)


if __name__ == "__main__":
    main()
