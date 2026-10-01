"""Numerical kernels for the atrial sign-graph manuscript.

The substrate equation is solved on a periodic two-dimensional chart.  The
Fourier implementation is deliberately compact, but retains the two
structures used in the analysis: the classifier is implicit in the new
anisotropic Laplacian and the non-negative gradient factor is lagged.  The
maximal-monotone graph is solved directly by weighted ADMM.

The electrophysiology readout uses a variable-diffusivity monodomain model
with smoothed Mitchell--Schaeffer kinetics.  It is an idealised atrial-sheet
solver, not a patient-specific clinical simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.signal import resample
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres, spsolve


Array = NDArray[np.float64]


def _rms(x: Array) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def _safe_rel(num: float, den: float) -> float:
    return float(num / max(den, 1.0e-30))


@dataclass
class SpectralGrid:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0
    kx_weight: float = 1.0
    ky_weight: float = 1.0

    def __post_init__(self) -> None:
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.A = self.kx_weight * self.KX**2 + self.ky_weight * self.KY**2
        self.A2 = self.A**2

    @staticmethod
    def fft(u: Array) -> NDArray[np.complex128]:
        return np.fft.fftn(u)

    @staticmethod
    def ifft(uh: NDArray[np.complex128]) -> Array:
        return np.fft.ifftn(uh).real

    def anisotropic_laplacian(self, u: Array) -> Array:
        """Return L_K u = div(K grad u), so its symbol is -A."""
        return self.ifft(-self.A * self.fft(u))

    def grad(self, u: Array) -> Tuple[Array, Array]:
        uh = self.fft(u)
        return self.ifft(1j * self.KX * uh), self.ifft(1j * self.KY * uh)

    def grad_norm_k(self, u: Array) -> Array:
        ux, uy = self.grad(u)
        return np.sqrt(self.kx_weight * ux**2 + self.ky_weight * uy**2)

    def dealiased_cubic(self, u: Array) -> Array:
        """Evaluate ``u**3`` with twofold Fourier padding and restriction.

        A cubic product needs the twofold rule to prevent its unresolved
        modes from folding back into the retained Fourier band.  Odd grids
        avoid the singled-out Nyquist coefficient for which Fourier
        prolongation is convention-dependent.
        """
        if self.nx % 2 == 0 or self.ny % 2 == 0:
            raise ValueError(
                "dealiased_cubic requires odd grid dimensions to avoid "
                "Nyquist-mode ambiguity"
            )
        mx = 2 * self.nx
        my = 2 * self.ny
        padded = resample(resample(u, mx, axis=0), my, axis=1)
        cubic = padded**3
        restricted = resample(resample(cubic, self.nx, axis=0), self.ny, axis=1)
        return np.asarray(restricted.real, dtype=float)

    def h2k_norm(self, u: Array) -> float:
        lu = self.anisotropic_laplacian(u)
        ux, uy = self.grad(u)
        return float(
            np.sqrt(
                np.mean(u**2)
                + np.mean(self.kx_weight * ux**2 + self.ky_weight * uy**2)
                + np.mean(lu**2)
            )
        )

    def modal_tails(self, u: Array, fraction: float = 0.75) -> Tuple[float, float]:
        uh = self.fft(u) / (self.nx * self.ny)
        ix = np.abs(np.fft.fftfreq(self.nx) * self.nx)
        iy = np.abs(np.fft.fftfreq(self.ny) * self.ny)
        IX, IY = np.meshgrid(ix, iy, indexing="ij")
        cutoff_x = fraction * np.max(ix)
        cutoff_y = fraction * np.max(iy)
        tail = (IX >= cutoff_x) | (IY >= cutoff_y)
        e0 = np.abs(uh) ** 2
        ed = self.A2 * e0
        return _safe_rel(float(e0[tail].sum()), float(e0.sum())), _safe_rel(
            float(ed[tail].sum()), float(ed.sum())
        )


@dataclass
class PhaseParameters:
    mu: float = 3.0e-3
    nu: float = 2.0e-3
    dt: float = 5.0e-5
    classifier: str = "graph"  # graph, smooth, passive
    N: float = 32.0
    admm_tol: float = 1.0e-8
    admm_maxiter: int = 8000
    rho_factor: float = 40.0
    adaptive_stabilisation: bool = True
    fixed_stabilisation: float = 4.0


@dataclass
class PhaseState:
    u: Array
    z: Optional[Array] = None
    y: Optional[Array] = None
    diagnostics: list[Dict[str, float]] = field(default_factory=list)


def classifier_arctan(z: Array, N: float) -> Array:
    return (2.0 / np.pi) * np.arctan(N * z)


def classifier_arctan_prime(z: Array, N: float) -> Array:
    return (2.0 * N / np.pi) / (1.0 + (N * z) ** 2)


def _smooth_prox(v: Array, tau: Array, N: float) -> Array:
    """Proximal map for a primitive whose derivative is a_N.

    It solves z-v+tau*a_N(z)=0.  Strict monotonicity gives a unique root.
    The safeguarded Newton iterations are vectorised and remain inside the
    bracket [v-tau, v+tau], because |a_N| <= 1.
    """
    lo = v - tau
    hi = v + tau
    z = np.clip(v, lo, hi)
    for _ in range(40):
        f = z - v + tau * classifier_arctan(z, N)
        if float(np.max(np.abs(f))) <= 5.0e-14 * (1.0 + float(np.max(np.abs(v)))):
            break
        fp = 1.0 + tau * classifier_arctan_prime(z, N)
        proposal = z - f / fp
        proposal = np.where((proposal > lo) & (proposal < hi), proposal, 0.5 * (lo + hi))
        fproposal = proposal - v + tau * classifier_arctan(proposal, N)
        lo = np.where(fproposal < 0.0, proposal, lo)
        hi = np.where(fproposal >= 0.0, proposal, hi)
        z = proposal
    return z


def phase_step(
    grid: SpectralGrid,
    state: PhaseState,
    u_ref: Array,
    confidence: Array,
    par: PhaseParameters,
    forcing: Optional[Array] = None,
    data_forcing: Optional[Array] = None,
) -> PhaseState:
    """Advance one split phase-field step with linear confidence fidelity."""
    u = state.u
    dt = par.dt
    if forcing is None:
        forcing = np.zeros_like(u)
    umax = float(np.max(np.abs(u)))
    S = (
        max(4.0, 1.0 + 3.0 * max(1.0, umax) ** 2)
        if par.adaptive_stabilisation
        else par.fixed_stabilisation
    )

    uh = grid.fft(u)
    nonlin = grid.dealiased_cubic(u) - u
    rhs_hat = (1.0 + dt * S * grid.A) * uh - dt * grid.A * grid.fft(nonlin)
    rhs_hat += dt * grid.fft(forcing)
    B = 1.0 + dt * S * grid.A + dt * par.mu * grid.A2
    G = grid.grad_norm_k(u)

    if par.classifier == "passive" or par.nu == 0.0:
        c_hat = rhs_hat / B
        c = grid.ifft(c_hat)
        z = grid.anisotropic_laplacian(c)
        y = np.zeros_like(c)
        iterations = 0
        passive_residual = grid.ifft(B * c_hat - rhs_hat)
        r_state = _safe_rel(_rms(passive_residual), max(1.0, _rms(grid.ifft(rhs_hat))))
        r_pri = r_dual = r_box = r_comp = r_graph = 0.0
    elif par.classifier == "smooth":
        alpha = dt * par.nu
        rhs_scale = max(1.0, _rms(grid.ifft(rhs_hat)))
        c_hat = rhs_hat / B
        c = grid.ifft(c_hat)
        r_pri = r_dual = r_box = r_graph = 0.0
        for iterations in range(1, 101):
            Kc = grid.anisotropic_laplacian(c)
            p = alpha * G * classifier_arctan(Kc, par.N)
            F = grid.ifft(B * grid.fft(c) - rhs_hat - grid.A * grid.fft(p))
            rel = _safe_rel(_rms(F), rhs_scale)
            if rel <= par.admm_tol:
                break
            weight = alpha * G * classifier_arctan_prime(Kc, par.N)
            wbar = float(np.mean(weight))

            def jmv(flat: NDArray[np.float64]) -> NDArray[np.float64]:
                d = flat.reshape(u.shape)
                Kd = grid.anisotropic_laplacian(d)
                dp = weight * Kd
                out = grid.ifft(B * grid.fft(d) - grid.A * grid.fft(dp))
                return out.ravel()

            def pmv(flat: NDArray[np.float64]) -> NDArray[np.float64]:
                d = flat.reshape(u.shape)
                out = grid.ifft(grid.fft(d) / (B + wbar * grid.A2))
                return out.ravel()

            J = LinearOperator((u.size, u.size), matvec=jmv, dtype=float)
            M = LinearOperator((u.size, u.size), matvec=pmv, dtype=float)
            lin_rtol = max(1.0e-8, min(1.0e-3, 0.1 * rel))
            delta, info = gmres(
                J,
                -F.ravel(),
                M=M,
                restart=40,
                maxiter=80,
                rtol=lin_rtol,
                atol=1.0e-12,
            )
            if info != 0:
                raise RuntimeError(f"smooth-classifier GMRES failed with info={info}")
            delta = delta.reshape(u.shape)
            alpha_ls = 1.0
            old_norm = _rms(F)
            accepted = False
            while alpha_ls >= 2.0**-16:
                trial = c + alpha_ls * delta
                Ktrial = grid.anisotropic_laplacian(trial)
                ptrial = alpha * G * classifier_arctan(Ktrial, par.N)
                Ftrial = grid.ifft(B * grid.fft(trial) - rhs_hat - grid.A * grid.fft(ptrial))
                if _rms(Ftrial) <= (1.0 - 1.0e-4 * alpha_ls) * old_norm:
                    c = trial
                    accepted = True
                    break
                alpha_ls *= 0.5
            if not accepted:
                raise RuntimeError("smooth-classifier Newton line search failed")
        else:
            raise RuntimeError(f"smooth-classifier Newton failed, residual={rel:.3e}")
        Kc = grid.anisotropic_laplacian(c)
        p = alpha * G * classifier_arctan(Kc, par.N)
        F = grid.ifft(B * grid.fft(c) - rhs_hat - grid.A * grid.fft(p))
        r_state = _safe_rel(_rms(F), rhs_scale)
        r_comp = 0.0
        rho = par.rho_factor * alpha
        y = p / rho
        z = Kc
    else:
        alpha = dt * par.nu
        rho = par.rho_factor * alpha
        if state.z is None or state.z.shape != u.shape:
            c = grid.ifft(rhs_hat / B)
            z = grid.anisotropic_laplacian(c)
            y = np.zeros_like(c)
        else:
            z = np.array(state.z, copy=True)
            y = np.array(state.y, copy=True) if state.y is not None else np.zeros_like(u)

        rhs_scale = max(1.0, _rms(grid.ifft(rhs_hat)))
        zold = np.array(z, copy=True)
        r_pri = r_dual = np.inf
        for iterations in range(1, par.admm_maxiter + 1):
            numerator = rhs_hat - rho * grid.A * grid.fft(z - y)
            c_hat = numerator / (B + rho * grid.A2)
            c = grid.ifft(c_hat)
            Kc = grid.anisotropic_laplacian(c)
            v = Kc + y
            tau = alpha * G / rho
            if par.classifier == "graph":
                znew = np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
            else:
                raise ValueError(f"unknown classifier {par.classifier!r}")
            y += Kc - znew
            r_pri = _safe_rel(_rms(Kc - znew), max(1.0, _rms(Kc)))
            r_dual = _safe_rel(_rms(rho * grid.ifft(grid.A * grid.fft(znew - z))), rhs_scale)
            zold = z
            z = znew
            if max(r_pri, r_dual) <= par.admm_tol:
                break
        else:
            raise RuntimeError(
                f"ADMM failed: classifier={par.classifier}, residuals=({r_pri:.3e},{r_dual:.3e})"
            )

        p = rho * y
        c_hat = grid.fft(c)
        state_res = grid.ifft(B * c_hat - rhs_hat - grid.A * grid.fft(p))
        r_state = _safe_rel(_rms(state_res), max(1.0, _rms(grid.ifft(B * c_hat)), rhs_scale))
        if par.classifier == "graph":
            cap = alpha * G
            r_box = _safe_rel(float(np.max(np.maximum(np.abs(p) - cap, 0.0))), float(np.max(cap)))
            comp_num = np.mean(np.abs(cap * np.abs(grid.anisotropic_laplacian(c)) - p * grid.anisotropic_laplacian(c)))
            comp_den = np.mean(cap * np.abs(grid.anisotropic_laplacian(c)))
            r_comp = _safe_rel(float(comp_num), float(comp_den))
            active_mask = cap > 1.0e-10 * max(float(np.max(cap)), 1.0e-30)
            xi = np.divide(p, cap, out=np.zeros_like(p), where=active_mask)
            gamma = 1.0 / max(1.0, _rms(grid.anisotropic_laplacian(c)))
            proj = np.clip(xi + gamma * grid.anisotropic_laplacian(c), -1.0, 1.0)
            r_graph = _rms((xi - proj)[active_mask]) if np.any(active_mask) else 0.0
        else:
            raise ValueError(f"unknown classifier {par.classifier!r}")

    predictor = c
    # The natural EP confidence term is linear and may vanish.  This is the
    # exact algebraic solve of its backward-Euler split step; no clipping is
    # applied.
    if data_forcing is None:
        data_forcing = confidence * u_ref
    u_new = (predictor + dt * data_forcing) / (1.0 + dt * confidence)
    source = data_forcing - confidence * u_new + forcing
    mass_defect = abs(float(np.mean(u_new - u) - dt * np.mean(source)))
    rho0, rhod = grid.modal_tails(u_new)
    znew = grid.anisotropic_laplacian(predictor)
    monotonicity_min = 0.0
    if par.classifier == "smooth":
        monotonicity_min = float(np.min(classifier_arctan(znew, par.N) * znew))

    diag = {
        "admm_iterations": float(iterations),
        "primal_residual": float(r_pri),
        "dual_residual": float(r_dual),
        "state_residual": float(r_state),
        "box_residual": float(r_box),
        "complementarity_residual": float(r_comp),
        "graph_projection_residual": float(r_graph),
        "mass_defect": float(mass_defect),
        "tail_l2": float(rho0),
        "tail_laplacian": float(rhod),
        "min_classifier_product": float(monotonicity_min),
        "u_min": float(np.min(u_new)),
        "u_max": float(np.max(u_new)),
        "stabilisation": float(S),
    }
    return PhaseState(u=u_new, z=znew, y=y, diagnostics=state.diagnostics + [diag])


def solve_phase(
    grid: SpectralGrid,
    u0: Array,
    u_ref: Array,
    confidence: Array,
    par: PhaseParameters,
    nsteps: int,
    forcing_callback: Optional[Callable[[int, float, Array], Array]] = None,
    keep_history: bool = False,
    data_forcing: Optional[Array] = None,
) -> Tuple[PhaseState, Optional[list[Array]]]:
    state = PhaseState(np.array(u0, dtype=float, copy=True))
    history: Optional[list[Array]] = [state.u.copy()] if keep_history else None
    for n in range(nsteps):
        forcing = None if forcing_callback is None else forcing_callback(n + 1, (n + 1) * par.dt, state.u)
        state = phase_step(
            grid,
            state,
            u_ref,
            confidence,
            par,
            forcing,
            data_forcing=data_forcing,
        )
        if history is not None:
            history.append(state.u.copy())
    return state, history


def diffusivity_factor(u: Array, eta_min: float = 1.0e-3, kappa: float = 8.0) -> Array:
    """Map +1 lesion / -1 viable substrate to a positive diffusion factor."""
    z = np.clip(kappa * u, -60.0, 60.0)
    return eta_min + (1.0 - eta_min) / (1.0 + np.exp(z))


def conductivity_factor(u: Array, eta_min: float = 1.0e-3, kappa: float = 8.0) -> Array:
    """Backward-compatible alias for :func:`diffusivity_factor`."""
    return diffusivity_factor(u, eta_min, kappa)


@dataclass
class EPParameters:
    dx: float = 0.5  # mm
    dy: float = 0.5  # mm
    dt: float = 0.05  # ms
    d_long: float = 0.32  # mm^2/ms; prescribed and characterised in the numerical section
    d_trans: float = 0.0512
    tau_in: float = 0.30
    tau_out: float = 6.0
    tau_open: float = 100.0
    tau_close: float = 80.0
    v_gate: float = 0.13
    gate_slope: float = 80.0
    activation_threshold: float = 0.5
    eta_min: float = 1.0e-3
    conductivity_slope: float = 8.0
    face_average: str = "harmonic"


def _face_conductivity(left: Array, right: Array, mean: str) -> Array:
    """Return a positive face transmissibility for diagonal diffusion.

    The harmonic mean is the production choice because it preserves the
    resistance of a low-diffusivity layer.  The arithmetic option is retained
    only for the reported flux-sensitivity comparison.
    """
    if mean == "harmonic":
        return np.divide(
            2.0 * left * right,
            left + right,
            out=np.zeros_like(left),
            where=(left + right) > 0.0,
        )
    if mean == "arithmetic":
        return 0.5 * (left + right)
    raise ValueError("face_average must be 'harmonic' or 'arithmetic'")


def _div_diag_periodic(
    v: Array,
    dx_cond: Array,
    dy_cond: Array,
    dx: float,
    dy: float,
    face_average: str = "harmonic",
) -> Array:
    """Conservative diagonal-tensor diffusion with periodic boundaries."""
    dxe = _face_conductivity(dx_cond, np.roll(dx_cond, -1, axis=0), face_average)
    dxw = np.roll(dxe, 1, axis=0)
    dyn = _face_conductivity(dy_cond, np.roll(dy_cond, -1, axis=1), face_average)
    dys = np.roll(dyn, 1, axis=1)
    fx_e = dxe * (np.roll(v, -1, axis=0) - v) / dx
    fx_w = dxw * (v - np.roll(v, 1, axis=0)) / dx
    fy_n = dyn * (np.roll(v, -1, axis=1) - v) / dy
    fy_s = dys * (v - np.roll(v, 1, axis=1)) / dy
    return (fx_e - fx_w) / dx + (fy_n - fy_s) / dy


def _div_diag_noflux(
    v: Array,
    dx_cond: Array,
    dy_cond: Array,
    dx: float,
    dy: float,
    face_average: str = "harmonic",
) -> Array:
    """Conservative diagonal-tensor diffusion with zero normal boundary flux."""
    out = np.zeros_like(v)
    dxf = _face_conductivity(dx_cond[:-1, :], dx_cond[1:, :], face_average)
    fxf = dxf * (v[1:, :] - v[:-1, :]) / dx
    out[:-1, :] += fxf / dx
    out[1:, :] -= fxf / dx
    dyf = _face_conductivity(dy_cond[:, :-1], dy_cond[:, 1:], face_average)
    fyf = dyf * (v[:, 1:] - v[:, :-1]) / dy
    out[:, :-1] += fyf / dy
    out[:, 1:] -= fyf / dy
    return out


def _gate_coefficients(v: Array, par: EPParameters) -> Tuple[Array, Array]:
    z = np.clip(par.gate_slope * (v - par.v_gate), -60.0, 60.0)
    H = 1.0 / (1.0 + np.exp(-z))
    A = (1.0 - H) / par.tau_open
    B = A + H / par.tau_close
    return A, B


@dataclass
class EPSolution:
    v: Array
    h: Array
    activation: Array
    peak: Array
    traces: Dict[str, Array]
    diagnostics: Dict[str, float]
    snapshots: Dict[float, Array]


def solve_monodomain(
    substrate: Array,
    par: EPParameters,
    t_end: float,
    stimulus: Callable[[float, Array, Array], Array],
    v0: Optional[Array] = None,
    h0: Optional[Array] = None,
    trace_points: Optional[Dict[str, Tuple[int, int]]] = None,
    snapshot_times: Iterable[float] = (),
    fiber_axis: str = "x",
    boundary: str = "noflux",
    gate_forcing: Optional[Callable[[float, Array, Array], Array]] = None,
    diffusivity_scale: Optional[Array] = None,
) -> EPSolution:
    """Run an idealised monodomain simulation with first-order splitting.

    The gate is advanced exactly for its frozen-voltage linear ODE; voltage
    uses an explicit conservative diffusion/reaction update.  Every reported
    production run checks the explicit diffusion CFL number.
    """
    nx, ny = substrate.shape
    # These are cell centres on [0,nx*dx] x [0,ny*dy].
    x = (np.arange(nx)[:, None] + 0.5) * par.dx
    y = (np.arange(ny)[None, :] + 0.5) * par.dy
    if diffusivity_scale is None:
        eta = diffusivity_factor(substrate, par.eta_min, par.conductivity_slope)
    else:
        eta = np.asarray(diffusivity_scale, dtype=float)
        if eta.shape != substrate.shape:
            raise ValueError("diffusivity_scale must have the substrate shape")
        if not (np.isfinite(eta).all() and np.all(eta > 0.0)):
            raise ValueError("diffusivity_scale must be finite and strictly positive")
    if fiber_axis == "x":
        Dx, Dy = par.d_long * eta, par.d_trans * eta
    elif fiber_axis == "y":
        Dx, Dy = par.d_trans * eta, par.d_long * eta
    else:
        raise ValueError("fiber_axis must be 'x' or 'y'")

    cfl = par.dt * (2.0 * float(np.max(Dx)) / par.dx**2 + 2.0 * float(np.max(Dy)) / par.dy**2)
    if cfl >= 0.95:
        raise ValueError(f"explicit diffusion CFL too large: {cfl:.3f}")

    v = np.zeros_like(substrate) if v0 is None else np.array(v0, dtype=float, copy=True)
    h = np.ones_like(substrate) if h0 is None else np.array(h0, dtype=float, copy=True)
    activation = np.full_like(substrate, np.nan, dtype=float)
    peak = np.array(v, copy=True)
    trace_points = trace_points or {}
    trace_store: Dict[str, list[float]] = {name: [] for name in trace_points}
    trace_store["time"] = []
    requested = sorted(float(s) for s in snapshot_times)
    snapshots: Dict[float, Array] = {}
    snap_index = 0
    nsteps_float = t_end / par.dt
    nsteps = int(round(nsteps_float))
    if not np.isclose(nsteps_float, nsteps, rtol=0.0, atol=1.0e-10):
        raise ValueError("t_end must be an integer multiple of dt")
    max_reaction = 0.0
    max_diffusion = 0.0
    max_stimulus = 0.0
    max_diffusion_mass_defect = 0.0
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    hmin = float(np.min(h))
    hmax = float(np.max(h))

    for n in range(nsteps):
        t = n * par.dt
        oldv = v
        stim = np.asarray(stimulus(t, x, y), dtype=float)
        if stim.shape != v.shape:
            stim = np.broadcast_to(stim, v.shape)
        if boundary == "periodic":
            diff = _div_diag_periodic(v, Dx, Dy, par.dx, par.dy, par.face_average)
        elif boundary == "noflux":
            diff = _div_diag_noflux(v, Dx, Dy, par.dx, par.dy, par.face_average)
        else:
            raise ValueError("boundary must be 'periodic' or 'noflux'")
        reaction = h * v**2 * (1.0 - v) / par.tau_in - v / par.tau_out
        v = v + par.dt * (diff + reaction + stim)
        peak = np.maximum(peak, v)

        A, B = _gate_coefficients(v, par)
        decay = np.exp(-B * par.dt)
        h_inf = np.divide(A, B, out=np.zeros_like(A), where=B > 0.0)
        h = h_inf + (h - h_inf) * decay
        if gate_forcing is not None:
            gh = np.asarray(gate_forcing(t + par.dt, x, y), dtype=float)
            if gh.shape != h.shape:
                gh = np.broadcast_to(gh, h.shape)
            h = h + par.dt * gh

        crossed = np.isnan(activation) & (oldv < par.activation_threshold) & (v >= par.activation_threshold)
        denom = np.maximum(v - oldv, 1.0e-14)
        frac = np.clip((par.activation_threshold - oldv) / denom, 0.0, 1.0)
        activation[crossed] = t + par.dt * frac[crossed]

        trace_store["time"].append(t + par.dt)
        for name, ij in trace_points.items():
            trace_store[name].append(float(v[ij]))
        while snap_index < len(requested) and t + par.dt >= requested[snap_index] - 0.5 * par.dt:
            snapshots[requested[snap_index]] = v.copy()
            snap_index += 1

        max_reaction = max(max_reaction, _rms(reaction))
        max_diffusion = max(max_diffusion, _rms(diff))
        max_stimulus = max(max_stimulus, _rms(stim))
        max_diffusion_mass_defect = max(max_diffusion_mass_defect, abs(float(np.mean(diff))))
        vmin = min(vmin, float(np.min(v)))
        vmax = max(vmax, float(np.max(v)))
        hmin = min(hmin, float(np.min(h)))
        hmax = max(hmax, float(np.max(h)))
        if not (np.isfinite(v).all() and np.isfinite(h).all()):
            raise FloatingPointError(f"non-finite EP state at t={t:.3f} ms")

    traces = {k: np.asarray(vals, dtype=float) for k, vals in trace_store.items()}
    diagnostics = {
        "cfl": float(cfl),
        "v_min": float(vmin),
        "v_max": float(vmax),
        "h_min": float(hmin),
        "h_max": float(hmax),
        "max_reaction_rms": float(max_reaction),
        "max_diffusion_rms": float(max_diffusion),
        "max_stimulus_rms": float(max_stimulus),
        "max_diffusion_mass_defect": float(max_diffusion_mass_defect),
        "activated_fraction": float(np.mean(np.isfinite(activation))),
    }
    return EPSolution(v, h, activation, peak, traces, diagnostics, snapshots)


@dataclass
class CapacityResult:
    value: float
    inner_flux: float
    outer_flux: float
    energy: float
    relative_residual: float
    flux_energy_defect: float
    potential: Array


def annular_capacity(
    diffusivity_scale: Array,
    par: EPParameters,
    center: Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    fiber_axis: str = "x",
) -> CapacityResult:
    """Discrete effective diffusive conductance across an annulus.

    The potential is one on the inner electrode and zero on the outer
    electrode.  Harmonic face transmissibilities are identical to those used
    by the production monodomain diffusion operator.  The returned capacity is
    both the Dirichlet energy and the net inner-to-outer flux, up to the stated
    algebraic defect.
    """
    eta = np.asarray(diffusivity_scale, dtype=float)
    if eta.ndim != 2 or not (np.isfinite(eta).all() and np.all(eta > 0.0)):
        raise ValueError("diffusivity_scale must be a positive finite 2-D array")
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")

    nx, ny = eta.shape
    x = (np.arange(nx) + 0.5) * par.dx
    y = (np.arange(ny) + 0.5) * par.dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    radius = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    inner = radius <= inner_radius
    outer = radius >= outer_radius
    unknown = ~(inner | outer)
    if not (np.any(inner) and np.any(outer) and np.any(unknown)):
        raise ValueError("annular masks are empty")

    if fiber_axis == "x":
        Dx, Dy = par.d_long * eta, par.d_trans * eta
    elif fiber_axis == "y":
        Dx, Dy = par.d_trans * eta, par.d_long * eta
    else:
        raise ValueError("fiber_axis must be 'x' or 'y'")

    ids = np.full((nx, ny), -1, dtype=int)
    ids[unknown] = np.arange(int(np.sum(unknown)))
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros(int(np.sum(unknown)), dtype=float)

    def face_value(i: int, j: int, ni: int, nj: int) -> float:
        if ni != i:
            left = np.asarray([Dx[i, j]])
            right = np.asarray([Dx[ni, nj]])
            area_over_distance = par.dy / par.dx
        else:
            left = np.asarray([Dy[i, j]])
            right = np.asarray([Dy[ni, nj]])
            area_over_distance = par.dx / par.dy
        return float(_face_conductivity(left, right, par.face_average)[0] * area_over_distance)

    neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i, j in zip(*np.where(unknown)):
        row = int(ids[i, j])
        diagonal = 0.0
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            conductance = face_value(i, j, ni, nj)
            diagonal += conductance
            if unknown[ni, nj]:
                rows.append(row)
                cols.append(int(ids[ni, nj]))
                values.append(-conductance)
            elif inner[ni, nj]:
                rhs[row] += conductance
        rows.append(row)
        cols.append(row)
        values.append(diagonal)

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(rhs.size, rhs.size),
        dtype=float,
    ).tocsr()
    interior = np.asarray(spsolve(matrix, rhs), dtype=float)
    residual = matrix @ interior - rhs
    relative_residual = _safe_rel(float(np.linalg.norm(residual)), float(np.linalg.norm(rhs)))

    potential = np.zeros_like(eta)
    potential[inner] = 1.0
    potential[unknown] = interior
    inner_flux = 0.0
    outer_flux = 0.0
    energy = 0.0
    for i in range(nx):
        for j in range(ny):
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                if (inner[i, j] and inner[ni, nj]) or (outer[i, j] and outer[ni, nj]):
                    continue
                conductance = face_value(i, j, ni, nj)
                jump = potential[i, j] - potential[ni, nj]
                energy += conductance * jump * jump
                if inner[i, j] and unknown[ni, nj]:
                    inner_flux += conductance * jump
                elif unknown[i, j] and inner[ni, nj]:
                    inner_flux -= conductance * jump
                if unknown[i, j] and outer[ni, nj]:
                    outer_flux += conductance * jump
                elif outer[i, j] and unknown[ni, nj]:
                    outer_flux -= conductance * jump

    value = 0.5 * (inner_flux + outer_flux)
    scale = max(abs(value), abs(energy), 1.0e-30)
    flux_energy_defect = max(
        abs(inner_flux - outer_flux),
        abs(inner_flux - energy),
        abs(outer_flux - energy),
    ) / scale
    return CapacityResult(
        value=float(value),
        inner_flux=float(inner_flux),
        outer_flux=float(outer_flux),
        energy=float(energy),
        relative_residual=float(relative_residual),
        flux_energy_defect=float(flux_energy_defect),
        potential=potential,
    )


def activation_cv(
    activation: Array,
    spacing: float,
    axis: int,
    fit_interval: Tuple[float, float],
) -> Tuple[float, float, int]:
    """Regression conduction velocity in m/s (numerically equal to mm/ms)."""
    profile = np.nanmedian(activation, axis=1 - axis)
    distance = (np.arange(profile.size) + 0.5) * spacing
    mask = np.isfinite(profile) & (distance >= fit_interval[0]) & (distance <= fit_interval[1])
    if int(mask.sum()) < 3:
        return np.nan, np.nan, int(mask.sum())
    slope, intercept = np.polyfit(distance[mask], profile[mask], 1)
    fitted = slope * distance[mask] + intercept
    ss_res = float(np.sum((profile[mask] - fitted) ** 2))
    ss_tot = float(np.sum((profile[mask] - np.mean(profile[mask])) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    return float(1.0 / slope), float(r2), int(mask.sum())


def apd90(time: Array, trace: Array, activation_threshold: float = 0.5) -> float:
    above = np.flatnonzero(trace >= activation_threshold)
    if above.size == 0:
        return np.nan
    i0 = int(above[0])
    if i0 == 0:
        activation_time = float(time[0])
    else:
        fraction = (activation_threshold - trace[i0 - 1]) / max(
            trace[i0] - trace[i0 - 1], 1.0e-14
        )
        activation_time = float(time[i0 - 1] + fraction * (time[i0] - time[i0 - 1]))
    peak_index = i0 + int(np.argmax(trace[i0:]))
    peak = float(trace[peak_index])
    baseline = float(np.min(trace[: i0 + 1]))
    level = baseline + 0.1 * (peak - baseline)
    down = np.flatnonzero(trace[peak_index:] <= level)
    if down.size == 0:
        return np.nan
    j = peak_index + int(down[0])
    if j == 0:
        repolarisation_time = float(time[0])
    else:
        denominator = trace[j - 1] - trace[j]
        fraction = (trace[j - 1] - level) / max(denominator, 1.0e-14)
        repolarisation_time = float(time[j - 1] + fraction * (time[j] - time[j - 1]))
    return repolarisation_time - activation_time


def ring_lesion_reference(
    X: Array,
    Y: Array,
    center: Tuple[float, float],
    radius: float,
    width: float,
    gap_width: float,
    gap_angle: float = 0.0,
    transition: float = 0.35,
) -> Array:
    """Smooth +1 ablation ring with a -1 conducting gap."""
    dx = X - center[0]
    dy = Y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    radial = np.tanh((0.5 * width - np.abs(r - radius)) / transition)
    if gap_width <= 0.0:
        gap_mask = np.zeros_like(r)
    else:
        dtheta = np.angle(np.exp(1j * (theta - gap_angle)))
        half_angle = 0.5 * gap_width / radius
        edge = max(transition / radius, 1.0e-4)
        gap_mask = 0.5 * (1.0 - np.tanh((np.abs(dtheta) - half_angle) / edge))
    lesion = radial * (1.0 - gap_mask)
    # radial is near +1 inside the lesion band and -1 away; suppressing the
    # band in the gap must return the viable state -1.
    return np.where(gap_mask > 1.0e-8, (1.0 - gap_mask) * radial - gap_mask, radial)


def ring_gap_width(
    u: Array,
    X: Array,
    Y: Array,
    center: Tuple[float, float],
    radius: float,
    gap_angle: float = 0.0,
) -> float:
    """Estimate the viable arc containing ``gap_angle`` by interpolation."""
    ntheta = 4096
    th = gap_angle + np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
    xr = center[0] + radius * np.cos(th)
    yr = center[1] + radius * np.sin(th)
    dx = float(X[1, 0] - X[0, 0])
    dy = float(Y[0, 1] - Y[0, 0])
    profile = map_coordinates(
        u,
        np.vstack(((xr - float(X[0, 0])) / dx, (yr - float(Y[0, 0])) / dy)),
        order=1,
        mode="grid-wrap",
        prefilter=False,
    )
    viable = profile < 0.0
    i0 = ntheta // 2
    if not viable[i0]:
        return 0.0
    left = i0
    right = i0
    while left > 0 and viable[left - 1]:
        left -= 1
    while right + 1 < ntheta and viable[right + 1]:
        right += 1
    dtheta = 2.0 * np.pi / ntheta
    left_fraction = 0.0
    if left > 0:
        denominator = profile[left] - profile[left - 1]
        if abs(float(denominator)) > 1.0e-14:
            left_fraction = float(-profile[left - 1] / denominator)
    right_fraction = 1.0
    if right + 1 < ntheta:
        denominator = profile[right + 1] - profile[right]
        if abs(float(denominator)) > 1.0e-14:
            right_fraction = float(-profile[right] / denominator)
    angular_width = (
        right - left + 1.0 + right_fraction - left_fraction
    ) * dtheta
    return float(angular_width * radius)


def segmentation_metrics(pred: Array, truth: Array) -> Dict[str, float]:
    pb = pred > 0.0
    tb = truth > 0.0
    inter = float(np.logical_and(pb, tb).sum())
    union = float(np.logical_or(pb, tb).sum())
    denom = float(pb.sum() + tb.sum())
    return {
        "rmse": _rms(pred - truth),
        "dice": 2.0 * inter / max(denom, 1.0),
        "jaccard": inter / max(union, 1.0),
        "lesion_fraction_bias": float(np.mean(pb) - np.mean(tb)),
    }
