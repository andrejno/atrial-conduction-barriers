"""Conservative P1 surface monodomain readout with fixed isotropic diffusion.

No vertex clipping, stiffness modification, fibre inference or fitted kinetics
is performed.  Lumped mass and the intrinsic P1 stiffness are used in an IMEX
Euler voltage step.  The gate is integrated exactly for the new frozen voltage.
Natural zero flux holds on every opening.  Obtuse meshes need not satisfy a
discrete maximum principle; positive off-diagonal entries and state extrema are
reported explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from model import EPParameters, _gate_coefficients, diffusivity_factor
from surface_fem import assemble_p1
from surface_mesh import TriangleMesh

@dataclass
class SurfaceEPSolution:
    v: np.ndarray
    h: np.ndarray
    activation: np.ndarray
    snapshots: dict[float, np.ndarray]
    diagnostics: dict[str, float]


def solve_surface_ep(mesh: TriangleMesh, signed_score: np.ndarray, *, dt: float=0.05,
                     t_end: float=210., d0: float=.128,
                     stimulus_mask: np.ndarray | None=None,
                     stimulus_amplitude: float=1.2, stimulus_duration: float=2.,
                     snapshot_times: tuple[float,...]=(25.,50.,100.,150.,210.),
                     v0: np.ndarray | None=None, h0: np.ndarray | None=None,
                     diffusion_only: bool=False,
                     constant_diffusivity: bool=False,
                     nodal_diffusivity: np.ndarray | None=None,
                     forcing: Callable[[float],np.ndarray] | None=None) -> SurfaceEPSolution:
    start=perf_counter()
    n=mesh.n_vertices
    score=np.asarray(signed_score,float)
    if score.shape!=(n,) or not np.isfinite(score).all():
        raise ValueError('signed_score must be finite with one value per vertex')
    steps=int(round(t_end/dt))
    if not np.isclose(steps*dt,t_end,atol=1e-10,rtol=0):
        raise ValueError('t_end must be divisible by dt')
    par=EPParameters(dt=dt)
    coefficient=np.full(n,d0) if constant_diffusivity else d0*diffusivity_factor(score)
    if nodal_diffusivity is not None:
        coefficient=np.asarray(nodal_diffusivity,float)
        if coefficient.shape!=(n,) or np.any(coefficient<=0):
            raise ValueError('nodal_diffusivity must be positive with one value per vertex')
    operators=assemble_p1(mesh,coefficient,coefficient_location='node')
    area=operators.vertex_area; K=operators.stiffness
    A=(diags(area)+dt*K).tocsc(); factor=splu(A)
    setup=perf_counter()-start
    v=np.zeros(n) if v0 is None else np.array(v0,dtype=float,copy=True)
    h=np.ones(n) if h0 is None else np.array(h0,dtype=float,copy=True)
    activation=np.full(n,np.nan)
    activation[v>=par.activation_threshold]=0.
    stim=np.zeros(n) if stimulus_mask is None else stimulus_amplitude*np.asarray(stimulus_mask,float)
    requested=sorted(snapshot_times); snapshots={}; index=0
    extrema=[float(v.min()),float(v.max()),float(h.min()),float(h.max())]
    max_residual=max_mass=0.
    solve_start=perf_counter()
    for step in range(steps):
        t=step*dt; old=v
        reaction=np.zeros(n) if diffusion_only else h*v*v*(1-v)/par.tau_in-v/par.tau_out
        source=reaction + (stim if t<stimulus_duration-1e-12 else 0.)
        if forcing is not None: source=source+forcing(t)
        rhs=area*(v+dt*source)
        v=factor.solve(rhs)
        if not diffusion_only:
            gateA,gateB=_gate_coefficients(v,par)
            equilibrium=gateA/gateB
            h=equilibrium+(h-equilibrium)*np.exp(-gateB*dt)
        crossing=np.isnan(activation)&(old<par.activation_threshold)&(v>=par.activation_threshold)
        fraction=np.clip((par.activation_threshold-old)/np.maximum(v-old,1e-30),0,1)
        activation[crossing]=t+dt*fraction[crossing]
        extrema=[min(extrema[0],float(v.min())),max(extrema[1],float(v.max())),
                 min(extrema[2],float(h.min())),max(extrema[3],float(h.max()))]
        if step%20==0 or step==steps-1:
            residual=A@v-rhs
            max_residual=max(max_residual,float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-30)))
            max_mass=max(max_mass,abs(float(area@(v-old-dt*source)))/operators.total_area)
            if not np.isfinite(v).all() or not np.isfinite(h).all():
                raise FloatingPointError(f'nonfinite EP state at {t+dt} ms')
        while index<len(requested) and t+dt>=requested[index]-dt/2:
            snapshots[requested[index]]=v.copy(); index+=1
    sparse=K.tocoo(); off=sparse.row!=sparse.col
    diagnostics={'dt_ms':dt,'t_end_ms':t_end,'d0_mm2_per_ms':d0,
                 'setup_seconds':setup,'advance_seconds':perf_counter()-solve_start,
                 'wall_seconds':perf_counter()-start,'n_vertices':n,'n_triangles':mesh.n_triangles,
                 'linear_factorizations':1,'linear_solves':steps,
                 'v_min':extrema[0],'v_max':extrema[1],'h_min':extrema[2],'h_max':extrema[3],
                 'maximum_sampled_relative_linear_residual':max_residual,
                 'maximum_sampled_mass_balance_defect':max_mass,
                 'positive_offdiagonal_fraction':float(np.mean(sparse.data[off]>1e-12)),
                 'positive_offdiagonal_maximum':float(np.maximum(sparse.data[off],0).max()),
                 'activation_area_fraction':float(area@np.isfinite(activation)/area.sum()),
                 'stimulus_area_mm2':float(area@(stim>0)),
                 'minimum_vertex_area_mm2':float(area.min())}
    if not diffusion_only and (extrema[2]<-1e-12 or extrema[3]>1+1e-12):
        raise FloatingPointError('gate left invariant interval')
    return SurfaceEPSolution(v,h,activation,snapshots,diagnostics)
