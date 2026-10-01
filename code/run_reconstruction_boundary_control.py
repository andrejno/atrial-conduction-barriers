"""Boundary-coverage control of the Experiment 4 nested-blackout extension.

This control was designed after inspecting the support-exclusion sweep. It is
reported separately, not as a prespecified replacement of that sweep. Contact
centres outside the buffered missing sector are retained. Their observation
kernels are truncated to zero in the sector, as in the original solver. Thus
additional *exterior* data are admitted without observing any hidden score.
All five geometries, four rotations and three sizes are included without tuning.
"""
from __future__ import annotations

import json
import numpy as np
import run_reconstruction_extension as r

r.OUT = r.ROOT / "data" / "reconstruction_boundary_fields"
r.a.DATA = r.ROOT / "data" / "reconstruction_boundary_control"
r.a.FIG = r.ROOT / "figures" / "reconstruction_boundary_control"
r.a.DATA.mkdir(parents=True, exist_ok=True)
r.a.FIG.mkdir(parents=True, exist_ok=True)
r.DESIGN = {
    **r.DESIGN,
    "acquisition_control": "Boundary coverage: retain exterior contact centres, truncate all kernels inside buffered hidden region.",
    "design_status": "Separate acquisition control devised after inspecting the first support-exclusion sweep; frozen before its own evaluation.",
    "excluded_contact_region": "Nominal arc enlarged by 1-mm guard, radial half-width 3mm; no additional kernel-support exclusion.",
    "refinement_cases": [[g,w,angle] for g,angle in [("complete_ring",0),("narrow_gap",0),("wide_gap",90)] for w in [2,6]],
}


def boundary_contacts(spec, seed, width, angle):
    rng = np.random.default_rng(seed)
    theta = np.linspace(-np.pi, np.pi, 260, endpoint=False)
    radius = r.a.RADIUS + rng.normal(0, .55, theta.size)
    x = np.r_[r.a.CENTER[0] + radius*np.cos(theta), rng.uniform(0, r.a.LENGTH, 320)]
    y = np.r_[r.a.CENTER[1] + radius*np.sin(theta), rng.uniform(0, r.a.LENGTH, 320)]
    exact = r.a.geometry_score_field(x, y, spec["gaps"])
    noise = rng.normal(0, r.a.CONTACT_NOISE_SD, x.size)
    radius = np.hypot(x-r.a.CENTER[0], y-r.a.CENTER[1])
    theta = np.arctan2(y-r.a.CENTER[1], x-r.a.CENTER[0])
    arc = r.a.RADIUS*np.abs(r.a._wrapped_angle(theta-np.deg2rad(angle)))
    hidden = (np.abs(radius-r.a.RADIUS)<=r.a.LESION_WIDTH/2+r.a.HOLDOUT_BUFFER) & (arc<=width/2+r.a.HOLDOUT_BUFFER)
    keep = ~hidden
    return {"train_x":x[keep],"train_y":y[keep],"train_score":(exact+noise)[keep],
            "train_exact":exact[keep],"n_training":int(keep.sum()),"n_ring_contacts":int(keep[:260].sum()),
            "noise_rms":float(np.sqrt(np.mean(noise[keep]**2))),
            "coordinate_sha256":r.digest(np.column_stack((x[keep],y[keep]))),
            "noise_sha256":r.digest(noise[keep]),
            "pre_exclusion_coordinate_sha256":r.digest(np.column_stack((x,y))),
            "contact_indices":np.flatnonzero(keep)}


r.contacts = boundary_contacts

if __name__ == "__main__":
    r.main()
