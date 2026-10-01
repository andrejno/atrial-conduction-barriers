# Attribution for the patient-derived data

Experiment 6 uses data from:

Martínez Díaz, Patricia; Goetz, Christian; Dasi, Albert; Unger, Laura Anna; Haas, Annika;
Dössel, Olaf; Luik, Armin; and Loewe, Axel (2024). **Atrial Models with Personalized
Effective Refractory Period**, version 1.0. Zenodo.
[DOI: 10.5281/zenodo.10726677](https://doi.org/10.5281/zenodo.10726677).

The source dataset is distributed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
This attribution accompanies the source-derived data redistributed with this package.
The source authors do not endorse the present reconstruction method or its conclusions.

The corresponding clinical and modelling study is Martínez Díaz *et al.* (2024),
**Impact of effective refractory period personalization on arrhythmia vulnerability in
patient-specific atrial computer models**, *Europace*, 26(10), euae215.
[DOI: 10.1093/europace/euae215](https://doi.org/10.1093/europace/euae215).

## Files used and changes made

The analysis uses P1 and P3–P7. P2 is excluded because its mesh is a right atrium.
The package bundles nineteen original source members under
`external_data/zenodo_erp/meshes/`: each selected case's simple VTK surface and annotated
bilayer point/element files, together with the source region-tag table. These members
retain the source dataset's CC BY 4.0 licence. The fetcher can verify or restore this
subset against the pinned official archive. The full 1.6-GB raw archive is not bundled.

The included code makes the following documented changes:

- Converts the legacy surface representation to triangles, splitting quadrilaterals
  along the shorter diagonal while preserving source-cell provenance.
- Removes one degenerate triangle from P4. The cleaning report records this explicitly;
  near-coincident points are not merged.
- Transfers anatomical labels from the annotated bilayer representation to the simple
  surface's five boundary loops, with transfer distances retained in the output table.
- Uses geometry-based voxel subsampling to create artificial sparse contacts and
  prespecified left/right perivenous blackouts.
- Transforms clinical bipolar-voltage values to the threshold-anchored score
  `b(V) = 2^(-V/0.1 mV)` and applies screened, passive, and graph reconstruction.
- Computes withheld-region errors, score-calibration summaries, model-derived capacity,
  and viable boundary-arc measures, and produces the corresponding figure.

The bundled original members are retained as downloaded; the transformations above are
applied by the analysis code and stored separately in derived outputs.

`data/zenodo_pvi_representative.npz` redistributes source-derived P3 anatomy and voltage
information, together with transformed scores, masks, contact selections, and computed
reconstructions. The patient result tables and Figure 6 also derive from the source
dataset. These are modified analytical outputs, not a replacement for the original
clinical data release. Keep this attribution and the licence link with redistributed
copies of those source-derived materials.

`data/zenodo_pvi_provenance.json`, `data/zenodo_pvi_mesh_inventory.csv`, and
`data/zenodo_pvi_boundary_transfer.csv` document the source hashes and processing.
`data/zenodo_pvi_contact_geometry.csv` records the contact-support locality checks.
The source maps are interpolated clinical voltage references; the present package does
not turn them into independent measurements of electrical block or recurrence outcomes.

## Added outputs in the Journal of Scientific Computing revision

The derived arrays and tables under `data/patient_extension/` retain the same source attribution. They add geometry-defined missingness controls and discretisation checks without acquiring additional patient measurements. Midpoint refinement subdivides the existing piecewise-planar surfaces and prolongs the original available fields.

The `data/patient_surface_ep_*` files and `figures/patient_surface_ep.*` use source-derived P3 anatomy and voltage scores as inputs to an additional idealised forward model. They are derived simulation outputs, not measured activation maps. Figure 6b and its stored fields are likewise modified analytical outputs. The original patient source members remain unchanged.
