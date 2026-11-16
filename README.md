# cc-inclusive-final-repo: the three final figures from one self-contained script

Made 2026-11-16 (user: "reproduce the three plots in cc-inclusive final v5, don't change any, only make the script self contained,
don't rely on any other script outside the folder"). The figures, the numbers and the drawing are those of
`../cc-inclusive-final-v5` (2026-11-03); the only change is where the code lives: `scripts/final_figures.py` imports nothing but
numpy and matplotlib and reads every input from this folder's `data/`.

    python scripts/final_figures.py       # ~3 min; here: pixi run --manifest-path ../../ndp-platform/pixi.toml python scripts/final_figures.py

| output | content |
|---|---|
| `figures/cutflow_efficiency_purity.png` | signal efficiency and purity after each cut, GENIE full simulation vs GENIE + surrogate, with the ratio panel |
| `figures/selection_efficiency_purity_pT_pL.png` | efficiency vs true and purity vs reconstructed muon pT and pL, full simulation and the three surrogates |
| `figures/muon_pT_pL_1d.png` | selected events vs reconstructed muon pT and pL: data, full simulation, GENIE / NuWro / GiBUU + surrogate |
| `figures/final_figures.json` | every number drawn, the four universes and the components of each band |
| `data/inputs.json` | every input file read (size, date, link target); `data/run_2026-11-16.log` = the run's printout |

The samples, weights, normalisation, selection and uncertainties are described in the docstring of `scripts/final_figures.py`
(and, with the history of the choices, in `../cc-inclusive-final-v5/README.md`).

## Layout

    scripts/final_figures.py   the whole analysis from the tables to the figures (1,229 lines)
    scripts/link_inputs.py     puts the inputs in data/ (links by default, --copy for copies); imports only final_figures.INPUTS
    data/                      the 33 inputs (links to ../cc-inclusive/data), inputs.json, the run log
    figures/                   the three figures and final_figures.json
    configs/                   empty (nothing is configured outside the script)
    .gitignore                 keeps the inputs of data/ (links or copies, 2.8 GB) and Python/Jupyter leftovers out of the git repository

## Inputs (2.8 GB; `python scripts/link_inputs.py [SOURCE] [--copy]`, default SOURCE `../cc-inclusive/data`)

The tables are one row per event, extracted from the ROOT files by the scripts of `../cc-inclusive/scripts` named below. That
extraction is upstream of this folder; nothing here runs it.

| file in `data/` | MB | what | made by (cc-inclusive) |
|---|---|---|---|
| `minerva_data_selection.npz` | 89 | data, playlist 1A, reconstructed rows and POT | `extract_minerva_data_selection.py` |
| `minerva_genie_reco_selection.npz` | 455 | GENIE full simulation, 41 files, reconstructed rows and POT | `extract_minerva_reco_selection.py` |
| `minerva_genie_cc_fullvol_truth.npz`, `minerva_genie_other_fullvol_truth.npz` | 360, 96 | its Truth tree at z >= 4000 mm: numu CC, and every other species and current | `extract_minerva_truth_fullvol.py` |
| `minerva_genie_{cc,other}_fullvol_truth_enu.npz` | 78, 29 | true neutrino energy and species of those rows | `extract_minerva_truth_enu.py` |
| `minerva_me1A_numu_flux_mat.json` | 0.01 | playlist-1A flux with and without the nu-e constraint (MAT FluxReweighter) | flux dump from MAT |
| `genie_inclusive_v4_surrogate_{selection,universes,truth}.npz` | 102, 198, 180 | GENIE + ensemble surrogate: reconstructed rows, the four members' universes, the input population | `extract_surrogate_selection.py`, `extract_surrogate_universes.py`, `extract_surrogate_truth_population.py` |
| `nuwro_geo_surrogate_{selection,universes,truth}.npz` | 96, 226, 202 | NuWro geo sample + surrogate, the same three tables | the same, kind `nuwro_geo` |
| `gibuu_geo_surrogate_{selection,universes,truth}.npz` | 109, 255, 235 | GiBUU geo sample + surrogate, the same three tables | the same, kind `gibuu_geo` |
| `nuwro_inclusive_v5_surrogate_selection.npz` | 89 | October NuWro inclusive table: read only for the inferred f_in recorded in the JSON | `extract_surrogate_selection.py nuwro_inclusive_v5` |
| `<campaign>/nuwro_run.json` (16) | 0.08 | job sizes of the sixteen October NuWro campaigns, for the same f_in | `harvest_nuwro_campaign.py` |

The last two rows do not enter any figure: they feed the before/after record of the JSON key `genie_truth_tree`
(`f_in_inferred_before` and the ratios to it), kept so that the JSON is unchanged from version 5.

## What the script contains

`scripts/final_figures.py` = `../cc-inclusive-final-v5/scripts/final_figures.py` with its imports replaced by copies of the code:

| copied from | what |
|---|---|
| `cc-inclusive/scripts/datacomp_common.py` | `load_samples` (closure-test set), the nu-e constraint weight, the event-key joins, `gen_signal`, `efficiency` (here `dc_efficiency`), `cut_flow`, `reco_hists`, `chi2`, the JSON writer, `edge_axis`, labels |
| `cc-inclusive/scripts/fullvol_common.py` | volumes, `in_volume`, `hist`, `purity`, `tidy`, `truth_categories`, `nuwro_target_fraction`, the cut list |
| `cc-inclusive/scripts/compare_selection_nuwro_surrogate_vs_genie.py` | cut names, `passed`, the pT and pL edges, the signal's angle and pL limits |
| `cc-inclusive/scripts/datacomp_genie_nuwro_surrogate.py` | `compare` |
| `cc-inclusive/scripts/systematics_common.py` | `chi2_cov` |
| `cc-inclusive/scripts/cutflow_closure.py` | `stage_masks`, `epistemic_terms`, `stage_rows`, `compute` (here `closure_compute`) |
| `cc-inclusive/scripts/xsec_reco_2d.py`, `extract_surrogate_selection.py` | the generator names; the list of the sixteen NuWro campaigns |
| `ndp-platform/scripts/plot_style.py` + `docs/style_preferences.yaml` | the four colours (blue `#0072B2`, green `#009E73`, purple `#CC79A7`, NuWro `#a63603`); the rcParams block there is empty |

26 of the 47 functions are character-for-character the originals (2 are new: `load`, `read_json`). The others differ only in:

* reading: `load()` / `read_json()` on `data/<name>` (recording the file for `data/inputs.json`) instead of `np.load` on
  `cc-inclusive/data/<name>`; the GENIE surrogate table is named directly (`genie_inclusive_v4_...`) instead of through the
  `genie_inclusive_v6 -> v5 -> v4` links;
* the sample-set switches resolved to the one version 5 ran with (`inclusive-v6`: every species, six cuts, three surrogates), and the
  branches of the other sets and of the rate-matched NuWro/GiBUU samples left out: `load_samples` keeps the closure-test path,
  `gen_signal` / `dc_efficiency` the Truth-tree denominator; no `NDP_SURROGATE_SET`;
* the K factor of the muon figure (an empty dictionary in version 5, since the geo samples carry their own normalisation) written as
  such, so `xsec_reco_2d.generator_factor` is not needed;
* `record_inputs` lists the files of `data/` and no longer creates links (that is `link_inputs.py`); a check at start that the
  33 inputs are there;
* docstrings and error messages.

## Check against version 5 (2026-11-16)

Run of 2026-11-16 (2 min 54 s; Python 3.14.7, numpy 2.5.2, matplotlib 3.11.1, the pixi default environment of ndp-platform), the 33
input files unchanged since the version-5 run (sizes and dates of `../cc-inclusive-final-v5/data/inputs.json`):

* the three PNG files are byte-identical to those of `../cc-inclusive-final-v5/figures`
  (md5 `2d79b83e...` cut flow, `65f4117c...` efficiency/purity, `da0f4562...` muon pT/pL);
* `figures/final_figures.json`: of its 18,042 values, 12 differ, all of them the paths of the geo tables under `geo_samples/*/tables`
  (now relative, `data/...`, instead of the absolute `cc-inclusive/data/...`); every number is equal;
* the 126 lines of numbers printed by the run are identical to `../cc-inclusive-final-v5/data/run_2026-11-03.log`;
* the modules loaded from outside the Python environment are `scripts/final_figures.py` and nothing else.
