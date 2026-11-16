#!/usr/bin/env python
"""The three final figures of the CC-inclusive surrogate study, as one self-contained script (2026-11-16).

    python scripts/final_figures.py          # ~2 min; needs numpy and matplotlib only
    (here: pixi run --manifest-path ../../ndp-platform/pixi.toml python scripts/final_figures.py)

This is analysis_village/cc-inclusive-final-v5/scripts/final_figures.py (2026-11-03) with everything it imported copied in: the
functions and constants of analysis_village/cc-inclusive/scripts (datacomp_common, fullvol_common,
compare_selection_nuwro_surrogate_vs_genie, datacomp_genie_nuwro_surrogate, systematics_common, cutflow_closure, xsec_reco_2d,
extract_surrogate_selection) and the colours of ndp-platform/scripts/plot_style.py. Nothing outside this folder is imported, and every
input is read from this folder's data/ (scripts/link_inputs.py puts the tables there; INPUTS below lists them). The numbers and the
drawing are those of version 5: the code copied in is limited to what version 5 executes (sample set "inclusive-v6": every species and
both currents, six cuts, three surrogates), with the switches for the other sample sets resolved and their branches left out.

Writes figures/cutflow_efficiency_purity.png, figures/selection_efficiency_purity_pT_pL.png, figures/muon_pT_pL_1d.png,
figures/final_figures.json (every number drawn, the universes and the components) and data/inputs.json (every input read).

Samples (one row per reconstructed event; tables made by the extraction scripts of analysis_village/cc-inclusive):
    data   MINERvA playlist 1A, all 253 files (data/minerva_data_selection.npz); exposure = sum of Meta.POT_Used.
    full   GENIE official MC with the full detector simulation, all 41 files, runs 110000-110040 (data/minerva_genie_reco_selection.npz;
           generated events: the Truth-tree tables data/minerva_genie_{cc,other}_fullvol_truth.npz). Row weight = `wgt`, the
           production's PPFX flux central-value weight, x MINERvA's nu-e flux-constraint weight at the true neutrino energy
           (data/minerva_me1A_numu_flux_mat.json), so the sample follows the published flux. Scale to the data: POT_data / POT_41.
    gsur   GENIE + surrogate: the producer's four-member ensemble surrogate run on a share of the official MC's generated events of
           runs 110000-110029 with the true vertex at z >= 4000 mm, every species and both currents
           (data/genie_inclusive_v4_surrogate_{selection,universes,truth}.npz). Row weight = the event's official weight joined by
           event key x the nu-e weight at the true neutrino energy. The generated population is the surrogate file's own Truth tree
           (every event the surrogate was run on): the efficiency denominator is the weighted sum over its signal rows, the exposure
           its row share f_T of the official 30-run population of the volume times POT_30 (use_genie_truth_tree).
    nsur, bsur   NuWro + surrogate and GiBUU + surrogate on the geo samples (analysis_village/minerva-geo-samples: generated with
           the MINERvA GDML vertices and the published flux x each generator's own cross section per nuclide;
           data/{nuwro,gibuu}_geo_surrogate_{selection,universes,truth}.npz). Row weight = gen_weight (events at the file's gen_pot),
           scale POT_data / gen_pot: the generator's own cross section and composition, no composition weight, no rate match, no K
           factor. The efficiency denominator is the file's Truth tree, the sum of gen_weight over its signal rows (load_geo).
    Background completion: the surrogate samples hold events generated at z >= 4000 mm only; the selected full-simulation rows
    generated outside that volume (13.5 events) are added to every surrogate prediction at the full simulation's scale.

Signal = the MINERvA 101 tutorial's truth definition (numu CC, true vertex z 5980-8422 mm in the 850 mm hexagon, theta_mu <= 20 deg,
p_L >= 1.5 GeV/c about the beam axis): the `truth` bits of the tables, all six set. Selection = the tutorial's six reconstruction
cuts in its order (CUTS), for the data, the full simulation and the surrogates alike. Binning: the paper's (arXiv:2106.16210) pT and pL
edges. Everything is shown at the data exposure.

Uncertainties. Statistical: binomial on the effective count (efficiency, purity), sum of squared weights (events). Surrogate model
uncertainty: half the spread (max - min)/2 over the four ensemble universes, taken as 1 sigma. Universe k = one member for every row
at once ("all together"): the Tracker, Apothem and angle cuts re-evaluated on member k's own reconstructed vertex and muon angle, the
reconstructed pT and pL from member k's own muon momentum and angle, and every row weighted by surrogate_univ_weight[k] = the product
of the three flag factors univ_weight_reco x _minos x _charge (member k's over the nominal member's probability of the row's
reconstruction, MINOS-match and charge-sign outcomes); the MINOS-match, dead-time and charge bits and the row set stay the nominal
ones. The JSON also gives the half spread with one part of the universe at a time (COMPONENTS); the parts are correlated and do not
add in quadrature to the total. Not covered: any misfit shared by the four members (measured by the closure test, the residual panel
of the cut-flow figure); four members give a band, not a covariance.

The inferred fraction f_in (the GENIE surrogate's input share deduced from the NuWro composition targets of the October campaigns,
which the Truth tree replaced) is still computed, only to record the before/after comparison in the JSON (key genie_truth_tree):
that is what data/nuwro_inclusive_v5_surrogate_selection.npz and the sixteen data/<campaign>/nuwro_run.json are read for.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
DATA = FINAL / "data"
OUT = FINAL / "figures"
SAMPLE_SET = "inclusive-v6"                      # the cc-inclusive sample set this script is the frozen copy of (recorded in the JSON)

# ---------------- the inputs, all in data/
G_RECO = DATA / "minerva_genie_reco_selection.npz"                 # GENIE full simulation, reconstructed rows of the 41 files
G_TRUTH = DATA / "minerva_genie_cc_fullvol_truth.npz"              # its Truth tree, numu CC of the full volume (+ the _enu companion)
G_TRUTH_OTHER = DATA / "minerva_genie_other_fullvol_truth.npz"     # its Truth tree, every other species and current (+ the _enu companion)
D_RECO = DATA / "minerva_data_selection.npz"                       # data, playlist 1A
NUE_FLUX = DATA / "minerva_me1A_numu_flux_mat.json"                # MAT FluxReweighter fluxes of playlist 1A: PPFX with and without the nu-e constraint
GS_RECO = DATA / "genie_inclusive_v4_surrogate_selection.npz"      # GENIE + surrogate (ensemble), reconstructed rows
GENIE_TRUTH = DATA / "genie_inclusive_v4_surrogate_truth.npz"      # the GENIE surrogate's input population (its file's Truth tree)
N_RECO = DATA / "nuwro_inclusive_v5_surrogate_selection.npz"       # October NuWro inclusive table: only for the inferred f_in recorded in the JSON
TABLE = {"gsur": GS_RECO}
GEO = {"nsur": "nuwro_geo", "bsur": "gibuu_geo"}                         # version 5: the surrogates taken from the geo-sample files
GEO_TABLE = {s: {q: DATA / f"{g}_surrogate_{q}.npz" for q in ("selection", "universes", "truth")} for s, g in GEO.items()}
GEO_TEXT = {"nsur": "NuWro geo sample + surrogate", "bsur": "GiBUU geo sample + surrogate"}
NUCLIDE_Z = {"H1": 1, "C12": 6, "N14": 7, "O16": 8, "Al27": 13, "Si28": 14, "Cl35": 17, "Ti48": 22, "Fe56": 26, "Pb207": 82}
# campaigns of the October NuWro inclusive file: (mc_run base, campaign directory, beam PDG, mc_current); the table's `campaign` column is the index into this list
INCLUSIVE_CAMPAIGNS = [(900000, "nuwro_me_fhc_tracker_2026-10", 14, 1), (910000, "nuwro_me_fhc_targets_pbfe_2026-10", 14, 1),
                       (920000, "nuwro_me_fhc_tracker_nc_2026-10", 14, 2), (930000, "nuwro_me_fhc_targets_pbfe_nc_2026-10", 14, 2)]
for _i, (_vol, _sp, _cur) in enumerate([(v, s, c) for v in ("tracker", "targets_pbfe") for s in ("numubar", "nue", "nuebar") for c in ("cc", "nc")]):
    INCLUSIVE_CAMPAIGNS.append((941000 + 1000 * _i, f"nuwro_me_fhc_{_vol}_{_sp}_{_cur}_2026-10", {"numubar": -14, "nue": 12, "nuebar": -12}[_sp], 1 if _cur == "cc" else 2))
INPUTS = [G_RECO, G_TRUTH, G_TRUTH.with_name(G_TRUTH.stem + "_enu.npz"), G_TRUTH_OTHER, G_TRUTH_OTHER.with_name(G_TRUTH_OTHER.stem + "_enu.npz"), D_RECO, NUE_FLUX,
          GS_RECO, GS_RECO.with_name(GS_RECO.name.replace("_surrogate_selection.npz", "_surrogate_universes.npz")), GENIE_TRUTH, N_RECO,
          *(p for s in GEO for p in GEO_TABLE[s].values()), *(DATA / name / "nuwro_run.json" for _base, name, _pdg, _current in INCLUSIVE_CAMPAIGNS)]

# ---------------- every input file read is recorded (for data/inputs.json)
READ: set[Path] = set()


def load(path: Path):
    """np.load of an input table."""
    READ.add(Path(path))
    return np.load(path)


def read_json(path: Path):
    READ.add(Path(path))
    return json.loads(Path(path).read_text())


# ---------------- selection, volumes, binning, labels (cc-inclusive: compare_selection_nuwro_surrogate_vs_genie, fullvol_common, datacomp_common, xsec_reco_2d)
CUTS = ("Tracker", "Apothem", "Muon Theta < 20", "Has MINOS Match", "Deadtime", "Muon Charge Sign")   # bit i of a table's `cuts` = CUTS[i] passed
CUT_LIST = (0, 1, 2, 3, 4, 5)                    # the cuts applied, in the tutorial's order
COMMON_BITS = 0b111111                           # Tracker, Apothem, Muon Theta < 20, Has MINOS Match, Deadtime, Muon Charge Sign
CUTS_TEXT = "six cuts (dead-time cut included)"
DEADTIME_BIT = 0b010000
THETA_MAX, PL_MIN = 20.0 * np.pi / 180.0, 1.5
PT_PAPER = np.array([0, 0.075, 0.15, 0.25, 0.325, 0.4, 0.475, 0.55, 0.7, 0.85, 1, 1.25, 1.5, 2.5, 4.5])      # the paper's edges (arXiv:2106.16210; the tutorial uses the same)
PL_PAPER = np.array([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 40, 60])
Z_FULL, APOTHEM_FULL = (4000.0, 8602.0), None    # mm, the volume of the surrogate samples: z >= 4000 mm, no transverse cut
Z_TUT, APOTHEM_TUT = (5980.0, 8422.0), 850.0     # mm, the tutorial's volume
SURROGATES = ("gsur", "nsur", "bsur")
LABEL = {"data": "MINERvA data, playlist 1A", "full": "GENIE, full simulation", "gsur": "GENIE + surrogate", "nsur": "NuWro + surrogate", "bsur": "GiBUU + surrogate",
         "bkg": "backgrounds from the full simulation\n(events generated outside the volume)"}
WEIGHT_TEXT = "GENIE: PPFX flux CV weight (`wgt`); NuWro: composition weight; GiBUU: composition × over-ceiling event weight / campaign mean over-ceiling weight"
GENERATOR = {"nsur": "nuwro", "bsur": "gibuu"}   # the surrogates that carry their own generator's cross section
GEN_NAME = {"nuwro": "NuWro", "gibuu": "GiBUU"}
STAGES = ["reconstructed event"] + [CUTS[i] for i in CUT_LIST]
K_PRE = CUT_LIST.index(2) + 1          # stage after Tracker, Apothem, angle: the rows that face the MINOS-match decision
K_MINOS = CUT_LIST.index(3) + 1        # stage after the MINOS match
K_CHARGE = CUT_LIST.index(5) + 1       # stage after the charge-sign cut (the last)

SUR = SURROGATES
OFFSETS = {2: (-0.17, 0.17), 3: (-0.25, 0.0, 0.25)}[len(SUR)]        # side-by-side bars of the surrogates in the ratio panels
STAGE_LABEL = {"Tracker": "fiducial volume z", "Apothem": "fiducial volume xy"}   # cut-flow tick labels for the tutorial's vertex cuts (user request 2026-10-28); the JSON keeps the tutorial names in cut_stages


# ---------------- the ensemble universes (version 2, 2026-10-28; version 3, 2026-10-29: with the producer's universe weight): the surrogate's
# model uncertainty = half the spread over the four members
K_MEMBERS = 4
VALUE_BITS = np.uint8(0b000111)                  # Tracker, Apothem, Muon Theta < 20: the cuts with a per-member value
VERTEX_BITS, ANGLE_BIT = np.uint8(0b000011), np.uint8(0b000100)   # version 4: the vertex cuts and the angle cut separately
COMPONENTS = ["reco", "minos", "charge", "flags", "vertex", "muon", "kinematics", "all"]   # version 4: the parts of the universe, in the JSON's order


# ================= helpers copied from analysis_village/cc-inclusive/scripts (2026-11-16) =================

def passed(cuts: np.ndarray, i: int) -> np.ndarray:
    return ((cuts >> i) & 1).astype(bool)


def in_hexagon(x, y, apothem):
    ax, ay = np.abs(x), np.abs(y)
    return (ax < apothem) & (ay < (2.0 * apothem - ax) / np.sqrt(3.0))


def in_volume(v, z_range=Z_FULL, apothem=APOTHEM_FULL):
    m = (v[:, 2] >= z_range[0]) & (v[:, 2] <= z_range[1])
    return m if apothem is None else m & in_hexagon(v[:, 0], v[:, 1], apothem)


def hist(x, edges, w=None):
    return np.histogram(x, edges, weights=w)[0]


def tidy(ax):
    ax.grid(True, axis="y", color="0.88", linewidth=0.8, linestyle="-"); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def style():
    """Colours and line styles; data = black points. The colours are those of ndp-platform/scripts/plot_style.py as the figures were
    made (Okabe-Ito blue, green and purple; the mc_signal role colour for NuWro); matplotlib's rcParams are left at their defaults."""
    col = {"full": "#0072B2", "gsur": "#009E73", "nsur": "#a63603", "bsur": "#CC79A7", "data": "0.05", "bkg": "0.55"}
    line = {"full": dict(linewidth=2.2), "gsur": dict(linewidth=1.6, linestyle=(0, (4, 2))), "nsur": dict(linewidth=1.7), "bsur": dict(linewidth=1.7, linestyle=(0, (6, 1.5, 1.5, 1.5))),
            "bkg": dict(linewidth=0.9)}
    return col, line


def event_key(run, subrun, nth) -> np.ndarray:
    """One int64 per official-MC event."""
    return (run.astype(np.int64) * 100_000 + subrun.astype(np.int64)) * 10_000_000 + nth.astype(np.int64)


def join_weight(gt, run, subrun, nth) -> np.ndarray:
    """Truth-tree cvweight of the events (run, subrun, nth); every event must be in the table."""
    return join_truth_column(gt, gt["cvweight"], run, subrun, nth).astype(float)


def join_truth_column(gt, column: np.ndarray, run, subrun, nth) -> np.ndarray:
    """`column` (one value per row of the Truth table `gt`) for the events (run, subrun, nth); every event must be in the table."""
    keys = event_key(gt["run"], gt["subrun"], gt["nth"])
    order = np.argsort(keys, kind="stable")
    k_sorted = keys[order]
    q = event_key(run, subrun, nth)
    i = np.searchsorted(k_sorted, q)
    i = np.minimum(i, len(k_sorted) - 1)
    found = k_sorted[i] == q
    if not found.all():
        raise SystemExit(f"{(~found).sum()} of {len(q)} events are not in the Truth table")
    return column[order[i]]


def nue_constraint_weight(enu_gev: np.ndarray, pdg: np.ndarray) -> np.ndarray:
    """MINERvA's nu-e flux-constraint weight per event: the ratio of the constrained to the unconstrained PPFX flux of
    playlist 1A (NUE_FLUX, dumped from MAT's FluxReweighter) at the true neutrino energy. As in MAT's
    FluxReweighter::GetFluxCVWeight, pdg > 0 takes the numu ratio and pdg < 0 the numubar ratio (nue/nuebar events get the
    muon-flavour ratio of the same sign), and an energy outside the flux histogram gets 1. The official MC's `wgt` is the
    PPFX weight only (verified 2026-10-14: events / (PPFX flux x nucleons x POT) reproduces GENIE's own cross section to
    1-2 %), so multiplying it by this weight distributes the simulation as the published, constrained flux, which is the
    flux the data were taken with and the one the cross-section formula divides by (6.230e-4 nu/m^2/POT; PPFX alone is
    6.917e-4). Event-averaged over tracker numu CC the weight is 0.900."""
    f = read_json(NUE_FLUX)
    edges = np.asarray(f["edges_gev"])
    w = np.ones(len(enu_gev))
    for sign, nu, con in ((1, "ppfx", "ppfx_nue_constrained"), (-1, "numubar_ppfx", "numubar_ppfx_nue_constrained")):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.asarray(f[nu]) > 0, np.asarray(f[con]) / np.asarray(f[nu]), 1.0)
        m = (np.sign(pdg) == sign) & (enu_gev >= edges[0]) & (enu_gev < edges[-1])
        w[m] = r[np.searchsorted(edges, enu_gev[m], side="right") - 1]
    return w


def truth_enu(table: Path, n_rows: int, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(E_nu in GeV, pdg) for every row of a Truth table, from its <table>_enu.npz companion (cc-inclusive
    extract_minerva_truth_enu.py); the companion's row count and sampled keys must match the table."""
    comp = table.with_name(table.stem + "_enu.npz")
    if not comp.exists():
        raise SystemExit(f"{comp} is missing: the nu-e constraint needs the Truth table's _enu companion")
    z = load(comp)
    if int(z["n_rows"]) != n_rows or not np.array_equal(z["key_sample"], keys[:: int(z["key_stride"])]):
        raise SystemExit(f"{comp.name} does not match {table.name} row for row")
    return z["enu_gev"].astype(float), z["pdg"]


def truth_constraint_weight(table: Path, n_rows: int, keys: np.ndarray) -> np.ndarray:
    """nue_constraint_weight for every row of a Truth table (truth_enu)."""
    return nue_constraint_weight(*truth_enu(table, n_rows, keys))


def truth_categories(gt, go, runs=None) -> dict[tuple[int, int, int], int]:
    """Truth events of V_FULL per (mc_incoming, mc_current, mc_targetZ), from the numu CC table and the table of the rest."""
    out: dict[tuple[int, int, int], int] = {}
    m = in_volume(gt["vtx"]) & (True if runs is None else np.isin(gt["run"], runs))
    for z, c in zip(*np.unique(gt["target_Z"][m], return_counts=True)):
        out[(14, 1, int(z))] = int(c)
    m = in_volume(go["vtx"]) & (True if runs is None else np.isin(go["run"], runs))
    key = np.stack([go["incoming"][m], go["current"][m], go["target_Z"][m]], axis=1)
    for k, c in zip(*np.unique(key, axis=0, return_counts=True)):
        out[tuple(int(x) for x in k)] = int(c)
    return out


def nuwro_target_fraction(n, truth30: dict) -> tuple[float, dict]:
    """The composition target of the October NuWro inclusive file relative to the official MC: for every (campaign, nuclide) whose
    nuwro_wcomp is not 1, target = wcomp x (events of the nuclide in the campaign) over the Truth count of the same
    species, current and nuclide. Returns the CC value (one number; its spread is checked) and every category's ratio.
    This is the inferred f_in of the GENIE surrogate's input, which the surrogate file's Truth tree replaced (use_genie_truth_tree)."""
    ratios, cc = {}, []
    for c, (_base, name, pdg, current) in enumerate(INCLUSIVE_CAMPAIGNS):
        m_c = n["campaign"] == c
        if not m_c.any():
            continue
        available: dict[int, int] = {}
        for j in read_json(DATA / name / "nuwro_run.json")["jobs"].values():
            available[NUCLIDE_Z[j["label"]]] = available.get(NUCLIDE_Z[j["label"]], 0) + int(j["n_events"])
        for z in np.unique(n["mc_targetZ"][m_c]):
            u = np.unique(n["wcomp"][m_c & (n["mc_targetZ"] == z)])
            if len(u) != 1:
                raise SystemExit(f"nuwro_wcomp takes {len(u)} values for Z = {z} of {name}; one per campaign and nuclide expected")
            if abs(u[0] - 1) > 1e-9:
                r = float(u[0]) * available[int(z)] / truth30[(pdg, current, int(z))]
                ratios[f"{name}:Z{int(z)}"] = {"wcomp": float(u[0]), "campaign_events": available[int(z)], "truth_events_30_runs": truth30[(pdg, current, int(z))], "target_over_truth": r}
                if current == 1:
                    cc.append(r)
    if not cc or np.ptp(cc) > 1e-9:
        raise SystemExit(f"the CC composition targets of the NuWro file are not one fraction of the Truth counts: {cc}")
    return float(np.mean(cc)), ratios


def load_samples(nue_constraint: bool = True) -> tuple[dict, dict, dict]:
    """(S, D, info) of the closure-test set: the GENIE full simulation and GENIE + surrogate as uniform dicts (key "full" and "sur"),
    the data, and the normalisation, counts and checks. Every row weight `w` is the generator weight; `scale` takes a weighted count
    to the data exposure. nue_constraint: multiply the weight of every GENIE event, reconstructed rows (full simulation and
    GENIE + surrogate) and generated events (the Truth tables, through their *_enu.npz companions), by nue_constraint_weight, so
    the GENIE samples correspond to the published flux. The GENIE + surrogate sample is returned as a share f_in (inferred from the
    NuWro composition targets) of the official Truth tables; use_genie_truth_tree then replaces that by the surrogate's own Truth tree."""
    sur = "gsur"
    g, gt, d = load(G_RECO), load(G_TRUTH), load(D_RECO)
    t = load(TABLE[sur])
    if int(t["deadtime_available"]) != 1:
        raise SystemExit(f"{TABLE[sur].name} has no dead-time bit")
    pot_data, pot_41 = float(d["file_pot"].sum()), float(g["file_pot"].sum())
    runs30 = np.unique(t["mc_run"])
    pot_30 = float(g["file_pot"][np.isin(g["file_run"], runs30)].sum())
    cvw = gt["cvweight"].astype(float)
    nue = {"applied": nue_constraint}
    if nue_constraint:
        c_gt = truth_constraint_weight(G_TRUTH, len(cvw), event_key(gt["run"], gt["subrun"], gt["nth"]))
        nue["truth_numu_cc_fullvol_mean"] = float((cvw * c_gt).sum() / cvw.sum())
        cvw = cvw * c_gt
    gt_pop = in_volume(gt["vtx"])
    gt_pop30 = gt_pop & np.isin(gt["run"], runs30)
    gt_sig = gt_pop & in_volume(gt["vtx"], Z_TUT, APOTHEM_TUT) & (np.arctan2(gt["pT"], gt["pL"]) <= THETA_MAX) & (gt["pL"] >= PL_MIN)
    go = load(G_TRUTH_OTHER)                     # every species: the rest of the Truth tree, and f_in from the NuWro composition targets
    truth30 = truth_categories(gt, go, runs30)
    f_in, _ = nuwro_target_fraction(load(N_RECO), truth30)
    gt_all = {"run": np.concatenate([gt["run"], go["run"]]), "subrun": np.concatenate([gt["subrun"], go["subrun"]]),
              "nth": np.concatenate([gt["nth"], go["nth"]]), "cvweight": np.concatenate([gt["cvweight"], go["cvweight"]])}
    W41, W30 = float(cvw[gt_pop].sum()), float(cvw[gt_pop30].sum())
    s_full = pot_data / pot_41

    # the full simulation: all 41 files, every reconstructed row, generator weight = wgt
    g_pop = in_volume(g["mc_vtx"])
    g_sel = (g["cuts"] & COMMON_BITS) == COMMON_BITS
    w_g = g["wgt"].astype(float)
    j = join_weight(gt_all, g["mc_run"][g_pop], g["mc_subrun"][g_pop], g["mc_nthEvtInFile"][g_pop])
    max_diff = float(np.abs(j - w_g[g_pop]).max())
    if max_diff > 1e-6:
        raise SystemExit(f"reco-tree wgt and Truth-tree mc_cvweight_total differ by up to {max_diff} on the same events")
    if nue_constraint:
        c_g = nue_constraint_weight(g["E_nu"].astype(float), g["mc_incoming"])
        nue["full_selected_mean"] = float((w_g * c_g)[g_sel].sum() / w_g[g_sel].sum())
        w_g = w_g * c_g
    S = {"full": {"name": "full", "tab": g, "w": w_g, "pop": g_pop, "sel_all": g_sel, "bkg": g_sel & ~g_pop, "scale": s_full,
                  "n_gen": W41, "truth": (gt["pT"][gt_sig].astype(float), gt["pL"][gt_sig].astype(float), cvw[gt_sig]), "f_in": 1.0, "ip": None, "p": None}}
    # GENIE + surrogate
    p = t["surrogate_p_reco_exists"].astype(float)
    w_t = join_weight(gt_all, t["mc_run"], t["mc_subrun"], t["mc_nthEvtInFile"])
    if nue_constraint:                       # at the true NEUTRINO energy from the Truth tables (2026-10-26 fix: the table's true_E is the muon energy)
        enu_truth, pdg_truth = truth_enu(G_TRUTH, len(gt["cvweight"]), event_key(gt["run"], gt["subrun"], gt["nth"]))
        enu_o, pdg_o = truth_enu(G_TRUTH_OTHER, len(go["cvweight"]), event_key(go["run"], go["subrun"], go["nth"]))
        enu_truth, pdg_truth = np.concatenate([enu_truth, enu_o]), np.concatenate([pdg_truth, pdg_o])
        keys_t = (t["mc_run"], t["mc_subrun"], t["mc_nthEvtInFile"])
        t_enu = join_truth_column(gt_all, enu_truth, *keys_t)
        t_pdg = join_truth_column(gt_all, pdg_truth, *keys_t)
        e_mu = t["true_E"].astype(float)
        if (t_pdg != t["mc_incoming"]).any() or (t_enu < e_mu - 0.01).any():
            raise SystemExit(f"GENIE + surrogate rows: {(t_pdg != t['mc_incoming']).sum()} joined pdg mismatches, {(t_enu < e_mu - 0.01).sum()} rows with E_nu below the muon energy")
        c_t = nue_constraint_weight(t_enu, t["mc_incoming"])
        t_sel = (t["cuts"] & COMMON_BITS) == COMMON_BITS
        nue["gsur_selected_mean"] = float((w_t * c_t)[t_sel].sum() / w_t[t_sel].sum())
        nue["gsur_selected_mean_at_muon_energy"] = float((w_t * nue_constraint_weight(e_mu, t["mc_incoming"]))[t_sel].sum() / w_t[t_sel].sum())   # the pre-2026-10-26 weight, for the record
        nue["gsur_constraint_energy"] = "true neutrino energy joined from the Truth tables by event key (fixed 2026-10-26; until then the table's true_E = muon energy)"
        w_t = w_t * c_t
    sig30 = gt_sig & np.isin(gt["run"], runs30)
    S["sur"] = {"name": "gsur", "tab": t, "w": w_t, "pop": np.ones(len(p), bool), "scale": pot_data / (f_in * pot_30), "n_gen": f_in * W30,
                "truth": (gt["pT"][sig30].astype(float), gt["pL"][sig30].astype(float), cvw[sig30]), "f_in": f_in, "ip": w_t / p, "p": p}
    for s in S.values():
        tab = s["tab"]
        s["cuts"], s["sig"] = tab["cuts"], tab["truth"] == 0b111111
        s["true"] = (tab["true_pT"].astype(float), tab["true_pL"].astype(float))
        s["reco"] = (tab["reco_pT"].astype(float), tab["reco_pL"].astype(float))
        s["sel"] = s["pop"] & ((s["cuts"] & COMMON_BITS) == COMMON_BITS)
    S["sur"]["label"] = LABEL[sur]
    # data
    d_sel = (d["cuts"] & COMMON_BITS) == COMMON_BITS
    D = {"tab": d, "sel": d_sel, "reco": (d["reco_pT"].astype(float), d["reco_pL"].astype(float)), "pot": pot_data, "n_rows": int(len(d["cuts"])),
         "n_files": int(len(d["file_run"])), "n_empty_files": int((d["file_entries"] == 0).sum()), "n_sel": int(d_sel.sum()),
         "deadtime_survival": float(((d["cuts"][d_sel] & DEADTIME_BIT) != 0).mean())}
    dt_full = float(w_g[g_sel & ((g["cuts"] & DEADTIME_BIT) != 0)].sum() / w_g[g_sel].sum())
    gen_sig = {"full": float(S["full"]["truth"][2].sum()), "sur": float(f_in * S["sur"]["truth"][2].sum())}
    info = {"set": "closure-test", "surrogate": sur, "pot": {"data": pot_data, "mc_41_files": pot_41, "mc_runs_110000_110029": pot_30, "data_over_mc_41": s_full},
            "scales_to_data_exposure": {"full": s_full, sur: S["sur"]["scale"]},
            "genie_runs_all": [int(g["file_run"].min()), int(g["file_run"].max()), int(len(g["file_run"]))], "genie_runs_surrogate": [int(runs30.min()), int(runs30.max()), int(len(runs30))],
            "sample_set": SAMPLE_SET, "selection_cuts": CUTS_TEXT, "f_in": f_in, "population_counts": {"genie_41_files": int(gt_pop.sum()), "genie_41_files_weighted": W41, "genie_30_runs": int(gt_pop30.sum()), "genie_30_runs_weighted": W30,
                                                "surrogate_generated_numu_cc": None},
            "generated_signal_weighted": gen_sig, "deadtime_survival_of_five_cut_sample": {"data": D["deadtime_survival"], "full": dt_full},
            "weights": WEIGHT_TEXT + ("; GENIE reconstructed rows x nu-e flux-constraint weight (published flux)" if nue_constraint else ""),
            "nue_constraint": nue, "weight_join_check_max_abs_diff": max_diff, "cv_weight_mean": {"full_rows": float(w_g.mean()), "truth_population_41": W41 / int(gt_pop.sum())},
            "selection": [CUTS[i] for i in CUT_LIST], "signal": "MINERvA 101 truth signal definition and phase space",
            "normalisation": f"every sample at the data exposure, POT {pot_data:.6e}", "labels": {**LABEL, "sur": LABEL[sur]}}
    return S, D, info


def gen_signal(S, s: str, key: str, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generated (weighted) signal events per bin of the true momentum (key "pT" or "pL") and the variance of the estimate:
    the sample's Truth-tree signal rows, weighted, times its f_in."""
    d = S[s]
    k = 0 if key == "pT" else 1
    pt, pl, w = d["truth"]
    h = hist(pt if k == 0 else pl, edges, w)
    return d["f_in"] * h, np.zeros_like(h, dtype=float)


def dc_efficiency(S, s: str, key: str, edges: np.ndarray):
    """Selected signal, generated signal, efficiency and its statistical error per bin of the true momentum (weighted rows), for the
    full simulation and GENIE + surrogate (wrapped by efficiency() below for the geo samples)."""
    d = S[s]
    k = 0 if key == "pT" else 1
    m = d["sel"] & d["sig"]
    num, num_w2 = hist(d["true"][k][m], edges, d["w"][m]), hist(d["true"][k][m], edges, d["w"][m] ** 2)
    den, var_den = gen_signal(S, s, key, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = num / den
        n_eff = num**2 / num_w2 / np.where(e > 0, e, np.nan)      # effective generated count behind the selected rows
        if d["name"] == "gsur":                  # a random share f_in of the population (f_in = 1 with the Truth-tree denominator: binomial)
            se = np.sqrt(e * (1 - e * d["f_in"]) / n_eff)
        else:                                    # the full simulation
            se = np.sqrt(e * (1 - e) / n_eff)
    return num, den, e, se


def cut_flow(S, s: str, population: np.ndarray, n_signal_generated: float):
    """Rows of the tutorial's cut table at the data exposure for `population` (a mask on the sample's rows)."""
    d = S[s]
    m = population.copy()
    rows, prev = [], None
    for name, i in [("reconstructed event", None)] + [(CUTS[i], i) for i in CUT_LIST]:
        if i is not None:
            m = m & passed(d["cuts"], i)
        ev, sg = float(d["w"][m].sum()), float(d["w"][m & d["sig"]].sum())
        rows.append({"cut": name, "events": ev, "events_at_data_pot": d["scale"] * ev, "signal_events": sg, "rows": int(m.sum()),
                     "efficiency": sg / n_signal_generated, "purity": sg / ev if ev else None, "relative_to_previous": None if prev is None else ev / prev})
        prev = ev
    return rows, m


def reco_hists(S, D, key: str, edges: np.ndarray) -> dict:
    """Selected events per bin of the reconstructed momentum at the data exposure: data, full simulation (all rows), the
    surrogate's own rows, the background completion and their sum; each with the variance of the estimate."""
    k = 0 if key == "pT" else 1
    out = {}
    x = D["reco"][k][D["sel"]]
    out["data"] = {"h": hist(x, edges), "var": hist(x, edges), "above": float((x >= edges[-1]).mean()), "below": float((x < edges[0]).mean())}
    f, s = S["full"], S["sur"]
    for name, d, m in (("full", f, f["sel_all"]), ("bkg", f, f["bkg"]), ("sur_only", s, s["sel"])):
        xx, w = d["reco"][k][m], d["scale"] * d["w"][m]
        out[name] = {"h": hist(xx, edges, w), "var": hist(xx, edges, w**2), "above": float(w[xx >= edges[-1]].sum() / w.sum()), "below": float(w[xx < edges[0]].sum() / w.sum())}
    out["sur"] = {"h": out["sur_only"]["h"] + out["bkg"]["h"], "var": out["sur_only"]["var"] + out["bkg"]["var"],
                  "above": out["sur_only"]["above"], "below": out["sur_only"]["below"]}
    return out


def purity(reco_x, signal, w, edges):
    """Purity per bin of the reconstructed momentum with row weights (binomial error on the effective count)."""
    sel, sel_w2, sig = hist(reco_x, edges, w), hist(reco_x, edges, w**2), hist(reco_x[signal], edges, w[signal])
    with np.errstate(divide="ignore", invalid="ignore"):
        p = sig / sel
        n_eff = sel**2 / sel_w2
        stat = np.sqrt(p * (1 - p) / n_eff)
    return {"selected": sel.round(1).tolist(), "selected_signal": sig.round(1).tolist(), "purity": p.round(5).tolist(), "stat": stat.round(5).tolist()}


def chi2(obs: dict, pred: dict, min_events: float = 10.0) -> dict:
    """chi2 of obs against pred per bin with both statistical variances (independent); bins where either is below min_events are left out."""
    ok = (obs["h"] >= min_events) & (pred["h"] >= min_events)
    v = obs["var"] + pred["var"]
    c = float((((obs["h"] - pred["h"]) ** 2)[ok] / v[ok]).sum())
    return {"chi2": round(c, 2), "bins": int(ok.sum()), "chi2_per_bin": round(c / max(int(ok.sum()), 1), 3)}


def chi2_cov(obs: np.ndarray, pred: np.ndarray, cov: np.ndarray, use: np.ndarray) -> dict:
    """chi2 of obs against pred with a full covariance over the bins `use`."""
    d = (obs - pred)[use]
    c = cov[np.ix_(use, use)]
    val = float(d @ np.linalg.solve(c, d))
    return {"chi2": round(val, 2), "bins": int(use.sum()), "chi2_per_bin": round(val / max(int(use.sum()), 1), 3)}


def compare(h: dict, epi: dict) -> dict:
    """Per flattened bin: events, band, ratios to the full simulation, data ratios and chi2 (statistical only and with the fully correlated band)."""
    d, dv, f, fv = (h[k][q].reshape(-1).astype(float) for k, q in (("data", "h"), ("data", "var"), ("full", "h"), ("full", "var")))
    out = {"data_events": d, "data_stat": np.sqrt(dv), "full_events": f, "full_stat": np.sqrt(fv)}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["data_over_full"] = np.where(f > 0, d / f, np.nan)
        out["data_over_full_stat"] = out["data_over_full"] * np.sqrt(np.where(d > 0, 1 / d, 0.0) + fv / f**2)
        out["chi2_data_vs_full_stat_only"] = chi2({"h": d, "var": dv}, {"h": f, "var": fv})
        for s in SUR:
            p, pv, e = h[s]["h"].reshape(-1), h[s]["var"].reshape(-1), epi[s]["correlated"]
            r_ = np.where(f > 0, p / f, np.nan)
            r_stat, r_epi = r_ * np.sqrt(np.where(p > 0, pv / p**2, 0.0) + fv / f**2), np.where(f > 0, e / f, np.nan)
            out.update({f"{s}_events": p, f"{s}_stat": np.sqrt(pv), f"{s}_epistemic": e, f"{s}_total": np.hypot(np.sqrt(pv), e), f"{s}_epistemic_independent": epi[s]["independent"],
                        f"{s}_epistemic_fraction": np.where(p > 0, e / p, np.nan), f"{s}_background_completion": h[s + "_bkg"].reshape(-1),
                        **{f"{s}_epistemic_{k}": epi[s][k] for k in ("exists", "minos", "charge")},
                        f"{s}_over_full": r_, f"{s}_over_full_stat": r_stat, f"{s}_over_full_epistemic": r_epi, f"{s}_over_full_total": np.hypot(r_stat, r_epi),
                        f"data_over_{s}": np.where(p > 0, d / p, np.nan)})
            use = (d >= 10) & (p >= 10)
            out[f"chi2_data_vs_{s}"] = {"stat_only": chi2({"h": d, "var": dv}, {"h": p, "var": pv}),
                                        "with_epistemic": chi2_cov(d, p, np.diag(dv + pv) + np.outer(e, e), use)}
    out["totals_in_range"] = {"data": float(d.sum()), "full": float(f.sum()), **{s: float(out[f"{s}_events"].sum()) for s in SUR},
                              **{f"{s}_epistemic": float(out[f"{s}_epistemic"].sum()) for s in SUR}}
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in x.round(6).tolist()] if x.ndim == 1 else x.round(6).tolist()
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=1))


def edge_axis(ax, edges, xlabel, fontsize=8, rotation=60):
    """Equal-width bins labelled by their edges (rotation 90 keeps large labels apart, 2026-10-19)."""
    nb = len(edges) - 1
    ax.set_xticks(np.arange(nb + 1)); ax.set_xticklabels([f"{e:g}" for e in edges], rotation=rotation, fontsize=fontsize); ax.set_xlim(0, nb); ax.set_xlabel(xlabel)


# ---------------- the cut-flow closure test (cc-inclusive cutflow_closure.py): the numbers of the residual panel

def stage_masks(d: dict, population: np.ndarray) -> list[np.ndarray]:
    """m_0 = population, m_k = m_{k-1} and the k-th cut of CUT_LIST (the masks of cut_flow)."""
    m, masks = population.copy(), []
    masks.append(m)
    for i in CUT_LIST:
        m = m & passed(d["cuts"], i)
        masks.append(m)
    return masks


def epistemic_terms(d: dict, masks: list[np.ndarray], rows: np.ndarray) -> dict:
    """Fully correlated epistemic change of the weighted count of `rows` (a mask: the signal or the non-signal rows) at every stage,
    its three terms, and the quadrature sum over rows ("independent", a lower limit); row weights = the generator weights.
    From the ensemble spread of the three decision probabilities (epi_p_reco_exists, epi_p_minos, epi_p_charge_neg = s_e, s_m, s_c):
        dN_exists(k) = sum_{m_k} w s_e / p_e                                   every stage
        dN_minos(k)  = r_k sum_{m_pre} w s_m,  r_k = N_k / N_MINOS              stages from the MINOS match on
        dN_charge(k) = r'_k sum_{m_before charge} w s_c, r'_k = N_k / N_charge  stages from the charge sign on
    (the three-term estimate kept in the JSON's per_stage rows; the figures' model uncertainty is the universes' half spread)."""
    t, w = d["tab"], d["w"]
    s_e, s_m, s_c, p_e = (t[k].astype(float) for k in ("epi_p_reco_exists", "epi_p_minos", "epi_p_charge_neg", "surrogate_p_reco_exists"))
    tot = lambda m, x: float((w * x)[m & rows].sum())
    one = np.ones(len(w))
    n = [tot(m, one) for m in masks]
    pre_lin, pre_quad = tot(masks[K_PRE], s_m), tot(masks[K_PRE], w * s_m**2)
    ch_lin, ch_quad = tot(masks[K_CHARGE - 1], s_c), tot(masks[K_CHARGE - 1], w * s_c**2)
    out = {"selected": n, "exists": [], "minos": [], "charge": [], "correlated": [], "independent": []}
    for k, m in enumerate(masks):
        ex, exq = tot(m, s_e / p_e), tot(m, w * (s_e / p_e) ** 2)
        r_m = n[k] / n[K_MINOS] if k >= K_MINOS else 0.0
        r_c = n[k] / n[K_CHARGE] if k >= K_CHARGE else 0.0
        mi, miq = r_m * pre_lin, r_m**2 * pre_quad
        ch, chq = r_c * ch_lin, r_c**2 * ch_quad
        out["exists"].append(ex); out["minos"].append(mi); out["charge"].append(ch)
        out["correlated"].append(ex + mi + ch); out["independent"].append(float(np.sqrt(exq + miq + chq)))
    return out


def stage_rows(d: dict, masks: list[np.ndarray], n_gen: float) -> list[dict]:
    """Efficiency and purity with their statistical uncertainties at every stage (the sample's own rows, generator weights)."""
    w, sig, rows = d["w"], d["sig"], []
    for k, m in enumerate(masks):
        N, N2 = float(w[m].sum()), float((w[m] ** 2).sum())
        S, S2 = float(w[m & sig].sum()), float((w[m & sig] ** 2).sum())
        e, p = S / n_gen, S / N
        n_eff_e = S**2 / S2 / e                      # effective generated count behind the selected signal rows
        stat_e = float(np.sqrt(e * (1 - e * d["f_in"]) / n_eff_e))      # f_in = 1 for the full simulation
        stat_p = float(np.sqrt(p * (1 - p) * N2 / N**2))
        rows.append({"cut": STAGES[k], "rows": int(m.sum()), "events": N, "events_at_data_pot": d["scale"] * N, "signal_events": S, "signal_events_at_data_pot": d["scale"] * S,
                     "efficiency": e, "efficiency_stat": stat_e, "purity": p, "purity_stat": stat_p})
    return rows


def closure_compute(S: dict, info: dict) -> tuple[dict, dict]:
    """The numbers of the closure test for the loaded closure-test samples (S, info = load_samples()[0], [2]): per-stage
    rows res[full, full_all, gsur, gsur_plus_bkg] and the differences comp[population, all_reconstructed]. "population" = the full
    simulation over the surrogate's population against GENIE + surrogate over its own rows; "all_reconstructed" = the full simulation
    over ALL its reconstructed events against GENIE + surrogate + the background completion (the residual panel of the cut-flow figure)."""
    f, g = S["full"], S["sur"]
    n_gen = info["generated_signal_weighted"]
    masks = {"full": stage_masks(f, f["pop"]), "full_all": stage_masks(f, np.ones(len(f["cuts"]), bool)), "bkg": stage_masks(f, ~f["pop"]), "gsur": stage_masks(g, g["pop"])}
    res = {"full": stage_rows(f, masks["full"], n_gen["full"]), "full_all": stage_rows(f, masks["full_all"], n_gen["full"]), "gsur": stage_rows(g, masks["gsur"], n_gen["sur"])}

    # epistemic uncertainty of GENIE + surrogate, stage by stage
    es, eb = epistemic_terms(g, masks["gsur"], g["sig"]), epistemic_terms(g, masks["gsur"], ~g["sig"])
    for k, r in enumerate(res["gsur"]):
        dS, dB, S_, B_ = es["correlated"][k], eb["correlated"][k], es["selected"][k], eb["selected"][k]
        N_ = S_ + B_
        if abs(S_ - r["signal_events"]) > 1e-6 * S_ or abs(N_ - r["events"]) > 1e-6 * N_:
            raise SystemExit(f"stage {k}: epistemic_terms and stage_rows disagree on the counts")
        r["efficiency_epistemic"] = dS / n_gen["sur"]
        r["efficiency_epistemic_independent"] = es["independent"][k] / n_gen["sur"]
        r["purity_epistemic"] = (B_ * dS - S_ * dB) / N_**2
        r["purity_epistemic_independent"] = float(np.sqrt(B_**2 * es["independent"][k] ** 2 + S_**2 * eb["independent"][k] ** 2) / N_**2)
        r["efficiency_total"] = float(np.hypot(r["efficiency_stat"], r["efficiency_epistemic"]))
        r["purity_total"] = float(np.hypot(r["purity_stat"], r["purity_epistemic"]))
        r["epistemic_terms"] = {"signal": {q: es[q][k] for q in ("exists", "minos", "charge", "correlated", "independent")},
                                "background": {q: eb[q][k] for q in ("exists", "minos", "charge", "correlated", "independent")}}

    # GENIE + surrogate + background completion (the out-of-volume rows of the full simulation at its scale): only the purity moves
    res["gsur_plus_bkg"] = []
    for k, r in enumerate(res["gsur"]):
        mb = masks["bkg"][k]
        Wb, Wb2 = f["scale"] * float(f["w"][mb].sum()), f["scale"] ** 2 * float((f["w"][mb] ** 2).sum())
        mg = masks["gsur"][k]
        Wg, Wg2 = g["scale"] * float(g["w"][mg].sum()), g["scale"] ** 2 * float((g["w"][mg] ** 2).sum())
        Sg = g["scale"] * r["signal_events"]
        N_ = Wg + Wb
        p = Sg / N_
        stat_p = float(np.sqrt(p * (1 - p) * (Wg2 + Wb2) / N_**2))
        dS, dB = g["scale"] * es["correlated"][k], g["scale"] * eb["correlated"][k]      # the completion has no epistemic part
        epi_p = ((N_ - Sg) * dS - Sg * dB) / N_**2
        res["gsur_plus_bkg"].append({"cut": STAGES[k], "events_at_data_pot": N_, "background_completion_at_data_pot": Wb, "signal_events_at_data_pot": Sg,
                                     "efficiency": r["efficiency"], "efficiency_stat": r["efficiency_stat"], "efficiency_epistemic": r["efficiency_epistemic"], "efficiency_total": r["efficiency_total"],
                                     "purity": p, "purity_stat": stat_p, "purity_epistemic": epi_p, "purity_total": float(np.hypot(stat_p, epi_p))})

    # differences per stage
    def differences(a: list[dict], b: list[dict]) -> dict:
        out = {}
        for q in ("efficiency", "purity"):
            d = np.array([ra[q] - rb[q] for ra, rb in zip(a, b)])
            s_stat = np.array([np.hypot(ra[q + "_stat"], rb[q + "_stat"]) for ra, rb in zip(a, b)])
            epi = np.array([ra[q + "_epistemic"] for ra in a])
            s_tot = np.hypot(s_stat, epi)
            out[q] = {"difference": d.tolist(), "sigma_stat": s_stat.tolist(), "epistemic": epi.tolist(), "sigma_total": s_tot.tolist(),
                      "pull_stat_only": (d / s_stat).tolist(), "pull": (d / s_tot).tolist(), "ratio": [ra[q] / rb[q] for ra, rb in zip(a, b)],
                      "max_abs_pull": float(np.abs(d / s_tot).max()), "stages_within_1_sigma": int((np.abs(d / s_tot) <= 1).sum()), "stages_within_2_sigma": int((np.abs(d / s_tot) <= 2).sum())}
        return out
    comp = {"population": differences(res["gsur"], res["full"]), "all_reconstructed": differences(res["gsur_plus_bkg"], res["full_all"])}

    return res, comp


# ================= the final figures (cc-inclusive-final-v5/scripts/final_figures.py) =================

def load_geo(s: str, full: dict, D: dict, info_g: dict) -> tuple[dict, dict, dict]:
    """Version 5: the geo-sample surrogate `s` (nsur | bsur) as a (S, D, info) triple in the layout of load_samples:
    S = {"full": the GENIE full simulation of the closure-test set (common), "sur": the geo surrogate}. Row weight = gen_weight (events at
    gen_pot), scale = POT_data / gen_pot; the generated signal (efficiency denominator) = the Truth tree's signal rows with their gen_weight;
    f_in = 1 (the Truth tree is the whole input population, so the efficiency's statistical error is binomial on the effective count)."""
    t, T = load(GEO_TABLE[s]["selection"]), load(GEO_TABLE[s]["truth"])
    if int(t["deadtime_available"]) != 1 or "gen_weight" not in t.files or "gen_weight" not in T.files:
        raise SystemExit(f"{GEO_TABLE[s]['selection'].name} / {GEO_TABLE[s]['truth'].name}: not geo tables with the dead-time bit and gen_weight")
    if str(T["selection_table"]) != GEO_TABLE[s]["selection"].name or int(T["reco_exists"].sum()) != len(t["cuts"]):
        raise SystemExit(f"{GEO_TABLE[s]['truth'].name} does not belong to {GEO_TABLE[s]['selection'].name}")
    pot, pot_T = float(t["gen_pot"]), float(T["gen_pot"])
    if abs(pot - pot_T) > 1e-6 * pot:
        raise SystemExit(f"gen_pot differs between the selection ({pot}) and Truth ({pot_T}) tables of {s}")
    w, p = t["gen_weight"].astype(float), t["surrogate_p_reco_exists"].astype(float)
    rx = T["reco_exists"]
    if (T["member"][rx] != t["member"]).any() or np.abs(T["gen_weight"][rx] - w).max() > 1e-6:
        raise SystemExit(f"the Truth tree's reco rows do not reproduce the member / gen_weight of {GEO_TABLE[s]['selection'].name}")
    sig_T = T["truth"] == 0b111111
    n_gen = float(T["gen_weight"][sig_T].sum())
    cc = (T["mc_incoming"] == 14) & (T["mc_current"] == 1)
    d = {"name": s, "tab": t, "w": w, "pop": np.ones(len(p), bool), "scale": D["pot"] / pot, "n_gen": float(T["gen_weight"][cc].sum()), "f_in": 1.0, "ip": w / p, "p": p, "geo": True,
         "truth": (T["true_pT"][sig_T].astype(float), T["true_pL"][sig_T].astype(float), T["gen_weight"][sig_T].astype(float)),
         "cuts": t["cuts"], "sig": t["truth"] == 0b111111, "true": (t["true_pT"].astype(float), t["true_pL"].astype(float)),
         "reco": (t["reco_pT"].astype(float), t["reco_pL"].astype(float)), "label": LABEL[s]}
    d["sel"] = d["pop"] & ((d["cuts"] & COMMON_BITS) == COMMON_BITS)
    info = {"set": GEO[s], "surrogate": s, "sample_set": SAMPLE_SET, "selection_cuts": CUTS_TEXT, "f_in": 1.0,
            "pot": {"data": D["pot"], "gen_pot": pot, "data_over_gen_pot": D["pot"] / pot},
            "scales_to_data_exposure": {"full": full["scale"], s: D["pot"] / pot},
            "population_counts": {"truth_rows": int(len(T["truth"])), "truth_sum_gen_weight": float(T["gen_weight"].sum()), "truth_numu_cc_rows": int(cc.sum()),
                                  "truth_numu_cc_sum_gen_weight": float(T["gen_weight"][cc].sum()), "signal_rows": int(sig_T.sum()), "signal_sum_gen_weight": n_gen,
                                  "signal_rows_reconstructed": int((sig_T & rx).sum()), "reco_rows": int(len(p)), "reco_sum_gen_weight": float(w.sum()),
                                  "gen_pot": pot, "scale_to_data": D["pot"] / pot, "surrogate_generated_numu_cc": None},
            "generated_signal_weighted": {"full": info_g["generated_signal_weighted"]["full"], "sur": n_gen},
            "weights": info_g["weights"] + f"; {GEO_TEXT[s]}: gen_weight (events at gen_pot = {pot:.6e} POT; the generator's own cross section and nuclide composition, no composition weight, no K factor)",
            "nue_constraint": {"applied": False, "note": "the geo sample was generated with the published (nu-e constrained) flux; no reweighting"},
            "normalisation": f"every sample at the data exposure, POT {D['pot']:.6e}; {GEO_TEXT[s]}: sum(gen_weight) x POT_data / gen_pot",
            "tables": {q: str(path.relative_to(FINAL)) for q, path in GEO_TABLE[s].items()}, "labels": {**LABEL, "sur": LABEL[s]}}
    return {"full": full, "sur": d}, D, info


def use_genie_truth_tree(S: dict, info: dict) -> dict:
    """Version 5 (user request 2026-11-03: "use the genie Truth tree instead of f_in in v5"): replace the GENIE + surrogate sample's
    generated population (f_in x the official Truth tables of runs 110000-110029, load_samples) by the surrogate file's
    own Truth tree: the generated signal = its signal rows with the official cvweight (joined by event key) x the nu-e flux-constraint
    weight at the true neutrino energy; the exposure = its row share f_T = N_truth / N_30 of the official population of the volume
    (all species and currents) times POT_30; f_in = 1 in the error formula (binomial). Modifies S["sur"] and info in place; returns the
    record for the JSON (checks, the old and new normalisation)."""
    d = S["sur"]
    T = load(GENIE_TRUTH)
    if str(T["selection_table"]) != TABLE["gsur"].name:
        raise SystemExit(f"{GENIE_TRUTH.name} was extracted for {T['selection_table']}, not {TABLE['gsur'].name}")
    rx = T["reco_exists"]
    n = len(d["cuts"])
    if int(rx.sum()) != n or (T["reco_row"][rx] != np.arange(n)).any() or (T["member"][rx] != d["tab"]["member"]).any():
        raise SystemExit(f"{GENIE_TRUTH.name}: the reco_exists rows are not the selection table's rows in order")
    if not (np.allclose(T["true_pT"][rx], d["true"][0], rtol=1e-5, atol=1e-4) and np.allclose(T["true_pL"][rx], d["true"][1], rtol=1e-5, atol=1e-4) and (T["truth"][rx] == d["tab"]["truth"]).all()):
        raise SystemExit(f"{GENIE_TRUTH.name}: true momenta or truth bits of the reco rows differ from the selection table")
    if not in_volume(T["mc_vtx"].astype(float)).all():
        raise SystemExit(f"{GENIE_TRUTH.name}: rows outside the full volume")
    gt, go = load(G_TRUTH), load(G_TRUTH_OTHER)
    runs30 = np.unique(T["mc_run"])
    n30 = int((in_volume(gt["vtx"]) & np.isin(gt["run"], runs30)).sum() + (in_volume(go["vtx"]) & np.isin(go["run"], runs30)).sum())
    n_T = int(len(T["truth"]))
    f_T = n_T / n30
    nue = bool(info["nue_constraint"]["applied"])
    sig, cc = T["truth"] == 0b111111, (T["truth"] & 0b11) == 0b11
    w_cc = join_weight(gt, T["mc_run"][cc], T["mc_subrun"][cc], T["mc_nthEvtInFile"][cc])          # numu CC rows: in the numu CC Truth table
    if nue:
        w_cc = w_cc * nue_constraint_weight(T["true_E_nu"][cc].astype(float), T["mc_incoming"][cc])
    w_sig = w_cc[sig[cc]]
    old = {"f_in_inferred": info["f_in"], "scale": d["scale"], "generated_signal_weighted": info["generated_signal_weighted"]["sur"], "population_numu_cc_weighted": d["n_gen"],
           "truth_table_signal_rows": int(len(d["truth"][0]))}
    pot = info["pot"]
    d["truth"] = (T["true_pT"][sig].astype(float), T["true_pL"][sig].astype(float), w_sig)
    d["f_in"], d["n_gen"], d["scale"] = 1.0, float(w_cc.sum()), pot["data"] / (f_T * pot["mc_runs_110000_110029"])      # n_gen = the weighted numu CC population, as load_samples defines it (unused by the figures)
    info["generated_signal_weighted"]["sur"] = float(w_sig.sum())
    info["scales_to_data_exposure"]["gsur"] = d["scale"]
    info["f_in"] = f_T
    info["population_counts"].update({"surrogate_truth_tree_rows": n_T, "surrogate_truth_tree_numu_cc_rows": int(cc.sum()), "surrogate_truth_tree_numu_cc_weighted": float(w_cc.sum()),
                                      "surrogate_truth_tree_signal_rows": int(sig.sum()), "surrogate_generated_numu_cc": float(w_cc.sum()), "official_30_runs_all_species": n30})
    rec = {"table": GENIE_TRUTH.name, "what": "GENIE + surrogate: generated population = the surrogate file's Truth tree (every event the surrogate was run on); generated signal = "
                                               "its signal rows x official cvweight (joined by event key) x nu-e constraint at the true neutrino energy; exposure = row share f_T of the "
                                               "official 30-run population of the volume x POT_30; f_in = 1 in the efficiency error (binomial)",
           "truth_tree_rows": n_T, "official_30_runs_all_species_in_volume": n30, "f_T_row_share": f_T, "f_in_inferred_before": old["f_in_inferred"], "f_T_over_f_in": f_T / old["f_in_inferred"],
           "weighted_numu_cc_share": float(w_cc.sum() / info["population_counts"]["genie_30_runs_weighted"]), "signal_rows": int(sig.sum()), "signal_rows_reconstructed": int((sig & rx).sum()),
           "generated_signal_weighted": {"truth_tree": float(w_sig.sum()), "f_in_x_official_before": old["generated_signal_weighted"], "ratio": float(w_sig.sum()) / old["generated_signal_weighted"]},
           "population_numu_cc_weighted": {"truth_tree": float(w_cc.sum()), "f_in_x_official_before": old["population_numu_cc_weighted"], "ratio": float(w_cc.sum()) / old["population_numu_cc_weighted"]},
           "scale_to_data": {"truth_tree": d["scale"], "before": old["scale"], "ratio": d["scale"] / old["scale"]}, "nue_constraint_applied": nue, "pot_30": pot["mc_runs_110000_110029"]}
    return rec


def efficiency(S, s: str, key: str, edges: np.ndarray):
    """Version 5: dc_efficiency, except for a geo sample, whose generated signal is the Truth tree itself (exact, f_in = 1):
    the statistical error is the binomial one on the effective generated count behind the selected rows, as for the full simulation."""
    d = S[s]
    if not d.get("geo"):
        return dc_efficiency(S, s, key, edges)
    k = 0 if key == "pT" else 1
    m = d["sel"] & d["sig"]
    num, num_w2 = hist(d["true"][k][m], edges, d["w"][m]), hist(d["true"][k][m], edges, d["w"][m] ** 2)
    pt, pl, wT = d["truth"]
    den = hist(pt if k == 0 else pl, edges, wT)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = num / den
        n_eff = num**2 / num_w2 / np.where(e > 0, e, np.nan)
        se = np.sqrt(e * (1 - e) / n_eff)
    return num, den, e, se


def half_spread(values) -> np.ndarray:
    """(max - min) / 2 over the first axis (the universes): the 1 sigma of the user's prescription."""
    v = np.asarray(values, float)
    with np.errstate(invalid="ignore"):
        return 0.5 * (np.nanmax(v, axis=0) - np.nanmin(v, axis=0))


def load_universes(s: str, d: dict) -> dict:
    """The four ensemble universes of surrogate sample `s` (d = its loaded sample dict): the table's companion
    data/<table>_surrogate_universes.npz (made by cc-inclusive extract_surrogate_universes.py). Per universe k: the cut bits (member
    k's Tracker, Apothem and angle on its own vertex and angle, the nominal MINOS, dead-time and charge bits), member k's reconstructed
    pT and pL from its own muon momentum and angle, the six-cut selection mask on the sample's population, and (version 3) the row
    weights = the generator weights x surrogate_univ_weight[k] ("w"; the nominal generator weights under "w_nominal")."""
    path = GEO_TABLE[s]["selection"] if d.get("geo") else TABLE[s]      # version 5: the geo tables for nsur / bsur
    upath = path.with_name(path.name.replace("_surrogate_selection.npz", "_surrogate_universes.npz"))
    if not upath.exists():
        raise SystemExit(f"{upath} is missing: the universes companion of {path.name}")
    u = load(upath)
    n = len(d["cuts"]); rows = np.arange(n); member = u["member"]
    if len(member) != n or (member != d["tab"]["member"]).any() or ((u["univ_cuts"][rows, member] & VALUE_BITS) != (d["cuts"] & VALUE_BITS)).any():
        raise SystemExit(f"{upath.name} does not align with {path.name}")
    uw = u["univ_weight"].astype(float)      # version 3: the producer's universe weight = p_k / p_nom of the reco, MINOS and charge decisions given the nominal outcomes
    if not np.isfinite(uw).all() or (uw[rows, member] != 1).any():
        raise SystemExit(f"{upath.name}: surrogate_univ_weight must be finite and exactly 1 on the member's own universe")
    factors = None                           # version 4: the three flag factors of the weight
    if all(f"univ_weight_{q}" in u.files for q in ("reco", "minos", "charge")):
        factors = {q: u[f"univ_weight_{q}"].astype(float) for q in ("reco", "minos", "charge")}
        if not np.allclose(factors["reco"] * factors["minos"] * factors["charge"], uw, rtol=1e-5, atol=1e-6) or any((f[rows, member] != 1).any() or not np.isfinite(f).all() for f in factors.values()):
            raise SystemExit(f"{upath.name}: the flag factors must be finite, 1 on the member's own universe and multiply to surrogate_univ_weight")
    elif s == "gsur":
        raise SystemExit(f"{upath.name} has no univ_weight_reco/minos/charge columns: it must be extracted from the producer's 2026-10-29 GENIE ensemble file")
    P, th = u["univ_P"].astype(float), u["univ_theta"].astype(float)
    pt, pl = P * np.sin(th), P * np.cos(th)
    own, nom = pt[rows, member], d["reco"][0]; ok = np.isfinite(own) & np.isfinite(nom)
    if (np.abs(own[ok] - nom[ok]) > 1e-4 * (1 + np.abs(nom[ok]))).any():
        raise SystemExit(f"{upath.name}: the member's own universe does not reproduce the nominal reconstructed pT")
    cuts = [(d["cuts"] & ~VALUE_BITS) | (u["univ_cuts"][:, k] & VALUE_BITS) for k in range(K_MEMBERS)]
    pop = d["pop"]
    return {"path": upath, "member": member, "cuts": cuts, "sel": [pop & ((c & COMMON_BITS) == COMMON_BITS) for c in cuts],
            "reco": [(pt[:, k], pl[:, k]) for k in range(K_MEMBERS)],
            "w": [d["w"] * uw[:, k] for k in range(K_MEMBERS)], "w_nominal": d["w"], "univ_weight": uw, "factors": factors, "univ_cuts": u["univ_cuts"],
            "universe_weight": {"mean": uw.mean(0).tolist(), "p02": np.percentile(uw, 2, axis=0).tolist(), "p98": np.percentile(uw, 98, axis=0).tolist(),
                                "min": uw.min(0).tolist(), "max": uw.max(0).tolist(), "rows_above_10": (uw > 10).sum(0).tolist(),
                                "weighted_mean_over_population": [float((d["w"][pop] * uw[pop, k]).sum() / d["w"][pop].sum()) for k in range(K_MEMBERS)]},
            "flag_factors": None if factors is None else {q: {"mean": f.mean(0).tolist(), "p02": np.percentile(f, 2, axis=0).tolist(), "p98": np.percentile(f, 98, axis=0).tolist(),
                                                              "min": f.min(0).tolist(), "max": f.max(0).tolist(), "rows_above_5": (f > 5).sum(0).tolist()} for q, f in factors.items()},
            "value_cuts_agree_with_nominal": [float(((c & VALUE_BITS) == (d["cuts"] & VALUE_BITS)).mean()) for c in cuts]}


def components(d: dict, U: dict) -> dict:
    """Version 4: the universe sets with one part of the 'all' universe (U) at a time, everything else nominal - reco / minos / charge
    (one flag factor each, when the file has them), flags (the three together = U's weights with nominal cuts and momentum), vertex (the
    Tracker and Apothem cuts on the member's vertex), muon (the angle cut and the reconstructed pT, pL from the member's momentum and
    angle), kinematics (vertex + muon with nominal weights = the version-2 universe) - and 'all' = U. Each set has the keys the
    universe_* functions read (cuts, sel, reco, w)."""
    nom_cuts, nom_reco, w0, uc = d["cuts"], d["reco"], d["w"], U["univ_cuts"]
    sets = {}
    if U["factors"] is not None:
        for q, f in U["factors"].items():
            sets[q] = {"cuts": [nom_cuts] * K_MEMBERS, "reco": [nom_reco] * K_MEMBERS, "w": [w0 * f[:, k] for k in range(K_MEMBERS)]}
    sets["flags"] = {"cuts": [nom_cuts] * K_MEMBERS, "reco": [nom_reco] * K_MEMBERS, "w": U["w"]}
    sets["vertex"] = {"cuts": [(nom_cuts & ~VERTEX_BITS) | (uc[:, k] & VERTEX_BITS) for k in range(K_MEMBERS)], "reco": [nom_reco] * K_MEMBERS, "w": [w0] * K_MEMBERS}
    sets["muon"] = {"cuts": [(nom_cuts & ~ANGLE_BIT) | (uc[:, k] & ANGLE_BIT) for k in range(K_MEMBERS)], "reco": U["reco"], "w": [w0] * K_MEMBERS}
    sets["kinematics"] = {"cuts": U["cuts"], "reco": U["reco"], "w": [w0] * K_MEMBERS}
    for st in sets.values():
        st["sel"] = [d["pop"] & ((c & COMMON_BITS) == COMMON_BITS) for c in st["cuts"]]
    for k in range(K_MEMBERS):               # 'all' is the composition of the parts
        if (U["cuts"][k] != ((sets["vertex"]["cuts"][k] & ~ANGLE_BIT) | (sets["muon"]["cuts"][k] & ANGLE_BIT))).any():
            raise SystemExit("the universe's cuts are not the vertex cuts plus the angle cut of the member")
    sets["all"] = U
    return {q: sets[q] for q in COMPONENTS if q in sets}


def universe_w(d: dict, U: dict, k: int, weighted: bool) -> np.ndarray:
    """The row weights of universe k: the universe set's weights U["w"][k] (generator weights x the universe's weight; version 4: the
    components() sets carry their own), or the nominal generator weights (weighted=False, kept for compatibility, unused)."""
    return U["w"][k] if weighted else d["w"]


def universe_efficiencies(d: dict, U: dict, key: str, edges: np.ndarray, weighted: bool = True) -> np.ndarray:
    """[4, nb] the selection efficiency per true-momentum bin in every universe: its selected signal (its weights) over the common generated signal."""
    kv = 0 if key == "pT" else 1
    _num, den, _e, _se = efficiency({d["name"]: d}, d["name"], key, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.array([hist(d["true"][kv][sel & d["sig"]], edges, universe_w(d, U, k, weighted)[sel & d["sig"]]) / den for k, sel in enumerate(U["sel"])])


def universe_purities(d: dict, U: dict, key: str, edges: np.ndarray, weighted: bool = True) -> np.ndarray:
    """[4, nb] the purity per reconstructed-momentum bin in every universe: its selection, its own reconstructed momentum, its weights."""
    kv = 0 if key == "pT" else 1
    return np.array([purity(U["reco"][k][kv][sel], d["sig"][sel], universe_w(d, U, k, weighted)[sel], edges)["purity"] for k, sel in enumerate(U["sel"])], float)


def universe_hists(d: dict, U: dict, kv: int, edges: np.ndarray, weighted: bool = True) -> np.ndarray:
    """[4, nb] the selected surrogate rows per reconstructed-momentum bin at the data exposure in every universe (its selection, its own momentum, its weights)."""
    return np.array([hist(U["reco"][k][kv][sel], edges, (d["scale"] * universe_w(d, U, k, weighted))[sel]) for k, sel in enumerate(U["sel"])])


def universe_cutflow(d: dict, U: dict, n_gen: float, weighted: bool = True) -> list[list[dict]]:
    """The cut-flow rows (stage_rows) of every universe (its cut bits, its weights)."""
    return [stage_rows({**d, "cuts": c, "w": universe_w(d, U, k, weighted)}, stage_masks({"cuts": c}, d["pop"]), n_gen) for k, c in enumerate(U["cuts"])]


def figure_cutflow_and_efficiency_purity(plt, col, line, Line2D) -> dict:
    """cutflow_efficiency_purity.png and selection_efficiency_purity_pT_pL.png: the GENIE rows at `wgt` x nu-e flux constraint like the
    muon pT/pL figure (nue_constraint=True)."""
    loaded = {"gsur": load_samples(nue_constraint=True)}
    genie_truth = use_genie_truth_tree(loaded["gsur"][0], loaded["gsur"][2])      # version 5: the GENIE surrogate's own Truth tree instead of f_in
    for s in SUR:                                # version 5: NuWro and GiBUU from the geo-sample files
        if s in GEO:
            loaded[s] = load_geo(s, loaded["gsur"][0]["full"], loaded["gsur"][1], loaded["gsur"][2])
    S = {"full": loaded["gsur"][0]["full"], **{s: loaded[s][0]["sur"] for s in SUR}}
    f = S["full"]
    U = {s: load_universes(s, S[s]) for s in SUR}
    C = {s: components(S[s], U[s]) for s in SUR}        # version 4: the parts of the universe, for the JSON
    gen = {"full": loaded["gsur"][2]["generated_signal_weighted"]["full"], **{s: loaded[s][2]["generated_signal_weighted"]["sur"] for s in SUR}}
    out = {"weights": loaded["gsur"][2]["weights"] + "".join(f"; {GEO_TEXT[s]}: gen_weight (events at gen_pot), no composition weight, no K" for s in SUR if s in GEO),
           "nue_constraint": {s: loaded[s][2]["nue_constraint"] for s in SUR}, "selection": CUTS_TEXT,
           "surrogate_normalisation": {s: loaded[s][2]["population_counts"] for s in SUR},
           "geo_samples": {s: {"set": loaded[s][2]["set"], "pot": loaded[s][2]["pot"], "tables": loaded[s][2]["tables"], "generated_signal_weighted": loaded[s][2]["generated_signal_weighted"]["sur"],
                               "efficiency_denominator": "the Truth tree of the geo surrogate file (the input population), sum of gen_weight over its signal rows"} for s in SUR if s in GEO},
           "genie_truth_tree": genie_truth,
           "model_uncertainty": "half the spread (max - min)/2 over the four ensemble universes, 'all together': member k's own vertex and muon angle for the Tracker, Apothem and angle cuts, its own momentum and angle for the reconstructed pT/pL, and every row weighted by surrogate_univ_weight[k] = the product of the three flag factors univ_weight_reco x _minos x _charge = member k's over the nominal member's probability of the row's reconstruction, MINOS-match and charge-sign outcomes (nominal MINOS, dead-time and charge bits and rows); 'components' = the same half spread with one part of the universe at a time (reco, minos, charge, flags = the three factors together, vertex, muon, kinematics = vertex + muon = the version-2 universe), everything else nominal; correlated, not adding in quadrature to 'all'",
           "components": COMPONENTS,
           "universes": {s: {"table": U[s]["path"].name, "rows_per_member": np.bincount(U[s]["member"], minlength=K_MEMBERS).tolist(), "value_cuts_agree_with_nominal": U[s]["value_cuts_agree_with_nominal"],
                             "universe_weight": U[s]["universe_weight"], "flag_factors": U[s]["flag_factors"], "components_available": list(C[s])} for s in SUR}}

    # ---------------- numbers
    flows = {"full": cut_flow(S, "full", f["pop"], gen["full"])[0], "full_all": cut_flow(S, "full", np.ones(len(f["cuts"]), bool), gen["full"])[0],
             **{s: cut_flow(S, s, S[s]["pop"], gen[s])[0] for s in SUR}}
    stages = [r["cut"] for r in flows["full"]]
    eff_all = {n: float(flows[n][-1]["efficiency"]) for n in ("full", "full_all", *SUR)}
    pur_all = {"full": flows["full"][-1]["purity"], "full_all": flows["full_all"][-1]["purity"], **{s: flows[s][-1]["purity"] for s in SUR}}
    out.update({"cut_stages": stages, "cut_stage_labels": [STAGE_LABEL.get(s, s) for s in stages], "efficiency_overall": eff_all, "purity_overall": pur_all,
                "efficiency_per_stage": {n: [r["efficiency"] for r in flows[n]] for n in ("full", "full_all", *SUR)},
                "purity_per_stage": {n: [r["purity"] for r in flows[n]] for n in ("full", "full_all", *SUR)}})
    # residual panel numbers: closure_compute on the same loaded closure-test samples; its full_all and gsur rows must
    # reproduce the curves above exactly (same masks, same weights)
    res_c, comp_c = closure_compute(loaded["gsur"][0], loaded["gsur"][2])
    for n in ("full_all", "gsur"):
        for q in ("efficiency", "purity"):
            if max(abs(a[q] - (b[q] or 0.0)) for a, b in zip(res_c[n], flows[n])) > 1e-9:
                raise SystemExit(f"closure_compute and cut_flow disagree on the {n} {q} per stage")
    # residual panel, version 2: the model part of the thin bars = half the spread over the four universes of the GENIE + surrogate
    # efficiency and of its purity with the background completion (stat part, differences and the nominal unchanged)
    resid = {q: dict(comp_c["all_reconstructed"][q]) for q in ("efficiency", "purity")}
    Wb = np.array([r["background_completion_at_data_pot"] for r in res_c["gsur_plus_bkg"]])

    def cutflow_universes(Us: dict) -> dict:
        rows_k = universe_cutflow(S["gsur"], Us, gen["gsur"], True)
        E = np.array([[r["efficiency"] for r in rk] for rk in rows_k])
        P = np.array([[r["signal_events_at_data_pot"] / (r["events_at_data_pot"] + wb) for r, wb in zip(rk, Wb)] for rk in rows_k])
        Pg = np.array([[r["purity"] for r in rk] for rk in rows_k])
        return {"efficiency": E, "purity": Pg, "purity_with_background_completion": P,
                "half_spread": {"efficiency": half_spread(E), "purity": half_spread(Pg), "purity_with_background_completion": half_spread(P)}}

    cu = cutflow_universes(U["gsur"])                                  # the band: 'all together' (version 3's universe)
    cu_c = {q: cutflow_universes(Us) for q, Us in C["gsur"].items() if q != "all"}      # version 4: the components, for the JSON
    E_k, P_k = cu["efficiency"], cu["purity_with_background_completion"]
    out["cutflow_universes"] = {"gsur": {**cu, "nominal": {"efficiency": [r["efficiency"] for r in flows["gsur"]], "purity": [r["purity"] for r in flows["gsur"]],
                                                           "purity_with_background_completion": [r["purity"] for r in res_c["gsur_plus_bkg"]]},
                                         "components": cu_c}}
    for q, epi in (("efficiency", half_spread(E_k)), ("purity", half_spread(P_k))):
        dq, s_stat = np.array(resid[q]["difference"]), np.array(resid[q]["sigma_stat"]); s_tot = np.hypot(s_stat, epi)
        resid[q].update({"epistemic": epi.tolist(), "sigma_total": s_tot.tolist(), "pull": (dq / s_tot).tolist(), "max_abs_pull": float(np.abs(dq / s_tot).max()),
                         "stages_within_1_sigma": int((np.abs(dq / s_tot) <= 1).sum()), "stages_within_2_sigma": int((np.abs(dq / s_tot) <= 2).sum())})
        ref = np.array([r[q] for r in res_c["full_all"]])          # the relative difference (surrogate - full) / full, drawn in the panel (user request 2026-10-28)
        resid[q].update({"reference_full_all": ref.tolist(), "relative_difference": (dq / ref).tolist(), "relative_sigma_stat": (s_stat / ref).tolist(),
                         "relative_epistemic": (epi / ref).tolist(), "relative_sigma_total": (s_tot / ref).tolist(),
                         "ratio": (1 + dq / ref).tolist()})                 # drawn: surrogate / full (user request 2026-10-28); its errors = the relative_* ones
    out["residual"] = {"comparison": "GENIE + surrogate + background completion (the selected full-simulation rows generated outside the surrogate volume, at the full "
                                     "simulation's scale) minus GENIE full simulation over all its reconstructed events, per stage: cutflow_closure.py 'all_reconstructed'",
                       "units": "fractions; the figure draws the RATIO surrogate / full (key ratio = 1 + relative_difference) with the relative_* errors; difference, sigma_* and epistemic are absolute", "sigma_stat": "stat_full (+) stat_sur in quadrature", "epistemic": "half the spread over the four ensemble universes (vertex and angle cuts re-cut per member, rows weighted by surrogate_univ_weight[k])",
                       "sigma_total": "sigma_stat (+) epistemic", "pull": "difference / sigma_total",
                       "efficiency": resid["efficiency"], "purity": resid["purity"], "per_stage": {n: res_c[n] for n in ("full_all", "gsur", "gsur_plus_bkg")}}

    eff, pur = {}, {}
    for key, edges in (("pT", PT_PAPER), ("pL", PL_PAPER)):
        k = 0 if key == "pT" else 1
        e = {"edges": edges}
        for n in ("full", *SUR):
            num, den, val, se = efficiency(S, n, key, edges)
            e[n] = {"efficiency": val, "stat": se}
        for s in SUR:
            ek = universe_efficiencies(S[s], U[s], key, edges)            # [4, nb]: the efficiency in every universe ('all together')
            e[s]["universes"] = ek
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(e[s]["efficiency"] > 0, half_spread(ek) / e[s]["efficiency"], 0.0)
                e[s]["components"] = {}                                  # version 4: one part of the universe at a time
                for q, Us in C[s].items():
                    ekq = ek if q == "all" else universe_efficiencies(S[s], Us, key, edges)
                    e[s]["components"][q] = {"universes": ekq, "half_spread": half_spread(ekq), "model_rel": np.where(e[s]["efficiency"] > 0, half_spread(ekq) / e[s]["efficiency"], 0.0)}
            r = e[s]["efficiency"] / e["full"]["efficiency"]
            e[s].update({"model_rel": rel, "model": e[s]["efficiency"] * rel, "total": np.hypot(e[s]["stat"], e[s]["efficiency"] * rel), "over_full": r,
                         "over_full_stat": r * np.hypot(e[s]["stat"] / e[s]["efficiency"], e["full"]["stat"] / e["full"]["efficiency"]), "over_full_model": r * rel})
            e[s]["over_full_total"] = np.hypot(e[s]["over_full_stat"], e[s]["over_full_model"])
        eff[key] = e
        p = {"full": purity(f["reco"][k][f["sel"]], f["sig"][f["sel"]], f["w"][f["sel"]], edges),
             "full_all": purity(f["reco"][k][f["sel_all"]], f["sig"][f["sel_all"]], f["w"][f["sel_all"]], edges)}
        for s in SUR:
            d = S[s]
            p[s] = purity(d["reco"][k][d["sel"]], d["sig"][d["sel"]], d["w"][d["sel"]], edges)
            pk = universe_purities(d, U[s], key, edges)               # [4, nb]: the purity in every universe (its selection, its own momentum, its weights)
            ps_ = np.array(p[s]["purity"]); epi = half_spread(pk)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(ps_ > 0, epi / ps_, 0.0)
            p[s].update({"universes": pk.round(5).tolist(), "epistemic": epi.round(6).tolist(), "epistemic_rel": rel.round(6).tolist(),
                         "total": np.hypot(np.array(p[s]["stat"]), epi).round(5).tolist(),
                         "components": {q: {"universes": (pk if q == "all" else (pkq := universe_purities(d, Us, key, edges))).round(5).tolist(),
                                            "half_spread": half_spread(pk if q == "all" else pkq).round(6).tolist()} for q, Us in C[s].items()}})
        pf, sf = np.array(p["full_all"]["purity"]), np.array(p["full_all"]["stat"])
        for s in SUR:                                   # surrogate (its own rows) over the full simulation (all its reconstructed events)
            ps_, ss_ = np.array(p[s]["purity"]), np.array(p[s]["stat"])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = ps_ / pf
                p[s]["over_full_all"] = r.round(5).tolist()
                p[s]["over_full_all_stat"] = (r * np.hypot(ss_ / ps_, sf / pf)).round(5).tolist()
                p[s]["over_full_all_model"] = np.abs(r * np.array(p[s]["epistemic_rel"])).round(5).tolist()
                p[s]["over_full_all_total"] = np.hypot(r * np.hypot(ss_ / ps_, sf / pf), r * np.array(p[s]["epistemic_rel"])).round(5).tolist()
        pur[key] = p
    out["efficiency_vs_true"] = eff; out["purity_vs_reco"] = pur

    # ---------------- cut flow with the efficiency and the purity on one axes: the closure test only, GENIE full simulation and
    # GENIE + surrogate. The GENIE full simulation is drawn over all its reconstructed events (NC, wrong-sign, other flavours and
    # out-of-volume included). Canvas 4.0 x 3.0 in, print-size fonts (datacomp_genie_nuwro_all.py, 2026-10-22 version).
    x = np.arange(len(stages))
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(4.0, 4.8), sharex=True, height_ratios=(3, 1.9), constrained_layout=True)   # + residual panel (user request 2026-10-28), the canvas of cutflow_closure.png
    for n in ("full", "gsur"):
        k = "full_all" if n == "full" else n
        ax.plot(x, [100 * r["efficiency"] for r in flows[k]], marker="o", markersize=3.5, color=col[n], label=LABEL[n], **line[n])   # no final values in the legend (user request 2026-10-19)
        ax.plot(x, [100 * (r["purity"] or 0) for r in flows[k]], marker="s", markersize=3.5, markerfacecolor="white", markeredgewidth=1.1, color=col[n], **line[n])
    e1, p1 = 100 * flows["full_all"][1]["efficiency"], 100 * flows["full_all"][1]["purity"]
    ax.annotate("signal efficiency", (1, e1), xytext=(-0.1, e1 + 10), fontsize=8.5, color="0.25", arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.8))   # above the first points, clear of the rising purity curve on the 4 x 3 in canvas
    ax.annotate("purity", (1, p1), xytext=(0.3, p1 - 21), fontsize=8.5, color="0.25", arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.8))   # below the curve, clear of the legend
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", marker="o", markersize=3.5, linewidth=1.2), Line2D([], [], color="0.3", marker="s", markersize=3.5, markerfacecolor="white", markeredgewidth=1.1, linewidth=1.2)]
    labels += ["signal efficiency after the cut", "purity after the cut"]   # the marker entries without a "filled circles:" / "open squares:" prefix (user request 2026-10-19)
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=8, labelspacing=0.3, handlelength=1.6)
    ax.set_ylim(0, 100); ax.set_ylabel("efficiency, purity (%)", fontsize=10); ax.set_title("Efficiency and purity after each cut", loc="left", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=9); tidy(ax)   # no figure title (user request 2026-10-16)
    # residual panel (drawing code of cutflow_closure.py): GENIE + surrogate minus the full simulation per stage; thick bars statistical
    # (both samples), thin capped bars statistical (+) epistemic
    axd.axhline(1, color="0.6", linewidth=0.8)
    for q, marker, mfc, dx in (("efficiency", "o", None, -0.1), ("purity", "s", "white", 0.1)):
        d, s_stat, s_tot = (np.array(resid[q][k]) for k in ("ratio", "relative_sigma_stat", "relative_sigma_total"))
        axd.errorbar(x + dx, d, yerr=s_tot, fmt="none", ecolor=col["gsur"], elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=3)
        axd.errorbar(x + dx, d, yerr=s_stat, fmt=marker, color=col["gsur"], markersize=3.5, markerfacecolor=mfc or col["gsur"], markeredgewidth=1.1, elinewidth=2.0, capsize=0, linestyle="none", zorder=4)
    lim = 1.15 * max(np.abs(np.array(resid[q]["relative_difference"])).max() + np.array(resid[q]["relative_sigma_total"]).max() for q in ("efficiency", "purity"))
    axd.set_ylim(1 - lim, 1 + lim); axd.set_ylabel("ratio to full sim.", fontsize=10); axd.tick_params(axis="y", labelsize=8.5)   # surrogate / full (user request 2026-10-28, version 2)   # y label "residual" and no bar-legend text in the panel (user request 2026-10-28); units: percentage points (JSON, README)
    axd.set_xticks(x); axd.set_xticklabels([STAGE_LABEL.get(s, s) for s in stages], rotation=40, ha="right", fontsize=8); tidy(axd)
    fig.savefig(OUT / "cutflow_efficiency_purity.png", dpi=300); plt.close(fig)

    # ---------------- selection efficiency and purity against the muon momentum: uncertainties as error bars, not bands: on every
    # simulated curve, at the bin centres with the samples side by side, thick bars = statistical, thin capped bars = statistical (+)
    # model (half the spread over the four ensemble universes, with the universe weight) for the surrogates; the
    # full simulation has its statistical bar only; the ratio panels are points with the same two bars; the bar meanings go in the
    # caption, not the legend. Print-size fonts, vertical bin-edge labels, short axis labels; all line widths, markers, caps and
    # fonts scaled by 0.75 (sc), the curves by a further 0.6 and the thick bars to 1.3 x sc (datacomp_genie_nuwro_all.py, 2026-10-20 version).
    sc = 0.75                                                      # every line width, marker, cap and font x 0.75 (user request 2026-10-20)
    fs = {k: v * sc for k, v in {"tick": 9, "label": 10, "title": 10.5, "legend": 8.5}.items()}
    curve, thick = 0.6, 1.3 * sc                                   # curves thinner again (user request 2026-10-20: full simulation 1.0 pt, surrogates ~0.75 pt); thick statistical bar ~1 pt
    ln = {k: {**v, "linewidth": v["linewidth"] * sc * curve} for k, v in line.items()}   # the curve styles of style(), scaled
    top_dx = np.linspace(-0.3, 0.3, 1 + len(SUR))      # full simulation first, then the surrogates

    def bars(ax, x, y, st, to, c, marker=None, label=None):
        """Thick bar = statistical, thin capped bar = statistical (+) epistemic; a marker in the ratio panels."""
        ax.errorbar(x, y, yerr=to, fmt="none", ecolor=c, elinewidth=0.7 * sc, capsize=2.0 * sc, capthick=0.7 * sc)
        ax.errorbar(x, y, yerr=st, fmt=marker or "none", markersize=2.8 * sc, color=c, elinewidth=thick, label=label)

    def draw_eff_pur(key: str, sym: str, etop, ebot, ptop, pbot, legend: bool = True, labels: tuple[str, str] = ("", "")) -> None:
        """Efficiency, its ratio, purity, its ratio of one momentum on four stacked axes (one column of the side-by-side figure).
        `legend` False draws neither legend; `labels` = panel letters put in front of the "Efficiency" and "Purity" titles."""
        te, tp = eff[key], pur[key]; edges = te["edges"]; nb = len(edges) - 1; xe = np.arange(nb + 1); mid = xe[:-1] + 0.5
        for a in (etop, ebot, ptop, pbot):
            a.tick_params(labelsize=fs["tick"], width=0.8 * sc, length=3.5 * sc)
            for sp in a.spines.values():
                sp.set_linewidth(0.8 * sc)
        etop.stairs(te["full"]["efficiency"], xe, baseline=None, color=col["full"], label=LABEL["full"], **ln["full"])
        etop.errorbar(mid + top_dx[0], te["full"]["efficiency"], yerr=te["full"]["stat"], fmt="none", ecolor=col["full"], elinewidth=thick)
        pf, sf = np.array(tp["full_all"]["purity"]), np.array(tp["full_all"]["stat"])
        ptop.stairs(pf, xe, baseline=None, color=col["full"], label=LABEL["full"], **ln["full"])
        ptop.errorbar(mid + top_dx[0], pf, yerr=sf, fmt="none", ecolor=col["full"], elinewidth=thick)
        for b in (ebot, pbot):
            b.axhline(1.0, color="0.45", linewidth=0.9 * sc)
        ev, pv, plo = [], [], [pf - sf]
        for s, dx, dxt in zip(SUR, OFFSETS, top_dx[1:]):
            e_, es, et = te[s]["efficiency"], te[s]["stat"], te[s]["total"]
            etop.stairs(e_, xe, baseline=None, color=col[s], label=LABEL[s], **ln[s]); bars(etop, mid + dxt, e_, es, et, col[s])
            r, st, to = te[s]["over_full"], te[s]["over_full_stat"], te[s]["over_full_total"]
            bars(ebot, mid + dx, r, st, to, col[s], marker="s", label=LABEL[s])   # the legend title says "ratio to the full simulation"
            ev += [r + to, r - to]
            ps_, ss_, pt_ = np.array(tp[s]["purity"]), np.array(tp[s]["stat"]), np.array(tp[s]["total"])
            ptop.stairs(ps_, xe, baseline=None, color=col[s], label=LABEL[s], **ln[s]); bars(ptop, mid + dxt, ps_, ss_, pt_, col[s])
            rp, sp, rt = np.array(tp[s]["over_full_all"]), np.array(tp[s]["over_full_all_stat"]), np.array(tp[s]["over_full_all_total"])
            bars(pbot, mid + dx, rp, sp, rt, col[s], marker="s")
            pv += [rp + rt, rp - rt]; plo.append(ps_ - pt_)
        yp_lo = max(0.0, np.floor((np.nanmin(np.concatenate(plo)) - 0.10) * 20) / 20)   # room for the legend below the curves
        etop.set_ylim(0, 1.0); etop.set_ylabel("selection efficiency", fontsize=fs["label"]); etop.set_title(labels[0] + "Efficiency", loc="left", fontsize=fs["title"]); tidy(etop)
        ptop.set_ylim(yp_lo, 1.005); ptop.set_ylabel("purity", fontsize=fs["label"]); ptop.set_title(labels[1] + "Purity", loc="left", fontsize=fs["title"]); tidy(ptop)
        pad = max(0.06, 1.15 * np.nanmax(np.abs(np.concatenate(ev) - 1)))
        ebot.set_ylim(1 - pad, 1 + pad); ebot.set_ylabel("ratio", fontsize=fs["label"]); tidy(ebot); edge_axis(ebot, edges, rf"true muon {sym} (GeV/$c$)", fontsize=fs["tick"], rotation=90)
        pad = max(0.02, 1.15 * np.nanmax(np.abs(np.concatenate(pv) - 1)))
        pbot.set_ylim(1 - pad, 1 + pad); pbot.set_ylabel("ratio", fontsize=fs["label"]); tidy(pbot); edge_axis(pbot, edges, rf"reconstructed muon {sym} (GeV/$c$)", fontsize=fs["tick"], rotation=90)
        for a in (ebot, pbot):
            a.xaxis.label.set_size(fs["label"])
        for a in (etop, ptop):
            a.tick_params(labelbottom=False)
        if legend:
            etop.legend(loc="lower right", frameon=False, fontsize=fs["legend"])
            # no bar-meaning entries in the legend (user request 2026-10-19: that goes in the caption): thick = statistical, thin capped = statistical (+) epistemic
            ptop.legend(*ebot.get_legend_handles_labels(), loc="lower center", frameon=False, fontsize=fs["legend"], labelspacing=0.35, title="ratio panels:\nratio to the full simulation", title_fontsize=fs["legend"], alignment="left")

    # the two momenta side by side for a full-width figure* (7.0 in = the PRD text width), pT left and pL right, each column
    # efficiency / ratio / purity / ratio, the legends in the pT column only
    fig = plt.figure(figsize=(7.0, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=(3, 1.6, 3, 1.6))
    for j, (key, sym) in enumerate((("pT", r"$p_T$"), ("pL", r"$p_L$"))):
        etop = fig.add_subplot(gs[0, j]); ebot = fig.add_subplot(gs[1, j], sharex=etop)
        ptop = fig.add_subplot(gs[2, j]); pbot = fig.add_subplot(gs[3, j], sharex=ptop)
        draw_eff_pur(key, sym, etop, ebot, ptop, pbot, legend=(j == 0), labels=(("(a) ", "(c) "), ("(b) ", "(d) "))[j])   # (a) eff pT, (b) eff pL, (c) pur pT, (d) pur pL
    fig.savefig(OUT / "selection_efficiency_purity_pT_pL.png", dpi=300); plt.close(fig)
    return out


def figure_muon_1d(plt, col, line) -> dict:
    """muon_pT_pL_1d.png: selected events against the reconstructed muon pT and pL (nu-e constraint on the GENIE rows, the NuWro and
    GiBUU rows at their generator's own cross section)."""
    loaded = {"gsur": load_samples(nue_constraint=True)}
    genie_truth = use_genie_truth_tree(loaded["gsur"][0], loaded["gsur"][2])      # version 5: the GENIE surrogate's own Truth tree instead of f_in
    for s in SUR:                                # version 5: NuWro and GiBUU from the geo-sample files, at their own normalisation (no K factor)
        if s in GEO:
            loaded[s] = load_geo(s, loaded["gsur"][0]["full"], loaded["gsur"][1], loaded["gsur"][2])
    S = {s: loaded[s][0] for s in SUR}
    D = loaded["gsur"][1]
    U = {s: load_universes(s, S[s]["sur"]) for s in SUR}
    C = {s: components(S[s]["sur"], U[s]) for s in SUR}        # version 4: the parts of the universe, for the JSON
    K: dict[str, float] = {}                     # version 5: empty (the geo samples carry their own normalisation; no rate match to GENIE to undo)
    norm_text = "; ".join([f"data: raw counts; predictions scaled to the data exposure (POT {D['pot']:.3e}) by POT and the published flux", "GENIE: `wgt` x nu-e flux constraint"]
                          + [f"{GEN_NAME[GENERATOR[s]]}: geo sample, own cross section and composition, gen_weight x POT_data / gen_pot ({loaded[s][2]['pot']['gen_pot']:.6e} POT)" for s in SUR if s in GEO])
    res = {}
    for key, edges in (("pT", PT_PAPER), ("pL", PL_PAPER)):
        hs = {s: reco_hists(S[s], D, key, edges) for s in SUR}
        h = {"data": hs["gsur"]["data"], "full": hs["gsur"]["full"], **{s: hs[s]["sur"] for s in SUR}, **{s + "_bkg": hs[s]["bkg"]["h"] for s in SUR}}
        k = 0 if key == "pT" else 1
        nb = len(edges) - 1
        hk = {s: universe_hists(S[s]["sur"], U[s], k, edges) for s in SUR}      # [4, nb]: the selected surrogate rows of every universe at the data exposure ('all together')
        hk_c = {s: {q: (hk[s] if q == "all" else universe_hists(S[s]["sur"], Us, k, edges)) for q, Us in C[s].items()} for s in SUR}      # version 4: the components
        epi = {s: {"correlated": half_spread(hk[s]), "independent": half_spread(hk[s]), "exists": np.zeros(nb), "minos": np.zeros(nb), "charge": np.zeros(nb)} for s in SUR}
        res[key] = {"edges": edges, **compare(h, epi)}
        for s in SUR:                            # the universes themselves; the total's model part = half the spread of the universe totals
            for q in ("exists", "minos", "charge", "independent"):
                res[key].pop(f"{s}_epistemic_{q}")
            res[key][f"{s}_universes"] = hk[s]
            res[key][f"{s}_universes_total"] = hk[s].sum(1)
            res[key]["totals_in_range"][f"{s}_epistemic"] = float(half_spread(hk[s].sum(1)))
            res[key][f"{s}_components"] = {q: {"universes": h_, "half_spread": half_spread(h_), "total_half_spread": float(half_spread(h_.sum(1)))} for q, h_ in hk_c[s].items()}
    out = {"weights": loaded["gsur"][2]["weights"] + "".join(f"; {GEO_TEXT[s]}: gen_weight (events at gen_pot), no composition weight, no K" for s in SUR if s in GEO),
           "normalisation": norm_text, "K": {GENERATOR[s]: K[s] for s in K},
           "geo_samples": {s: {"set": loaded[s][2]["set"], "pot": loaded[s][2]["pot"], "tables": loaded[s][2]["tables"]} for s in SUR if s in GEO},
           "genie_truth_tree": genie_truth,
           "nue_constraint": {s: loaded[s][2]["nue_constraint"] for s in SUR},
           "band": "statistical (+) model in quadrature for the surrogates (model = half the spread (max - min)/2 over the four ensemble universes of the selected surrogate rows per bin, 'all together': each universe with its own vertex/angle cuts, its own reconstructed momentum and its rows weighted by surrogate_univ_weight[k] = the product of the three flag factors; the background completion is common), statistical for the full simulation; ratios likewise; chi2 with_epistemic = compare() with that half spread as a fully correlated band; *_components = the same with one part of the universe at a time (version 4)",
           "components": COMPONENTS,
           "samples": {"data": LABEL["data"] + " (statistical errors)", "full": LABEL["full"] + ", all reconstructed events", **{s: (GEO_TEXT.get(s, LABEL[s])) + " + background completion" for s in SUR}},
           **res}

    def band(ax, xe, y, e, c, label=None):
        ax.stairs(y + e, xe, baseline=y - e, fill=True, color=c, alpha=0.32, linewidth=0, label=label)

    def draw_1d(key: str, sym: str, log: bool, top, bot, ratio_legend: bool = True, prd: bool = False, legend: bool = True) -> None:
        """Events (top) and ratios to the full simulation (bottom) against the reconstructed momentum `key`, into the two axes
        (prd = the print layout: fonts, markers, spines and ticks x 0.75, curves x 0.75 x 0.6,
        legend without the event counts, no ratio legend, vertical bin-edge labels)."""
        t = res[key]; edges = t["edges"]; nb = len(edges) - 1; xe = np.arange(nb + 1); mid = xe[:-1] + 0.5
        tot = t["totals_in_range"]
        cnt = (lambda text: "") if prd else (lambda text: f" ({text})")
        sc, curve = (0.75, 0.6) if prd else (1.0, 1.0); thick = 1.3 * sc if prd else 1.0
        fs = {k: v * sc for k, v in {"tick": 9, "label": 10, "title": 10.5, "legend": 8.5}.items()} if prd else {"tick": 8, "label": None, "title": None, "legend": 9}
        ln = {k: {**v, "linewidth": v["linewidth"] * sc * curve} for k, v in line.items()}
        band(top, xe, t["full_events"], t["full_stat"], col["full"])
        top.stairs(t["full_events"], xe, baseline=None, color=col["full"], label=LABEL["full"] + cnt(f"{tot['full']:,.0f}"), **ln["full"])
        for s in SUR:
            band(top, xe, t[f"{s}_events"], t[f"{s}_total"], col[s])
            top.stairs(t[f"{s}_events"], xe, baseline=None, color=col[s], label=LABEL[s] + cnt(f"{tot[s]:,.0f} ± {tot[s + '_epistemic']:,.0f} epistemic"), **ln[s])
        top.errorbar(mid, t["data_events"], yerr=t["data_stat"], fmt="o", markersize=3.2 * sc if prd else 4.5, color=col["data"], elinewidth=thick, label=LABEL["data"] + cnt(f"{tot['data']:,.0f}"), zorder=5)
        if log:
            top.set_yscale("log"); top.set_ylim(max(1.0, 0.4 * t["data_events"][t["data_events"] > 0].min()), 4 * max(t["data_events"].max(), t["full_events"].max()))
        elif prd:                                # 2026-10-20 (user): fixed 0-100,000 for the pT panel of muon_pT_pL_1d.png
            top.set_ylim(0, 100_000)
        else:
            top.set_ylim(0, 1.3 * max(t["data_events"].max(), t["full_events"].max(), *(t[f"{s}_events"].max() for s in SUR)))
        top.set_ylabel("events per bin" if prd else "selected events per bin", fontsize=fs["label"]); tidy(top)
        if legend:                               # prd: upper left, over the low bins, with the headroom above
            if prd:                              # 2026-10-20 (user): the prd legend scaled by a further 0.75 (font, handles, markers, spacing)
                leg = 0.75
                top.legend(loc="upper left", frameon=False, fontsize=fs["legend"] * leg, handlelength=2.0 * leg, handletextpad=0.8 * leg,
                           labelspacing=0.5 * leg, borderpad=0.4 * leg, markerscale=leg)
            else:
                top.legend(loc="upper right", frameon=False, fontsize=fs["legend"])
        top.set_title(f"Reconstructed muon {sym}" + ("" if prd else ", paper bins"), loc="left", fontsize=fs["title"])
        bot.axhline(1.0, color=col["full"], linewidth=1.6 * sc * curve)
        vals = [t["data_over_full"] + t["data_over_full_stat"], t["data_over_full"] - t["data_over_full_stat"]]
        for s in SUR:
            r, e = t[f"{s}_over_full"], t[f"{s}_over_full_total"]
            band(bot, xe, r, e, col[s])
            bot.stairs(r, xe, baseline=None, color=col[s], label=f"{LABEL[s]} / full simulation", **ln[s])
            vals += [r + e, r - e]
        bot.errorbar(mid, t["data_over_full"], yerr=t["data_over_full_stat"], fmt="o", markersize=3.0 * sc if prd else 4, color=col["data"], elinewidth=thick, label="data / full simulation", zorder=5)
        pad = max(0.1, 1.12 * np.nanmax(np.abs(np.concatenate(vals) - 1))); bot.set_ylim(1 - pad, 1 + pad)
        bot.set_ylabel("ratio to full sim." if prd else "ratio to the GENIE full simulation", fontsize=fs["label"]); tidy(bot)
        if ratio_legend and not prd:
            bot.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)
        edge_axis(bot, edges, rf"reconstructed muon {sym} (GeV/$c$)" + ("" if prd else ", bin edges"), fontsize=fs["tick"], rotation=90 if prd else 60)
        if prd:
            for a in (top, bot):
                a.tick_params(labelsize=fs["tick"], width=0.8 * sc, length=3.5 * sc)
                for sp in a.spines.values():
                    sp.set_linewidth(0.8 * sc)
            bot.xaxis.label.set_size(fs["label"])

    # pT and pL side by side for a full-width figure* (7.0 in = the PRD text width, 3.5 in tall), each variable a column of
    # events over ratio; one legend (in the pT panel) without the event counts, no ratio legend
    fig = plt.figure(figsize=(7.0, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(3, 1.6))
    for j, (key, sym, log) in enumerate((("pT", r"$p_T$", False), ("pL", r"$p_L$", True))):
        top = fig.add_subplot(gs[0, j]); bot = fig.add_subplot(gs[1, j], sharex=top)
        draw_1d(key, sym, log, top, bot, prd=True, legend=(j == 0))
        top.tick_params(labelbottom=False)
    fig.savefig(OUT / "muon_pT_pL_1d.png", dpi=300); plt.close(fig)   # no figure title; the normalisation and chi2 are in final_figures.json
    return out


def record_inputs() -> dict:
    """data/inputs.json = every input file read (all in data/) with its size, its modification time and, where it is a link, its target."""
    entries = {}
    for p in sorted(READ):
        st = p.stat()
        entries[str(p.relative_to(FINAL))] = {"bytes": st.st_size, "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                                              "target": os.path.realpath(p) if p.is_symlink() else None}
    return {"analysis_code": "scripts/final_figures.py (self-contained)", "sample_set": SAMPLE_SET, "inputs": entries}


def main() -> None:
    missing = [str(p.relative_to(FINAL)) for p in INPUTS if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of the {len(INPUTS)} input files are missing (python scripts/link_inputs.py puts them in data/):\n  " + "\n  ".join(missing))
    for sub in ("configs", "data", "figures"):
        (FINAL / sub).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    col, line = style()
    numbers = {"sample_set": SAMPLE_SET, "figures": {}}
    numbers["figures"]["cutflow_efficiency_purity + selection_efficiency_purity_pT_pL"] = figure_cutflow_and_efficiency_purity(plt, col, line, Line2D)
    numbers["figures"]["muon_pT_pL_1d"] = figure_muon_1d(plt, col, line)
    write_json(OUT / "final_figures.json", numbers)
    (FINAL / "data" / "inputs.json").write_text(json.dumps(record_inputs(), indent=1))
    eff = numbers["figures"]["cutflow_efficiency_purity + selection_efficiency_purity_pT_pL"]["efficiency_overall"]
    tot = {k: numbers["figures"]["muon_pT_pL_1d"][k]["totals_in_range"] for k in ("pT", "pL")}
    print("overall efficiency:", {k: round(100 * v, 2) for k, v in eff.items()})
    print("selected events in the pT range:", {k: round(v) for k, v in tot["pT"].items()})
    fig1 = numbers["figures"]["cutflow_efficiency_purity + selection_efficiency_purity_pT_pL"]
    for s in SUR:
        print(f"{s}: value cuts agree with the nominal per member", fig1["universes"][s]["value_cuts_agree_with_nominal"],
              "; universe weight mean per member", [round(v, 4) for v in fig1["universes"][s]["universe_weight"]["mean"]])
    cu = fig1["cutflow_universes"]["gsur"]
    for q in ("efficiency", "purity_with_background_completion"):
        print(f"cut-flow half spread of the GENIE + surrogate {q} per stage (pt): all together", (100 * np.asarray(cu["half_spread"][q])).round(3).tolist())
        for c, v in cu["components"].items():
            print(f"    {c:10s}", (100 * np.asarray(v["half_spread"][q])).round(3).tolist())
    for key in ("pT", "pL"):
        for s in SUR:
            e = fig1["efficiency_vs_true"][key][s]; pq = fig1["purity_vs_reco"][key][s]; t = tot[key]
            print(f"{key} {s}: efficiency model part per bin (%) all together {(100 * np.asarray(e['model_rel'])).round(2).tolist()}")
            for c, v in e["components"].items():
                if c != "all":
                    print(f"    {c:10s} {(100 * np.asarray(v['model_rel'])).round(2).tolist()}")
            print(f"{key} {s}: purity model part per bin (pt) all together {(100 * np.asarray(pq['epistemic'])).round(2).tolist()}")
            for c, v in pq["components"].items():
                if c != "all":
                    print(f"    {c:10s} {(100 * np.asarray(v['half_spread'])).round(2).tolist()}")
            comps = numbers["figures"]["muon_pT_pL_1d"][key][f"{s}_components"]
            print(f"{key} {s}: selected events {t[s]:.0f}, model part of the total all together {t[s + '_epistemic']:.0f}; components " + ", ".join(f"{c} {v['total_half_spread']:.0f}" for c, v in comps.items() if c != "all"))
    for s in SUR:
        if s in GEO:
            g = fig1["geo_samples"][s]
            print(f"{s}: geo sample {g['set']}, gen_pot {g['pot']['gen_pot']:.6e}, generated signal (Truth tree) {g['generated_signal_weighted']:.1f}")
    gtt = fig1["genie_truth_tree"]
    print(f"gsur: Truth tree {gtt['truth_tree_rows']} rows = f_T {gtt['f_T_row_share']:.7f} of the 30-run population (f_in before {gtt['f_in_inferred_before']:.7f}, ratio {gtt['f_T_over_f_in']:.6f}); "
          f"generated signal {gtt['generated_signal_weighted']['truth_tree']:.1f} vs f_in x official {gtt['generated_signal_weighted']['f_in_x_official_before']:.1f} (ratio {gtt['generated_signal_weighted']['ratio']:.5f}); scale ratio {gtt['scale_to_data']['ratio']:.6f}")
    print("figures in", OUT)


if __name__ == "__main__":
    main()
