import io
import re
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


st.set_page_config(
    page_title="SDF Dihedral Angle Analyzer",
    page_icon="🧬",
    layout="wide",
)


APP_VERSION = "ver. 1.0"


def parse_torsion_definitions(text: str, numbering_mode: str) -> Tuple[List[Dict], List[str]]:
    """Parse torsion definitions from text.

    Accepted formats:
      torsion_A: 5-6-7-8
      torsion_B: 10, 11, 12, 13
      15 16 17 18
    """
    torsions: List[Dict] = []
    errors: List[str] = []

    use_one_based = numbering_mode.startswith("1")

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            name_part, atoms_part = line.split(":", 1)
            torsion_name = name_part.strip() or f"torsion_{len(torsions) + 1}"
        else:
            atoms_part = line
            torsion_name = f"torsion_{len(torsions) + 1}"

        atom_numbers = [int(x) for x in re.findall(r"\d+", atoms_part)]

        if len(atom_numbers) != 4:
            errors.append(
                f"Line {line_no}: four atom numbers are required, but {len(atom_numbers)} were found: {raw_line}"
            )
            continue

        if use_one_based:
            if any(n < 1 for n in atom_numbers):
                errors.append(f"Line {line_no}: 1-based atom numbers must be >= 1: {raw_line}")
                continue
            atoms_0_based = tuple(n - 1 for n in atom_numbers)
        else:
            atoms_0_based = tuple(atom_numbers)
            if any(n < 0 for n in atoms_0_based):
                errors.append(f"Line {line_no}: 0-based atom indices must be >= 0: {raw_line}")
                continue

        torsions.append(
            {
                "torsion_name": torsion_name,
                "atoms_input": tuple(atom_numbers),
                "atoms_0_based": atoms_0_based,
                "atoms_1_based": tuple(n + 1 for n in atoms_0_based),
                "central_bond_1_based": (atoms_0_based[1] + 1, atoms_0_based[2] + 1),
            }
        )

    return torsions, errors


def read_sdf(uploaded_bytes: bytes, sanitize: bool) -> Tuple[List[Chem.Mol], int]:
    """Read all molecule records from an SDF file."""
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(uploaded_bytes),
        sanitize=sanitize,
        removeHs=False,  # important: preserve atom numbering
        strictParsing=False,
    )

    mols: List[Chem.Mol] = []
    failed = 0
    for mol in supplier:
        if mol is None:
            failed += 1
        else:
            mols.append(mol)
    return mols, failed


def classify_dihedral(angle: float, syn_tol: float, anti_tol: float) -> str:
    """Classify a signed dihedral angle in degrees."""
    abs_angle = abs(angle)

    if abs_angle <= syn_tol:
        return "syn/cis-like"
    if abs(abs_angle - 180.0) <= anti_tol:
        return "anti/trans-like"
    if angle > 0:
        return "gauche-like (+)"
    if angle < 0:
        return "gauche-like (-)"
    return "unclassified"


def make_atom_table(mol: Chem.Mol) -> pd.DataFrame:
    """Create an atom-index preview table for the first molecule."""
    rows = []
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    for atom in mol.GetAtoms():
        idx0 = atom.GetIdx()
        row = {
            "atom_index_1_based": idx0 + 1,
            "atom_index_0_based": idx0,
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
        }
        if conf is not None:
            pos = conf.GetAtomPosition(idx0)
            row.update({"x": pos.x, "y": pos.y, "z": pos.z})
        rows.append(row)

    return pd.DataFrame(rows)


def collect_sdf_properties(mol: Chem.Mol) -> Dict[str, str]:
    """Return SDF properties as a simple dict."""
    props = {}
    for prop_name in mol.GetPropNames():
        try:
            props[f"prop_{prop_name}"] = mol.GetProp(prop_name)
        except Exception:
            continue
    return props


def calculate_dihedrals(
    mols: List[Chem.Mol],
    torsions: List[Dict],
    include_sdf_properties: bool,
    add_classification: bool,
    syn_tol: float,
    anti_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate torsion angles for all molecule records/conformers."""
    result_rows = []
    error_rows = []
    conformer_counter = 0

    for mol_record_no, mol in enumerate(mols, start=1):
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") else f"mol_{mol_record_no}"
        n_atoms = mol.GetNumAtoms()
        n_confs = mol.GetNumConformers()

        if n_confs == 0:
            error_rows.append(
                {
                    "mol_record_no": mol_record_no,
                    "molecule_name": mol_name,
                    "error": "No conformer/3D coordinates were found in this SDF record.",
                }
            )
            continue

        mol_props = collect_sdf_properties(mol) if include_sdf_properties else {}

        for conf_local_no, conf in enumerate(mol.GetConformers(), start=1):
            conformer_counter += 1
            if n_confs == 1:
                conformer_label = mol_name
            else:
                conformer_label = f"{mol_name}_conf{conf.GetId()}"

            for torsion in torsions:
                atoms0 = torsion["atoms_0_based"]
                atoms1 = torsion["atoms_1_based"]

                if max(atoms0) >= n_atoms:
                    error_rows.append(
                        {
                            "mol_record_no": mol_record_no,
                            "molecule_name": mol_name,
                            "conformer_label": conformer_label,
                            "torsion_name": torsion["torsion_name"],
                            "atoms_1_based": "-".join(map(str, atoms1)),
                            "error": f"Atom index is out of range. This molecule has {n_atoms} atoms.",
                        }
                    )
                    continue

                try:
                    angle = float(rdMolTransforms.GetDihedralDeg(conf, *atoms0))
                except Exception as exc:
                    error_rows.append(
                        {
                            "mol_record_no": mol_record_no,
                            "molecule_name": mol_name,
                            "conformer_label": conformer_label,
                            "torsion_name": torsion["torsion_name"],
                            "atoms_1_based": "-".join(map(str, atoms1)),
                            "error": str(exc),
                        }
                    )
                    continue

                row = {
                    "conformer_no": conformer_counter,
                    "mol_record_no": mol_record_no,
                    "conf_local_no": conf_local_no,
                    "molecule_name": mol_name,
                    "conformer_label": conformer_label,
                    "torsion_name": torsion["torsion_name"],
                    "atoms_1_based": "-".join(map(str, atoms1)),
                    "atoms_0_based": "-".join(map(str, atoms0)),
                    "central_bond_1_based": f"{torsion['central_bond_1_based'][0]}-{torsion['central_bond_1_based'][1]}",
                    "angle_deg": angle,
                    "abs_angle_deg": abs(angle),
                    "angle_0_360_deg": angle % 360.0,
                }

                if add_classification:
                    row["class"] = classify_dihedral(angle, syn_tol=syn_tol, anti_tol=anti_tol)

                row.update(mol_props)
                result_rows.append(row)

    return pd.DataFrame(result_rows), pd.DataFrame(error_rows)


def make_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a one-row-per-conformer wide table for easier comparison."""
    if df.empty:
        return df

    index_cols = ["conformer_no", "mol_record_no", "conf_local_no", "molecule_name", "conformer_label"]
    value_cols = ["angle_deg", "abs_angle_deg", "angle_0_360_deg"]

    wide_parts = []
    for value_col in value_cols:
        wide = df.pivot_table(
            index=index_cols,
            columns="torsion_name",
            values=value_col,
            aggfunc="first",
        )
        wide.columns = [f"{col}_{value_col}" for col in wide.columns]
        wide_parts.append(wide)

    wide_df = pd.concat(wide_parts, axis=1).reset_index()
    return wide_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("SDF Dihedral Angle Analyzer")
st.caption(APP_VERSION)

st.write(
    "Upload an SDF file containing multiple conformers, define one or more dihedral angles by four atom numbers, "
    "and calculate the angles for all conformers at once."
)

with st.sidebar:
    st.header("Input settings")

    numbering_mode = st.radio(
        "Atom numbering in your input",
        options=["1-based atom numbers", "0-based RDKit atom indices"],
        index=0,
        help="GaussView, Gaussian input files, and most molecular editors use 1-based atom numbers. RDKit internally uses 0-based atom indices.",
    )

    default_torsions = "torsion_A: 5-6-7-8\n# torsion_B: 10, 11, 12, 13"
    torsion_text = st.text_area(
        "Dihedral definitions",
        value=default_torsions,
        height=150,
        help="Enter one torsion per line. Use the format 'name: atom1-atom2-atom3-atom4'. Lines beginning with # are ignored.",
    )

    st.divider()
    st.header("Options")

    sanitize_sdf = st.checkbox(
        "Sanitize SDF while reading",
        value=False,
        help="Usually leave this off for conformer SDF files from Gaussian/Open Babel. Turning it on may reject unusual valence descriptions.",
    )

    include_sdf_properties = st.checkbox(
        "Include SDF properties in output",
        value=True,
        help="If the SDF records contain properties such as energy, they will be added to the result table.",
    )

    show_atom_preview = st.checkbox(
        "Show atom-index preview for the first molecule",
        value=True,
    )

    add_classification = st.checkbox(
        "Add simple angle classification",
        value=True,
        help="Classifies angles as syn/cis-like, anti/trans-like, gauche-like (+), or gauche-like (-).",
    )

    if add_classification:
        syn_tol = st.slider("syn/cis-like tolerance around 0°", 5.0, 60.0, 30.0, 5.0)
        anti_tol = st.slider("anti/trans-like tolerance around ±180°", 5.0, 60.0, 30.0, 5.0)
    else:
        syn_tol = 30.0
        anti_tol = 30.0


uploaded_file = st.file_uploader("Upload SDF file", type=["sdf"])

torsions, torsion_errors = parse_torsion_definitions(torsion_text, numbering_mode)

if torsion_errors:
    st.error("Some torsion definitions could not be parsed.")
    for err in torsion_errors:
        st.write(f"- {err}")

if torsions:
    with st.expander("Parsed torsion definitions", expanded=False):
        st.dataframe(pd.DataFrame(torsions), use_container_width=True)
else:
    st.warning("No valid torsion definitions were entered.")

if uploaded_file is None:
    st.info("Upload an SDF file to start the calculation.")
    st.stop()

if not torsions:
    st.stop()

uploaded_bytes = uploaded_file.getvalue()

try:
    mols, failed_records = read_sdf(uploaded_bytes, sanitize=sanitize_sdf)
except Exception as exc:
    st.error(f"The SDF file could not be read: {exc}")
    st.stop()

if not mols:
    st.error("No valid molecule records were found in the uploaded SDF file.")
    if failed_records:
        st.write(f"Failed records: {failed_records}")
    st.stop()

num_conformers = sum(mol.GetNumConformers() for mol in mols)

metric_cols = st.columns(4)
metric_cols[0].metric("Valid SDF records", len(mols))
metric_cols[1].metric("Conformers", num_conformers)
metric_cols[2].metric("Torsions", len(torsions))
metric_cols[3].metric("Failed SDF records", failed_records)

if show_atom_preview:
    with st.expander("Atom-index preview of the first molecule", expanded=False):
        atom_df = make_atom_table(mols[0])
        st.dataframe(atom_df, use_container_width=True)
        st.caption(
            "This table is useful for confirming whether your four atom numbers match the SDF atom order. "
            "Hydrogens are not removed, so the original atom order is preserved as much as possible."
        )

result_df, error_df = calculate_dihedrals(
    mols=mols,
    torsions=torsions,
    include_sdf_properties=include_sdf_properties,
    add_classification=add_classification,
    syn_tol=syn_tol,
    anti_tol=anti_tol,
)

if not error_df.empty:
    with st.expander("Calculation warnings/errors", expanded=True):
        st.dataframe(error_df, use_container_width=True)

if result_df.empty:
    st.error("No dihedral angles could be calculated.")
    st.stop()

st.header("Results")

selected_torsions = st.multiselect(
    "Torsions to display",
    options=sorted(result_df["torsion_name"].unique()),
    default=sorted(result_df["torsion_name"].unique()),
)

filtered_df = result_df[result_df["torsion_name"].isin(selected_torsions)].copy()

st.subheader("Long-format table")
st.dataframe(filtered_df, use_container_width=True)

wide_df = make_wide_table(filtered_df)

st.subheader("Wide-format table")
st.dataframe(wide_df, use_container_width=True)

st.download_button(
    label="Download long-format CSV",
    data=dataframe_to_csv_bytes(filtered_df),
    file_name="dihedral_angles_long.csv",
    mime="text/csv",
)

st.download_button(
    label="Download wide-format CSV",
    data=dataframe_to_csv_bytes(wide_df),
    file_name="dihedral_angles_wide.csv",
    mime="text/csv",
)

st.header("Plots")

plot_torsion = st.selectbox(
    "Torsion for plots",
    options=sorted(filtered_df["torsion_name"].unique()),
)
plot_df = filtered_df[filtered_df["torsion_name"] == plot_torsion].copy()

scatter_tooltips = [
    "conformer_no",
    "molecule_name",
    "conformer_label",
    "torsion_name",
    "atoms_1_based",
    alt.Tooltip("angle_deg:Q", format=".2f"),
    alt.Tooltip("angle_0_360_deg:Q", format=".2f"),
]
if "class" in plot_df.columns:
    scatter_tooltips.append("class")

scatter = (
    alt.Chart(plot_df)
    .mark_circle(size=70)
    .encode(
        x=alt.X("conformer_no:O", title="Conformer no."),
        y=alt.Y("angle_deg:Q", title="Signed dihedral angle (deg)", scale=alt.Scale(domain=[-180, 180])),
        tooltip=scatter_tooltips,
    )
    .properties(height=320)
)
st.altair_chart(scatter, use_container_width=True)

hist = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "angle_deg:Q",
            bin=alt.Bin(maxbins=36),
            title="Signed dihedral angle (deg)",
            scale=alt.Scale(domain=[-180, 180]),
        ),
        y=alt.Y("count():Q", title="Count"),
        tooltip=["count()"],
    )
    .properties(height=280)
)
st.altair_chart(hist, use_container_width=True)

if "class" in plot_df.columns:
    st.subheader("Class summary")
    class_summary = (
        plot_df.groupby(["torsion_name", "class"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["torsion_name", "class"])
    )
    st.dataframe(class_summary, use_container_width=True)

st.caption(
    "Angles are calculated with RDKit GetDihedralDeg and are reported as signed values from -180° to +180°. "
    "The 0–360° representation is also included for convenience."
)
