import re
from io import StringIO

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


APP_VERSION = "ver. 1.1"


# =========================
# Utility functions
# =========================

def parse_atom_numbers_from_line(line, expected_count):
    """
    Parse atom numbers from one line.

    Accepted formats:
        torsion_A: 5-6-7-8
        torsion_A: 5, 6, 7, 8
        torsion_A: 5 6 7 8
        5-6-7-8

    Returns:
        name, atom_numbers
    """
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    if ":" in line:
        name_part, atom_part = line.split(":", 1)
        name = name_part.strip()
    else:
        name = None
        atom_part = line

    nums = re.findall(r"\d+", atom_part)
    if len(nums) != expected_count:
        raise ValueError(
            f"Line '{line}' must contain exactly {expected_count} atom numbers."
        )

    atom_numbers = [int(x) for x in nums]

    return name, atom_numbers


def parse_definitions(text, expected_count, default_prefix):
    """
    Parse multiple definition lines.
    """
    definitions = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parsed = parse_atom_numbers_from_line(line, expected_count)
        if parsed is None:
            continue

        name, atom_numbers = parsed

        if not name:
            name = f"{default_prefix}_{len(definitions) + 1}"

        definitions.append(
            {
                "name": name,
                "atoms_input": atom_numbers,
                "line_no": line_no,
            }
        )

    return definitions


def convert_to_zero_based(atom_numbers, numbering_mode):
    """
    Convert user-specified atom numbers to RDKit 0-based atom indices.
    """
    if numbering_mode == "1-based atom numbers":
        return [x - 1 for x in atom_numbers]

    return atom_numbers


def classify_dihedral(angle_deg, syn_tol, anti_tol):
    """
    Simple classification of dihedral angle.

    angle_deg is in -180 to +180.
    """
    angle = angle_deg

    if abs(angle) <= syn_tol:
        return "syn/cis-like"

    if abs(abs(angle) - 180.0) <= anti_tol:
        return "anti/trans-like"

    if angle > 0:
        return "gauche-like (+)"

    return "gauche-like (-)"


def get_mol_name(mol, index):
    """
    Get conformer name from SDF.
    """
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()

    return f"conf_{index + 1}"


def mol_properties_to_dict(mol):
    """
    Extract SDF properties.
    """
    props = {}

    for prop_name in mol.GetPropNames():
        try:
            props[prop_name] = mol.GetProp(prop_name)
        except Exception:
            props[prop_name] = ""

    return props


def make_atom_index_preview(mol):
    """
    Create atom-index preview table for the first molecule.
    """
    rows = []

    for atom in mol.GetAtoms():
        idx0 = atom.GetIdx()
        rows.append(
            {
                "0-based RDKit index": idx0,
                "1-based atom number": idx0 + 1,
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "degree": atom.GetDegree(),
                "formal_charge": atom.GetFormalCharge(),
            }
        )

    return pd.DataFrame(rows)


def calculate_dihedrals(
    mols,
    dihedral_definitions,
    numbering_mode,
    include_props,
    add_classification,
    syn_tol,
    anti_tol,
):
    """
    Calculate dihedral angles for all conformers.
    """
    rows = []

    for i, mol in enumerate(mols):
        if mol is None:
            continue

        if mol.GetNumConformers() == 0:
            continue

        conf = mol.GetConformer()
        mol_name = get_mol_name(mol, i)

        props = mol_properties_to_dict(mol) if include_props else {}

        for definition in dihedral_definitions:
            atom_indices = convert_to_zero_based(
                definition["atoms_input"], numbering_mode
            )

            if any(idx < 0 or idx >= mol.GetNumAtoms() for idx in atom_indices):
                raise ValueError(
                    f"Invalid atom number in {definition['name']}: "
                    f"{definition['atoms_input']} for molecule {mol_name}. "
                    f"This molecule has {mol.GetNumAtoms()} atoms."
                )

            angle = rdMolTransforms.GetDihedralDeg(conf, *atom_indices)

            row = {
                "conformer_index": i + 1,
                "conformer_name": mol_name,
                "measurement_type": "dihedral",
                "name": definition["name"],
                "atoms_input": "-".join(map(str, definition["atoms_input"])),
                "atoms_0_based": "-".join(map(str, atom_indices)),
                "angle_deg": angle,
                "abs_angle_deg": abs(angle),
                "angle_360_deg": angle % 360,
            }

            if add_classification:
                row["classification"] = classify_dihedral(
                    angle, syn_tol=syn_tol, anti_tol=anti_tol
                )

            row.update(props)
            rows.append(row)

    return pd.DataFrame(rows)


def calculate_distances(
    mols,
    distance_definitions,
    numbering_mode,
    include_props,
):
    """
    Calculate atom-atom distances for all conformers.
    """
    rows = []

    for i, mol in enumerate(mols):
        if mol is None:
            continue

        if mol.GetNumConformers() == 0:
            continue

        conf = mol.GetConformer()
        mol_name = get_mol_name(mol, i)

        props = mol_properties_to_dict(mol) if include_props else {}

        for definition in distance_definitions:
            atom_indices = convert_to_zero_based(
                definition["atoms_input"], numbering_mode
            )

            if any(idx < 0 or idx >= mol.GetNumAtoms() for idx in atom_indices):
                raise ValueError(
                    f"Invalid atom number in {definition['name']}: "
                    f"{definition['atoms_input']} for molecule {mol_name}. "
                    f"This molecule has {mol.GetNumAtoms()} atoms."
                )

            distance = rdMolTransforms.GetBondLength(
                conf, atom_indices[0], atom_indices[1]
            )

            row = {
                "conformer_index": i + 1,
                "conformer_name": mol_name,
                "measurement_type": "distance",
                "name": definition["name"],
                "atoms_input": "-".join(map(str, definition["atoms_input"])),
                "atoms_0_based": "-".join(map(str, atom_indices)),
                "distance_A": distance,
            }

            row.update(props)
            rows.append(row)

    return pd.DataFrame(rows)


def make_wide_table(df, value_column):
    """
    Convert long table to wide table.
    """
    if df.empty:
        return pd.DataFrame()

    base_cols = ["conformer_index", "conformer_name"]

    wide = df.pivot_table(
        index=base_cols,
        columns="name",
        values=value_column,
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None

    return wide


def convert_df_to_csv_bytes(df):
    """
    Convert dataframe to CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="SDF Dihedral Angle Analyzer",
    page_icon="🧬",
    layout="wide",
)

st.title("SDF Dihedral Angle Analyzer")
st.caption(APP_VERSION)

st.write(
    "Upload an SDF file containing multiple conformers, define one or more "
    "dihedral angles and/or atom-atom distances, and calculate them for all "
    "conformers at once."
)

# Sidebar
st.sidebar.header("Input settings")

numbering_mode = st.sidebar.radio(
    "Atom numbering in your input",
    ["1-based atom numbers", "0-based RDKit atom indices"],
    index=0,
    help=(
        "Use 1-based atom numbers when you specify atom numbers from Gaussian, "
        "GaussView, Chem3D, Avogadro, etc. Use 0-based indices only when you "
        "intend to use RDKit atom indices directly."
    ),
)

dihedral_text = st.sidebar.text_area(
    "Dihedral definitions",
    value="torsion_A: 5-6-7-8\n# torsion_B: 10, 11, 12, 13",
    height=120,
    help=(
        "Define one dihedral per line using four atom numbers. "
        "Examples: torsion_A: 5-6-7-8 or torsion_B: 10, 11, 12, 13."
    ),
)

distance_text = st.sidebar.text_area(
    "Distance definitions",
    value="distance_A: 5-8\n# distance_B: 10, 13",
    height=100,
    help=(
        "Define one atom-atom distance per line using two atom numbers. "
        "Examples: distance_A: 5-8 or distance_B: 10, 13."
    ),
)

st.sidebar.header("Options")

sanitize = st.sidebar.checkbox(
    "Sanitize SDF while reading",
    value=False,
    help=(
        "If reading fails, try turning this off. For conformer geometry analysis, "
        "sanitization is often not necessary."
    ),
)

include_props = st.sidebar.checkbox(
    "Include SDF properties in output",
    value=True,
)

show_atom_preview = st.sidebar.checkbox(
    "Show atom-index preview for the first molecule",
    value=True,
)

add_classification = st.sidebar.checkbox(
    "Add simple dihedral angle classification",
    value=True,
)

syn_tol = st.sidebar.slider(
    "syn/cis-like tolerance around 0°",
    min_value=5.0,
    max_value=60.0,
    value=30.0,
    step=5.0,
)

anti_tol = st.sidebar.slider(
    "anti/trans-like tolerance around ±180°",
    min_value=5.0,
    max_value=60.0,
    value=30.0,
    step=5.0,
)

uploaded_file = st.file_uploader(
    "Upload SDF file",
    type=["sdf"],
)

# Parse definitions
try:
    dihedral_definitions = parse_definitions(
        dihedral_text,
        expected_count=4,
        default_prefix="torsion",
    )
except Exception as e:
    st.error(f"Error in dihedral definitions: {e}")
    st.stop()

try:
    distance_definitions = parse_definitions(
        distance_text,
        expected_count=2,
        default_prefix="distance",
    )
except Exception as e:
    st.error(f"Error in distance definitions: {e}")
    st.stop()

with st.expander("Parsed measurement definitions", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dihedral angles")
        if dihedral_definitions:
            st.dataframe(pd.DataFrame(dihedral_definitions), use_container_width=True)
        else:
            st.info("No dihedral definitions were provided.")

    with col2:
        st.subheader("Atom-atom distances")
        if distance_definitions:
            st.dataframe(pd.DataFrame(distance_definitions), use_container_width=True)
        else:
            st.info("No distance definitions were provided.")

if not uploaded_file:
    st.info("Upload an SDF file to start the calculation.")
    st.stop()

if not dihedral_definitions and not distance_definitions:
    st.warning("Please define at least one dihedral angle or atom-atom distance.")
    st.stop()

# Read SDF
sdf_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
supplier = Chem.ForwardSDMolSupplier(
    StringIO(sdf_text),
    sanitize=sanitize,
    removeHs=False,
)

mols = [mol for mol in supplier if mol is not None]

if len(mols) == 0:
    st.error("No valid molecules were read from the SDF file.")
    st.stop()

st.success(f"Successfully read {len(mols)} conformer(s) from the SDF file.")

first_mol = mols[0]

if show_atom_preview:
    with st.expander("Atom-index preview for the first molecule", expanded=True):
        st.write(
            "Use this table to confirm whether your atom numbering matches "
            "the SDF atom order."
        )
        atom_df = make_atom_index_preview(first_mol)
        st.dataframe(atom_df, use_container_width=True)

# Calculate
try:
    dihedral_df = calculate_dihedrals(
        mols=mols,
        dihedral_definitions=dihedral_definitions,
        numbering_mode=numbering_mode,
        include_props=include_props,
        add_classification=add_classification,
        syn_tol=syn_tol,
        anti_tol=anti_tol,
    )

    distance_df = calculate_distances(
        mols=mols,
        distance_definitions=distance_definitions,
        numbering_mode=numbering_mode,
        include_props=include_props,
    )

except Exception as e:
    st.error(f"Calculation error: {e}")
    st.stop()

# Results
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Dihedral angles",
        "Atom-atom distances",
        "Wide tables",
        "Plots",
    ]
)

with tab1:
    st.header("Dihedral angles")

    if dihedral_df.empty:
        st.info("No dihedral angles were calculated.")
    else:
        st.dataframe(dihedral_df, use_container_width=True)

        st.download_button(
            label="Download dihedral angles as CSV",
            data=convert_df_to_csv_bytes(dihedral_df),
            file_name="dihedral_angles_long.csv",
            mime="text/csv",
        )

with tab2:
    st.header("Atom-atom distances")

    if distance_df.empty:
        st.info("No atom-atom distances were calculated.")
    else:
        st.dataframe(distance_df, use_container_width=True)

        st.download_button(
            label="Download atom-atom distances as CSV",
            data=convert_df_to_csv_bytes(distance_df),
            file_name="atom_atom_distances_long.csv",
            mime="text/csv",
        )

with tab3:
    st.header("Wide-format tables")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dihedral angles, wide format")
        if dihedral_df.empty:
            st.info("No dihedral data.")
        else:
            dihedral_wide = make_wide_table(dihedral_df, "angle_deg")
            st.dataframe(dihedral_wide, use_container_width=True)

            st.download_button(
                label="Download dihedral wide table as CSV",
                data=convert_df_to_csv_bytes(dihedral_wide),
                file_name="dihedral_angles_wide.csv",
                mime="text/csv",
            )

    with col2:
        st.subheader("Atom-atom distances, wide format")
        if distance_df.empty:
            st.info("No distance data.")
        else:
            distance_wide = make_wide_table(distance_df, "distance_A")
            st.dataframe(distance_wide, use_container_width=True)

            st.download_button(
                label="Download distance wide table as CSV",
                data=convert_df_to_csv_bytes(distance_wide),
                file_name="atom_atom_distances_wide.csv",
                mime="text/csv",
            )

with tab4:
    st.header("Plots")

    if not dihedral_df.empty:
        st.subheader("Dihedral angle distribution")

        chart = (
            alt.Chart(dihedral_df)
            .mark_circle(size=70)
            .encode(
                x=alt.X("conformer_index:Q", title="Conformer index"),
                y=alt.Y("angle_deg:Q", title="Dihedral angle / degree"),
                color=alt.Color("name:N", title="Dihedral"),
                tooltip=[
                    "conformer_index",
                    "conformer_name",
                    "name",
                    "atoms_input",
                    alt.Tooltip("angle_deg:Q", format=".2f"),
                ],
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

    if not distance_df.empty:
        st.subheader("Atom-atom distance distribution")

        chart = (
            alt.Chart(distance_df)
            .mark_circle(size=70)
            .encode(
                x=alt.X("conformer_index:Q", title="Conformer index"),
                y=alt.Y("distance_A:Q", title="Distance / Å"),
                color=alt.Color("name:N", title="Distance"),
                tooltip=[
                    "conformer_index",
                    "conformer_name",
                    "name",
                    "atoms_input",
                    alt.Tooltip("distance_A:Q", format=".3f"),
                ],
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

    if dihedral_df.empty and distance_df.empty:
        st.info("No data to plot.")

st.divider()

st.caption(
    "Notes: Dihedral angles are reported in degrees using RDKit's "
    "GetDihedralDeg function. Atom-atom distances are reported in Å using "
    "the 3D coordinates stored in the SDF file. Hydrogen atoms are retained "
    "while reading the SDF file."
)
