import re
from io import StringIO

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import Draw


APP_VERSION = "ver. 1.2"


# =========================
# Utility functions
# =========================

def parse_atom_numbers_from_line(line, expected_count):
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
    if numbering_mode == "1-based atom numbers":
        return [x - 1 for x in atom_numbers]

    return atom_numbers


def classify_dihedral(angle_deg, syn_tol, anti_tol):
    if abs(angle_deg) <= syn_tol:
        return "syn/cis-like"

    if abs(abs(angle_deg) - 180.0) <= anti_tol:
        return "anti/trans-like"

    if angle_deg > 0:
        return "gauche-like (+)"

    return "gauche-like (-)"


def get_mol_name(mol, index):
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()

    return f"conf_{index + 1}"


def mol_properties_to_dict(mol):
    props = {}

    for prop_name in mol.GetPropNames():
        try:
            props[prop_name] = mol.GetProp(prop_name)
        except Exception:
            props[prop_name] = ""

    return props


def make_atom_index_preview(mol):
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


def make_numbered_molecule_image(mol, numbering_mode, width=800, height=550):
    """
    Draw the first conformer as a 2D molecule with atom indices.
    The original 3D coordinates are not modified for calculations.
    """
    mol2d = Chem.Mol(mol)

    for atom in mol2d.GetAtoms():
        idx0 = atom.GetIdx()

        if numbering_mode == "1-based atom numbers":
            label = str(idx0 + 1)
        else:
            label = str(idx0)

        atom.SetProp("atomNote", label)

    image = Draw.MolToImage(
        mol2d,
        size=(width, height),
        kekulize=False,
    )

    return image


def calculate_dihedrals(
    mols,
    dihedral_definitions,
    numbering_mode,
    include_props,
    add_classification,
    syn_tol,
    anti_tol,
):
    rows = []

    for i, mol in enumerate(mols):
        if mol is None or mol.GetNumConformers() == 0:
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
                    angle,
                    syn_tol=syn_tol,
                    anti_tol=anti_tol,
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
    rows = []

    for i, mol in enumerate(mols):
        if mol is None or mol.GetNumConformers() == 0:
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
                conf,
                atom_indices[0],
                atom_indices[1],
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
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index=["conformer_index", "conformer_name"],
        columns="name",
        values=value_column,
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None
    return wide


def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def definitions_to_text(definitions):
    lines = []

    for d in definitions:
        atoms = "-".join(map(str, d["atoms_input"]))
        lines.append(f"{d['name']}: {atoms}")

    return "\n".join(lines)


def get_atom_options(mol, numbering_mode):
    options = []

    for atom in mol.GetAtoms():
        idx0 = atom.GetIdx()
        atom_no = idx0 + 1 if numbering_mode == "1-based atom numbers" else idx0
        label = f"{atom_no}: {atom.GetSymbol()}"
        options.append((atom_no, label))

    return options


# =========================
# Session state
# =========================

if "dihedral_defs_interactive" not in st.session_state:
    st.session_state.dihedral_defs_interactive = []

if "distance_defs_interactive" not in st.session_state:
    st.session_state.distance_defs_interactive = []


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
    "Upload an SDF file containing multiple conformers. "
    "The first conformer is drawn with atom numbers, and you can select atoms "
    "for dihedral angles or atom-atom distances using dropdown menus."
)

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

st.sidebar.header("Options")

sanitize = st.sidebar.checkbox(
    "Sanitize SDF while reading",
    value=False,
)

include_props = st.sidebar.checkbox(
    "Include SDF properties in output",
    value=True,
)

show_atom_preview = st.sidebar.checkbox(
    "Show atom-index preview for the first molecule",
    value=False,
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

if not uploaded_file:
    st.info("Upload an SDF file to start.")
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

# =========================
# Molecule drawing and interactive selection
# =========================

st.header("1. Select atoms using the numbered molecule diagram")

col_fig, col_select = st.columns([1.2, 1])

with col_fig:
    st.subheader("First conformer with atom numbers")

mol_image = make_numbered_molecule_image(
    first_mol,
    numbering_mode=numbering_mode,
    width=800,
    height=550,
)

st.image(mol_image, use_container_width=True)

    st.caption(
        "The diagram is generated from the first conformer. "
        "Calculations are performed using the original 3D coordinates in the SDF file."
    )

with col_select:
    st.subheader("Add a measurement")

    atom_options = get_atom_options(first_mol, numbering_mode)
    atom_numbers = [x[0] for x in atom_options]
    atom_labels = {x[0]: x[1] for x in atom_options}

    measurement_type = st.radio(
        "Measurement type",
        ["Dihedral angle", "Atom-atom distance"],
        horizontal=True,
    )

    measurement_name = st.text_input(
        "Measurement name",
        value=(
            f"torsion_{len(st.session_state.dihedral_defs_interactive) + 1}"
            if measurement_type == "Dihedral angle"
            else f"distance_{len(st.session_state.distance_defs_interactive) + 1}"
        ),
    )

    def format_atom(x):
        return atom_labels.get(x, str(x))

    if measurement_type == "Dihedral angle":
        st.write("Select four atoms in the order A–B–C–D.")
        st.caption("The central bond is B–C.")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            a1 = st.selectbox("Atom A", atom_numbers, format_func=format_atom, key="dih_a1")
        with c2:
            a2 = st.selectbox("Atom B", atom_numbers, format_func=format_atom, key="dih_a2")
        with c3:
            a3 = st.selectbox("Atom C", atom_numbers, format_func=format_atom, key="dih_a3")
        with c4:
            a4 = st.selectbox("Atom D", atom_numbers, format_func=format_atom, key="dih_a4")

        if st.button("Add dihedral angle"):
            selected = [a1, a2, a3, a4]

            if len(set(selected)) < 4:
                st.warning("Please select four different atoms.")
            else:
                st.session_state.dihedral_defs_interactive.append(
                    {
                        "name": measurement_name.strip() or f"torsion_{len(st.session_state.dihedral_defs_interactive) + 1}",
                        "atoms_input": selected,
                        "line_no": len(st.session_state.dihedral_defs_interactive) + 1,
                    }
                )
                st.rerun()

    else:
        st.write("Select two atoms for the distance measurement.")

        c1, c2 = st.columns(2)

        with c1:
            a1 = st.selectbox("Atom A", atom_numbers, format_func=format_atom, key="dist_a1")
        with c2:
            a2 = st.selectbox("Atom B", atom_numbers, format_func=format_atom, key="dist_a2")

        if st.button("Add atom-atom distance"):
            selected = [a1, a2]

            if len(set(selected)) < 2:
                st.warning("Please select two different atoms.")
            else:
                st.session_state.distance_defs_interactive.append(
                    {
                        "name": measurement_name.strip() or f"distance_{len(st.session_state.distance_defs_interactive) + 1}",
                        "atoms_input": selected,
                        "line_no": len(st.session_state.distance_defs_interactive) + 1,
                    }
                )
                st.rerun()

    st.divider()

    if st.button("Clear all selected measurements"):
        st.session_state.dihedral_defs_interactive = []
        st.session_state.distance_defs_interactive = []
        st.rerun()

# =========================
# Manual input option
# =========================

st.header("2. Confirm or edit measurement definitions")

st.write(
    "You can use the interactive selections above, or manually edit/add definitions below."
)

default_dihedral_text = definitions_to_text(st.session_state.dihedral_defs_interactive)
default_distance_text = definitions_to_text(st.session_state.distance_defs_interactive)

col_manual1, col_manual2 = st.columns(2)

with col_manual1:
    dihedral_text = st.text_area(
        "Dihedral definitions",
        value=default_dihedral_text,
        height=160,
        help=(
            "One dihedral per line using four atom numbers. "
            "Example: torsion_A: 5-6-7-8"
        ),
    )

with col_manual2:
    distance_text = st.text_area(
        "Distance definitions",
        value=default_distance_text,
        height=160,
        help=(
            "One atom-atom distance per line using two atom numbers. "
            "Example: distance_A: 5-8"
        ),
    )

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
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Dihedral angles")
        if dihedral_definitions:
            st.dataframe(pd.DataFrame(dihedral_definitions), use_container_width=True)
        else:
            st.info("No dihedral definitions.")

    with c2:
        st.subheader("Atom-atom distances")
        if distance_definitions:
            st.dataframe(pd.DataFrame(distance_definitions), use_container_width=True)
        else:
            st.info("No distance definitions.")

if show_atom_preview:
    with st.expander("Atom-index preview for the first molecule", expanded=True):
        atom_df = make_atom_index_preview(first_mol)
        st.dataframe(atom_df, use_container_width=True)

if not dihedral_definitions and not distance_definitions:
    st.info("Add at least one dihedral angle or atom-atom distance.")
    st.stop()

# =========================
# Calculate
# =========================

st.header("3. Results")

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

    c1, c2 = st.columns(2)

    with c1:
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

    with c2:
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
    "Notes: The molecule diagram is shown only to help select atom numbers. "
    "Dihedral angles and distances are calculated from the original 3D coordinates "
    "stored in the SDF file."
)
