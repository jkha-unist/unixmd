from pathlib import Path
import numpy as np
import pytest
from mqc_runner import MQCArgs, run_case

# Usage: 
# Run pytest --markers to see registered markers
# e.g. Test all mqc runs in the Shin-Metiu model: pytest -m mqc -v
# e.g. Test SHXF runs in the Shin-Metiu model: pytest -m shxf -v

# For a specific mqc run by invoking the exact case ID: pytest -m mqc -k TEST-SHXF-FG-a-
# The ID format is TEST-{MQC name}-{width scheme}_{momentum jump scheme}
# MQC name - BOMD, Eh, SH, SHXF, EhXF, CT
# width scheme - FG, TD
# momentum jump scheme - e+, e-, v+, v-, p+, p-, a+, a-

REF_ROOT = Path("reference")

KEY_RESCALE = ["e", "v", "p", "a"]
KEY_REJECT = ["+", "-"]
KEY_WIDTH = ["FG", "TD"]

# Define the test matrix you care about
ALL_CASES = [
    # BOMD, Eh
    pytest.param(
        MQCArgs(md=0), "TEST-BOMD", 
        marks=(pytest.mark.mqc, pytest.mark.bomd)
    ),
    
    pytest.param(
        MQCArgs(md=1), "TEST-Eh", 
        marks=(pytest.mark.mqc, pytest.mark.eh)
    ),

    # SH
    *[ 
        pytest.param(
            MQCArgs(md=2, rescale=r, reject=j), f"TEST-SH-{KEY_RESCALE[r]}{KEY_REJECT[j]}", 
            marks=(pytest.mark.mqc, pytest.mark.sh)
        ) 
        for r in range(4) for j in range(2)
     ],

    # SHXF
    *[
        pytest.param(
            MQCArgs(md=3, width=w, rescale=r, reject=j), f"TEST-SHXF-{KEY_WIDTH[w]}-{KEY_RESCALE[r]}{KEY_REJECT[j]}",
            marks=(pytest.mark.mqc, pytest.mark.shxf)
        )
        for w in (0, 1) for r in range(4) for j in range(2)
     ],

    # EhXF
    *[
        pytest.param(
            MQCArgs(md=4, width=w, rescale=r, reject=j), f"TEST-EhXF-{KEY_WIDTH[w]}-{KEY_RESCALE[r]}{KEY_REJECT[j]}",
            marks=(pytest.mark.mqc, pytest.mark.ehxf)
        )
        for w in (0, 1) for r in range(4) for j in range(2)
     ],

    # CT
    pytest.param(
        MQCArgs(md=5), "TEST-CT", marks=(pytest.mark.mqc, pytest.mark.ct)
    ),

    # CTv2
    pytest.param(
        MQCArgs(md=6), "TEST-CTv2", marks=(pytest.mark.mqc, pytest.mark.ctv2)
    ),

    # SHXFv2
    *[
        pytest.param(
            MQCArgs(md=7, rescale=r, reject=j), f"TEST-SHXFv2-{KEY_RESCALE[r]}{KEY_REJECT[j]}",
            marks=(pytest.mark.mqc, pytest.mark.shxfv2)
        )
        for r in range(4) for j in range(2)
     ],
]

def _load_numeric(path: Path, tg: str):
    if tg.endswith(".xyz"):
        return np.loadtxt(path, skiprows=2, usecols=(1, 2))
    return np.loadtxt(path, skiprows=1)

def _load_energy_from_movie_xyz(path: Path):
    """Extract energy data from MOVIE.xyz comment lines.

    Format: step=N Ekin=X Epot=X Etot=X E0=X E1=X ...
    Returns array of [step, Ekin, Epot, Etot, E0, E1, ...] per frame.
    """
    energies = []
    with open(path, 'r') as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    frame_lines = nat + 2  # nat + natom line + comment line

    for i in range(0, len(lines), frame_lines):
        comment_line = lines[i + 1].strip()
        # Parse key=value pairs
        row = []
        for token in comment_line.split():
            if '=' in token:
                _, value = token.split('=', 1)
                row.append(float(value))
        energies.append(row)

    return np.array(energies)

def _compare_file(out_file: Path, ref_file: Path, tg: str):

    if tg == "MOVIE.xyz":
        # Compare energy data extracted from MOVIE.xyz comment lines
        out_data = _load_energy_from_movie_xyz(out_file)
        ref_data = _load_energy_from_movie_xyz(ref_file)
    else:
        out_data = _load_numeric(out_file, tg)
        ref_data = _load_numeric(ref_file, tg)

    assert out_data.shape == ref_data.shape, f"Shape mismatch: {tg}"

    # Compare absolute values for phase-dependent quantities (eigenvector phase convention)
    if tg in ["NACME", "DENSITY"]:
        # For DENSITY, coherences can have phase-dependent signs
        out_data = np.abs(out_data)
        ref_data = np.abs(ref_data)

    np.testing.assert_allclose(out_data, ref_data, rtol=1e-7, atol=1e-9)

@pytest.mark.parametrize("args,case_id", ALL_CASES)
def test_mqc_case(args, case_id):

    run_case(args)
    #return
    # Determine what to compare
    # Energy data now in MOVIE.xyz comment line (replaces MDENERGY)
    targets = ["MOVIE.xyz", "FINAL.xyz"]

    if args.md != 0:    # not BOMD
        # DENSITY replaces BOPOP + BOCOH
        targets += ["DENSITY", "NACME"]

    if args.md in (2, 3, 4, 7):   # have the SH feature (SH, SHXF, EhXF, SHXFv2)
        targets += ["SHSTATE", "SHPROB"]

    # Compare test results and the reference
    case_id = Path(case_id)
    for tg in targets:
        if args.md not in (5, 6):    # not CT or CTv2
            _compare_file(case_id / "md" / tg, REF_ROOT / case_id / "md" / tg, tg)
        else:    # CT or CTv2
            _compare_file(case_id / "TRAJ_1" / "md" / tg, REF_ROOT / case_id / "TRAJ_1" / "md" / tg, tg)
            _compare_file(case_id / "TRAJ_2" / "md" / tg, REF_ROOT / case_id / "TRAJ_2" / "md" / tg, tg)

