#!/usr/bin/env bash

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo "Run this script with: source create_vitis_workspace.sh" >&2
    exit 1
fi

mapfile -d '' -t vivado_setups < <(
    find /opt /tools /vivado "$HOME" -iname settings64.sh -ipath '*/vivado/*' -print0 2>/dev/null
)

mapfile -d '' -t vitis_setups < <(
    find /opt /tools /vivado "$HOME" -iname settings64.sh -ipath '*/vitis/*' -print0 2>/dev/null
)

source "${vivado_setups[0]}" || return 1
source "${vitis_setups[0]}" || return 1

command vivado -version || { echo "The setup script did not make vivado available" >&2; return 1; }
command vitis -v || { echo "The setup script did not make vitis available." >&2; return 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Settings updated. Beggining project setup..."

if ! command vivado -mode batch -source "$script_dir/circuits/microblaze/kria-bd-build.tcl" -tclargs "microblaze_build"; then 
    echo "Project most likely setup already. See error above just in case." >&2
fi

command vitis -s "$script_dir/vitis/vitis_project.py" || { echo "An error occurred while running the Vitis project script." >&2; return 1; }