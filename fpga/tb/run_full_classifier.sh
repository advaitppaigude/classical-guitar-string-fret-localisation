#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/.venv/bin/activate"

rtl="$repo/fpga/rtl"
tb="$repo/fpga/tb/full_classifier"

if (( $# == 0 )); then
    set -- b5_f2 b6_f3 b8_f4
fi

for format in "$@"; do
    case "$format" in
        b5_f2) bits=5 ;;
        b6_f3) bits=6 ;;
        b8_f4) bits=8 ;;
        b10_f5) bits=10 ;;
        *) echo "Unsupported format $format" >&2; exit 2 ;;
    esac

    run_dir="$tb/sim_build/full_$format"
    mkdir -p "$run_dir/rom"

    # Legacy widths share the B6_F3 filenames; B10_F5 selects its own files.
    rom_name=b6_f3
    if [[ "$format" == b10_f5 ]]; then
        rom_name=b10_f5
    fi
    ln -sfn "$rtl/rom/${format}_weights.mem" "$run_dir/rom/${rom_name}_weights.mem"
    ln -sfn "$rtl/rom/${format}_biases.mem" "$run_dir/rom/${rom_name}_biases.mem"

    make -C "$run_dir" -f "$tb/Makefile" -j2 BITS="$bits"
done
