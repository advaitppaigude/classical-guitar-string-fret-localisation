
from pathlib import Path
import numpy as np

# Must match the ordering of note_t in SystemVerilog
NOTES = ["E4", "D3", "E3", "F4", "G3"]

FORMATS = [
    (5, 2),
    (6, 3),
    (8, 4),
    (10, 5),
]

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
ROM_DIR = ROOT / "rtl/rom"
ROM_DIR.mkdir(parents=True, exist_ok=True)


def quantise(value, bits, frac_bits):
    scaled = int(round(float(value) * (1 << frac_bits)))

    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1

    return max(minimum, min(maximum, scaled))


def write_mem(path, values, bits):
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4

    with path.open("w") as f:
        for value in values:
            encoded = int(value) & mask
            f.write(f"{encoded:0{digits}X}\n")


def generate_rom(bits, frac_bits):
    acc_bits = 2 * bits + 4

    all_weights = []
    all_biases = []

    for note in NOTES:
        # Adjust filename pattern to match existing exports
        model_path = MODEL_DIR / f"{note}.npz"

        with np.load(model_path) as model:
            weights = model["weights"]
            biases = model["biases"]

        assert weights.shape == (3, 8)
        assert biases.shape == (3,)

        # Classes are stored consecutively:
        # A: weights 0-7, B: 8-15, C: 16-23
        for weight in weights.flatten(order="C"):
            all_weights.append(
                quantise(weight, bits, frac_bits)
            )

        # Biases share the product's fractional precision
        for bias in biases:
            all_biases.append(
                quantise(bias, acc_bits, 2 * frac_bits)
            )

    assert len(all_weights) == 120
    assert len(all_biases) == 15

    prefix = f"b{bits}_f{frac_bits}"

    write_mem(
        ROM_DIR / f"{prefix}_weights.mem",
        all_weights,
        bits
    )

    write_mem(
        ROM_DIR / f"{prefix}_biases.mem",
        all_biases,
        acc_bits
    )

    print(f"Generated {prefix}: 120 weights, 15 biases")


if __name__ == "__main__":
    for bits, frac_bits in FORMATS:
        generate_rom(bits, frac_bits)
