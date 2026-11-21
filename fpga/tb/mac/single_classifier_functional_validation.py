"""
purpose of this is to carry out testing to ensure that MAC unit works
will try all 150 input vectors and compare results to python ref model
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

agreement_results = {}
weight_clipping_results = {}
feature_clipping_results = {}

notes = ["E4", "D3", "E3", "F4", "G3"]

ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / "data" / "features.csv")


def quantise(value, total_bits, frac_bits):
    q = int(round(value * (2 ** frac_bits)))

    # range saturation
    min = (-2 ** (total_bits - 1))
    max = ((2** (total_bits - 1)) - 1)
    if (q < min):
        return min
    elif (q > max):
        return max

    return q

async def complete_MAC(dut, mac_duration):
    """Start with preloaded operands; return the completed class index."""
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(mac_duration):
        await FallingEdge(dut.clk)
        if int(dut.valid.value) == 1:
            # returning unsigned class index
            return int(dut.class_index.value)
    raise AssertionError(f"MAC did not finish within {mac_duration} clocks")


def fixed_point_inference(quantised_features, quantised_weights, quantised_biases, acc_bits):
    scores = []
    predicted_class_index = 0

    for i in range(len(quantised_weights)):
        current = quantised_biases[i]

        """
        iterating backwards over features
        because fsm uses down-counter rather than up-counter

        due to saturation behaviour, iterating in ascending order
        may produce different results
        """
        for j in range(len(quantised_features) - 1, -1, -1):
            current += quantised_features[j] * quantised_weights[i][j]

            current = max(
                -(2 ** (acc_bits - 1)),
                min(current, (2 ** (acc_bits - 1)) - 1)
            )

        scores.append(current)

    for i in range(1, len(scores)):
        if scores[i] > scores[predicted_class_index]:
            predicted_class_index = i

    return (scores, predicted_class_index)


@cocotb.test()
async def functional_validation(dut):

    # Start clock: 10 ns period
    cocotb.start_soon(
        Clock(dut.clk, 10, units="ns").start()
    )

    # Initialise controls before first clock edge
    dut.rst.value = 1
    dut.start.value = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    dut.rst.value = 0

    # Parameters are fixed at elaboration; validate only the compiled format.
    feature_bits = len(dut.feature_regs[0])
    fractional_bits = {5: 2, 6: 3, 8: 4, 10: 5}
    assert feature_bits in fractional_bits, f"Unsupported BITS={feature_bits}"
    feature_bits_sweep = [feature_bits]
    feature_frac_sweep = [fractional_bits[feature_bits]]
    total_prediction_checks = 0


    for note in notes:
        agreement_results[note] = []
        weight_clipping_results[note] = []
        feature_clipping_results[note] = []

        print(f"{note}:")
        with np.load(ROOT / "fpga" / "models" / f"{note}.npz") as model:
            feature_columns = model["feature_columns"]
            weights = model["weights"]
            biases = model["biases"]
            means = model["means"]
            scales = model["scales"]


        # Select all valid note recordings
        current_note_data = df[df["note"] == note].dropna(
            subset=list(feature_columns)
        )

        X_test = current_note_data[list(feature_columns)].to_numpy(dtype=float)



        print("Feature range:", np.min((X_test - means) / scales),
            np.max((X_test - means) / scales))

        print("Weight range:", weights.min(), weights.max())
        print("Bias range:", biases.min(), biases.max())

        for i in range(0, len(feature_bits_sweep)):

            FEATURE_BITS = feature_bits_sweep[i]
            FEATURE_FRAC = feature_frac_sweep[i]

            WEIGHT_BITS = feature_bits_sweep[i]
            WEIGHT_FRAC = feature_frac_sweep[i]

            PRODUCT_BITS = FEATURE_BITS + WEIGHT_BITS
            PRODUCT_FRAC = FEATURE_FRAC + WEIGHT_FRAC
            ACC_BITS = len(dut.biases[0])
            assert ACC_BITS == PRODUCT_BITS + 4, "Unexpected RTL accumulator width"

            quantised_biases = [quantise(b, ACC_BITS, PRODUCT_FRAC) for b in biases]
            quantised_weights = [[quantise(w, WEIGHT_BITS, WEIGHT_FRAC) for w in row] for row in weights]

            # int limits
            weight_min = -(2 ** (WEIGHT_BITS - 1))
            weight_max = (2 ** (WEIGHT_BITS - 1)) - 1

            feature_min = -(2 ** (FEATURE_BITS - 1))
            feature_max = (2 ** (FEATURE_BITS - 1)) - 1

            # checking weight clipping
            scaled_weights = np.rint(weights * (2 ** WEIGHT_FRAC))
            clipped_weights = np.sum(
                (scaled_weights < weight_min) |
                (scaled_weights > weight_max)
            )


            # checking feature clipping
            X_std = (X_test - means) / scales
            scaled_features = np.rint(X_std * (2 ** FEATURE_FRAC))

            clipped_features = np.sum(
                (scaled_features < feature_min) |
                (scaled_features > feature_max)
            )

            matches = 0

            for filename, features in zip(current_note_data["filename"], X_test):
                features_std = (features - means) / scales

                quantised_features = [quantise(f, FEATURE_BITS, FEATURE_FRAC) for f in features_std]

                ref_scores, ref_predicted_index = fixed_point_inference(quantised_features, quantised_weights, quantised_biases, ACC_BITS)

                operand_mask = (1 << FEATURE_BITS) - 1
                for index, value in enumerate(quantised_features):
                    dut.feature_regs[index].value = value & operand_mask

                for class_index, class_weights in enumerate(quantised_weights):
                    for index, value in enumerate(class_weights):
                        # Each class occupies eight consecutive coefficient entries.
                        dut.weights[class_index * 8 + index].value = value & operand_mask
                    dut.biases[class_index].value = quantised_biases[class_index] & ((1 << ACC_BITS) - 1)

                actual_index = await complete_MAC(dut, 12)
                assert actual_index == ref_predicted_index, (
                    f"{filename}: expected class {ref_predicted_index}, "
                    f"actual {actual_index}"
                )

                matches += 1
                total_prediction_checks += 1

            print(f"Format {FEATURE_BITS}/{FEATURE_FRAC} | Agreement: {matches}/{len(X_test)}")
            print("Clipped weights:", clipped_weights, "/", weights.size)
            print(f"Clipped features: {clipped_features}/{X_std.size}")
            print()

    dut._log.info("Validated %d exact classifier predictions", total_prediction_checks)
