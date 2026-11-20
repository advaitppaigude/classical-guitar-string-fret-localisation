
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


def signed_value(raw, bits):
    """Interpret a two's-complement signal as a signed integer."""
    if raw >= (1 << (bits - 1)):
        return raw - (1 << bits)
    return raw


@cocotb.test()
async def test_directed_mac(dut):

    # Start clock: 10 ns period
    cocotb.start_soon(
        Clock(dut.clk, 10, units="ns").start()
    )

    # Initialise controls before first clock edge
    dut.rst.value = 1
    dut.flush.value = 0
    dut.start.value = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    dut.rst.value = 0

    # Quantised B5_F2 operands (already integers)
    features = [2, -3, 0, 0, 0, 0, 0, 0]
    weights  = [4,  5, 0, 0, 0, 0, 0, 0]
    bias = 12

    # Expected score = bias + sum(feature * weight)
    expected = bias + sum(
        f * w for f, w in zip(features, weights)
    )

    # Drive unpacked array ports
    for i in range(8):
        dut.feature_regs[i].value = features[i] & 0x1F
        dut.weights[i].value = weights[i] & 0x1F

    dut.bias.value = bias & 0x3FFF

    # Start transaction
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.start.value = 0

    assert int(dut.busy.value) == 1, \
        "MAC did not become busy"

    # Wait for completion, with timeout
    for cycle in range(12):
        if int(dut.done.value) == 1:
            break
        await FallingEdge(dut.clk)
    else:
        assert False, "MAC did not finish within 12 cycles"

    actual = signed_value(int(dut.score.value), 14)

    assert actual == expected, \
        f"Expected score {expected}, got {actual}"

    assert int(dut.busy.value) == 0, \
        "busy should be low on completion"

    dut._log.info(
        f"PASS: expected={expected}, actual={actual}"
    )


async def await_mac(dut, features, weights, bias):
    dut.features.value = features
    dut.weights.value = weights
    dut.bias.value = bias
    for _ in range(9):
        await RisingEdge(dut.clk)


@cocotb.test()
async def test_exhaustive_mac(dut):
    # start clock for 10 ns period
    cocotb.start_soon(
        Clock(dut.clk, 10, units="ns").start()
    )

    # initialise controls before first clock edge
    dut.rst.value = 1
    dut.flush.value = 0
    dut.start.value = 0

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    dut.rst.value = 0

    dut.biases[0].value = 0
    dut.biases[1].value = 0
    dut.biases[2].value = 0

    for pos in range(8):
        for f in range (-16, 16):
            for w in range(-16, 16):
                actual = await run_mac(dut, features, weights, bias)
                expected = f * w
                features = [0] * 8
                weights = [0] * 8

                features[pos] = f
                weights[pos] = w