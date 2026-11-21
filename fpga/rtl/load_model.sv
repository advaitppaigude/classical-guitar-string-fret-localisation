/*
stores quantised coefficients for the five note-specific models

ach model contains 24 weights (3 classes, 8 features) and 3 biases

weight and bias widths are parameterised by BITS and ACC_BITS.
combinational model selection supplies coefficients to three parallel MAC units
*/


module load_model # (
    parameter int BITS = 6,
    parameter int ACC_BITS = 2 * BITS + 4,
    parameter WEIGHTS_FILE = (BITS == 10) ? "rom/b10_f5_weights.mem" : "rom/b6_f3_weights.mem",
    parameter BIASES_FILE  = (BITS == 10) ? "rom/b10_f5_biases.mem" : "rom/b6_f3_biases.mem"
)(
    input model_pkg::note_t note,
    output logic signed [BITS-1:0] weights_a [0:7],
    output logic signed [ACC_BITS-1:0] biases_a,
    output logic signed [BITS-1:0] weights_b [0:7],
    output logic signed [ACC_BITS-1:0] biases_b,
    output logic signed [BITS-1:0] weights_c [0:7],
    output logic signed [ACC_BITS-1:0] biases_c
);

    // Quartus requires one-dimensional memories for $readmemh.
    // Each note occupies 24 consecutive weights and three consecutive biases.
    logic signed [BITS-1:0] weight_rom [0:119];
    logic signed [ACC_BITS-1:0] bias_rom [0:14];
    int unsigned weight_base;
    int unsigned bias_base;

    initial begin
        $readmemh(WEIGHTS_FILE, weight_rom);
        $readmemh(BIASES_FILE, bias_rom);
    end

    assign weight_base = 24 * int'(note);
    assign bias_base   = 3 * int'(note);

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : select_weights
            assign weights_a[i] = weight_rom[weight_base + i];
            assign weights_b[i] = weight_rom[weight_base + 8 + i];
            assign weights_c[i] = weight_rom[weight_base + 16 + i];
        end
    endgenerate

    assign biases_a = bias_rom[bias_base];
    assign biases_b = bias_rom[bias_base + 1];
    assign biases_c = bias_rom[bias_base + 2];

endmodule
