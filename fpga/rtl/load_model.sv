/*
stores quantised coefficients for the five note-specific models

ach model contains 24 weights (3 classes, 8 features) and 3 biases

weight and bias widths are parameterised by BITS and ACC_BITS.
combinational model selection supplies coefficients to three parallel MAC units
*/


module load_model # (
    parameter int BITS = 6,
    parameter int ACC_BITS = 2 * BITS + 4,
    parameter string WEIGHTS_FILE = "rom/b6_f3_weights.mem",
    parameter string BIASES_FILE  = "rom/b6_f3_biases.mem"
)(
    input model_pkg::note_t note,
    output logic signed [BITS-1:0] weights_a [0:7],
    output logic signed [ACC_BITS-1:0] biases_a,
    output logic signed [BITS-1:0] weights_b [0:7],
    output logic signed [ACC_BITS-1:0] biases_b,
    output logic signed [BITS-1:0] weights_c [0:7],
    output logic signed [ACC_BITS-1:0] biases_c
);

    /*
    decided to implement a "2D array" structure
    so that the note can be used directly to access models
    and the relevant weights needed for the 3 MAC units
    */
    logic signed [BITS-1:0] weight_rom [0:4][0:23];
    logic signed [ACC_BITS-1:0] bias_rom [0:4][0:2];


    initial begin
        $readmemh(WEIGHTS_FILE, weight_rom);
        $readmemh(BIASES_FILE, bias_rom);
    end

    assign weights_a = weight_rom[note][0:7];
    assign weights_b = weight_rom[note][8:15];
    assign weights_c = weight_rom[note][16:23];
    
    assign biases_a  = bias_rom[note][0];
    assign biases_b  = bias_rom[note][1];
    assign biases_c  = bias_rom[note][2];

endmodule
