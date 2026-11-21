module full_classifier #(
    parameter int BITS = 6,
    parameter int ACC_BITS = 2 * BITS + 4
)(
    input logic clk,
    input logic rst,
    input logic start,
    input model_pkg::note_t note,
    input logic signed [BITS-1:0] features [0:7],
    output logic [1:0] class_index,
    output logic valid
);

    logic signed [BITS-1:0] weights_a [0:7];
    logic signed [ACC_BITS-1:0] biases_a;
    logic signed [BITS-1:0] weights_b [0:7];
    logic signed [ACC_BITS-1:0] biases_b;
    logic signed [BITS-1:0] weights_c [0:7];
    logic signed [ACC_BITS-1:0] biases_c;

    logic signed [BITS-1:0] weights [0:23];
    logic signed [ACC_BITS-1:0] biases [0:2];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : connect_weights
            assign weights[i]      = weights_a[i];
            assign weights[i + 8]  = weights_b[i];
            assign weights[i + 16] = weights_c[i];
        end
    endgenerate

    assign biases[0] = biases_a;
    assign biases[1] = biases_b;
    assign biases[2] = biases_c;

    load_model #(.BITS(BITS), .ACC_BITS(ACC_BITS)) u_load(.note(note), .weights_a(weights_a), .weights_b(weights_b), .weights_c(weights_c),
    .biases_a(biases_a), .biases_b(biases_b), .biases_c(biases_c));

    parametrised_note_classifier_fsm #(.BITS(BITS), .ACC_BITS(ACC_BITS)) u_classifier(.clk(clk), .rst(rst), .start(start), .feature_regs(features),
     .weights(weights),
     .biases(biases), .class_index(class_index), .valid(valid));

endmodule
