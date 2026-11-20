// Timing-analysis top: input register bank -> classifier -> output register bank.
// All registers use clk and synchronous active-high rst.
// start is registered with its operands. The wrapper adds two clocks of
// latency relative to the classifier alone; valid is registered with class_index.
module b8_f4_timing_wrapper #(
    parameter int BITS = 8,
    parameter int ACC_BITS = 2 * BITS + 4
)(
    input logic clk,
    input logic rst,
    input logic start,
    input logic signed [BITS-1:0] feature_regs [0:7],
    input logic signed [BITS-1:0] weights [0:23],
    input logic signed [ACC_BITS-1:0] biases [0:2],
    output logic [1:0] class_index,
    output logic valid
);

    logic start_q;
    logic signed [BITS-1:0] features_q [0:7];
    logic signed [BITS-1:0] weights_q [0:23];
    logic signed [ACC_BITS-1:0] biases_q [0:2];
    logic [1:0] core_class_index;
    logic core_valid;

    // Register every transaction input together.
    always_ff @(posedge clk) begin
        if (rst) begin
            start_q <= 1'b0;
            for (int i = 0; i < 8; i++) features_q[i] <= '0;
            for (int i = 0; i < 24; i++) weights_q[i] <= '0;
            for (int i = 0; i < 3; i++) biases_q[i] <= '0;
        end
        else begin
            start_q <= start;
            features_q <= feature_regs;
            weights_q <= weights;
            biases_q <= biases;
        end
    end

    b8_f4_fsm #(.BITS(BITS), .ACC_BITS(ACC_BITS)) u_classifier (
        .clk(clk),
        .rst(rst),
        .start(start_q),
        .feature_regs(features_q),
        .weights(weights_q),
        .biases(biases_q),
        .class_index(core_class_index),
        .valid(core_valid)
    );

    // Keep the result and its validity aligned at the output boundary.
    always_ff @(posedge clk) begin
        if (rst) begin
            class_index <= 2'b11;
            valid <= 1'b0;
        end
        else begin
            class_index <= core_class_index;
            valid <= core_valid;
        end
    end

endmodule
