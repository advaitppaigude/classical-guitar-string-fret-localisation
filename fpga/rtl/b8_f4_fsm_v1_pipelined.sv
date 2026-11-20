/*
overall architecture:

3 separate pipelined_parametrised_multiply_accumulate modules that work in parallel
to calculate the scores for each of the 3 possible classes

comparator to compare results and output index of biggest one
once MAC calculations have finished

*/


module b8_f4_fsm_v1_pipelined #(
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

    logic mac_a_busy;
    logic mac_a_done;
    logic signed [ACC_BITS-1:0] mac_a_score;
    logic mac_b_busy;
    logic mac_b_done;
    logic signed [ACC_BITS-1:0] mac_b_score;
    logic mac_c_busy;
    logic mac_c_done;
    logic signed [ACC_BITS-1:0] mac_c_score;

    pipelined_parametrised_multiply_accumulate #(.BITS(BITS), .ACC_BITS(ACC_BITS)) u_mac_a(.clk(clk), .rst(rst), .flush(1'b0), .start(start),
    .feature_regs(feature_regs), .weights(weights[0:7]), .bias(biases[0]),
    .busy(mac_a_busy), .done(mac_a_done), .score(mac_a_score));

    pipelined_parametrised_multiply_accumulate #(.BITS(BITS), .ACC_BITS(ACC_BITS)) u_mac_b(.clk(clk), .rst(rst), .flush(1'b0), .start(start),
    .feature_regs(feature_regs), .weights(weights[8:15]), .bias(biases[1]),
    .busy(mac_b_busy), .done(mac_b_done), .score(mac_b_score));

    pipelined_parametrised_multiply_accumulate #(.BITS(BITS), .ACC_BITS(ACC_BITS)) u_mac_c(.clk(clk), .rst(rst), .flush(1'b0), .start(start),
    .feature_regs(feature_regs), .weights(weights[16:23]), .bias(biases[2]),
    .busy(mac_c_busy), .done(mac_c_done), .score(mac_c_score));

    logic [1:0] biggest;
    compare #(.BITS(ACC_BITS)) u_compare(.a(mac_a_score), .b(mac_b_score), .c(mac_c_score), .biggest(biggest));

    always_ff @(posedge clk) begin
        valid <= 1'b0;
        if (rst) begin
            class_index <= 2'b11;
        end
        else if (mac_a_done && mac_b_done && mac_c_done) begin
            valid <= 1'b1;
            class_index <= biggest;
        end
    end

endmodule
