module de10lite_top (
    input logic clk,
    input logic [1:0] KEY,
    output logic [2:0] LEDR
);    

    logic valid_flag;
    logic valid;

    logic start;
    logic rst;

    assign LEDR[2] = valid_flag;

    logic signed [5:0] features [0:7];

    assign features[0] = -6'sd2;
    assign features[1] = -6'sd7;
    assign features[2] =  6'sd2;
    assign features[3] = -6'sd7;
    assign features[4] = -6'sd6;
    assign features[5] =  6'sd6;
    assign features[6] = -6'sd2;
    assign features[7] = -6'sd7;

    full_classifier u_full_classifier (
        .clk(clk),
        .rst(rst),
        .start(start),
        .note(model_pkg::E4),
        .features(features),
        .class_index(LEDR[1:0]), // expected is 2'b01
        .valid(valid)
    );


    logic key0_sample;
    logic key1_sample;
    logic key1_sync;
    logic key1_sync_prev;

    always_ff @(posedge clk) begin
        key0_sample <= ~KEY[0];
        rst <= key0_sample;
        key1_sample <= ~KEY[1];
        key1_sync <= key1_sample;
        key1_sync_prev <= key1_sync;
        start <= key1_sync & ~key1_sync_prev;
        if (rst) begin
            valid_flag <= 1'b0;
        end
        else begin
            valid_flag <= valid_flag | valid;
        end
    end

endmodule
