module de10lite_top_regression #(
    parameter int BITS = 6,
    parameter logic [7:0] TEST_COUNT = (BITS == 10) ? 8'd180 : 8'd150,
    parameter TEST_ROM_FILE = (BITS == 10) ? "rom/b10_f5_regression_with_later.mem" : "rom/b6_f3_regression.mem"
) (
    input logic clk,
    input logic [1:0] KEY,
    output logic [7:0] HEX0,
    output logic [7:0] HEX1,
    output logic [7:0] HEX2
);

    logic run_start;
    logic rst;
    logic [7:0] correct;
    logic done;
    logic [3:0] ones, tens, hundreds;

    de10lite_regression_fsm #(.BITS(BITS), .TEST_COUNT(TEST_COUNT), .TEST_ROM_FILE(TEST_ROM_FILE)) u_fsm (
        .clk(clk),
        .rst(rst),
        .run_start(run_start),
        .correct(correct),
        .done(done)
    );

    assign ones = 4'(correct % 8'd10);
    assign tens = 4'((correct / 8'd10) % 8'd10);
    assign hundreds = 4'(correct / 8'd100);

    // Common-anode display: bits 6:0 are g through a, and bit 7 is the decimal point.
    function automatic logic [7:0] decimal_segments(input logic [3:0] digit);
        case (digit)
            4'd0: decimal_segments = 8'hC0;
            4'd1: decimal_segments = 8'hF9;
            4'd2: decimal_segments = 8'hA4;
            4'd3: decimal_segments = 8'hB0;
            4'd4: decimal_segments = 8'h99;
            4'd5: decimal_segments = 8'h92;
            4'd6: decimal_segments = 8'h82;
            4'd7: decimal_segments = 8'hF8;
            4'd8: decimal_segments = 8'h80;
            4'd9: decimal_segments = 8'h90;
            default: decimal_segments = 8'hFF;
        endcase
    endfunction

    assign HEX0 = done ? decimal_segments(ones) : 8'hFF;
    assign HEX1 = done ? decimal_segments(tens) : 8'hFF;
    assign HEX2 = done ? decimal_segments(hundreds) : 8'hFF;

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
        run_start <= key1_sync & ~key1_sync_prev;
    end

endmodule
module de10lite_regression_fsm #(
    parameter int BITS = 6,
    parameter logic [7:0] TEST_COUNT = 8'd150,
    parameter TEST_ROM_FILE = "rom/b6_f3_regression.mem"
) (
    input logic clk,
    input logic rst,
    input logic run_start,
    output logic [7:0] correct,
    output logic done
);

    localparam int PAYLOAD_BITS = 8 * BITS;
    localparam int WORD_BITS = ((PAYLOAD_BITS + 5 + 3) / 4) * 4;

    logic [WORD_BITS-1:0] test_rom [0:TEST_COUNT-1];
    logic [PAYLOAD_BITS+4:0] test_case;
    logic signed [BITS-1:0] features [0:7];
    logic valid;
    logic active;
    logic [7:0] test_count;
    model_pkg::note_t note;
    logic start_classification;
    logic [1:0] actual_class_index;

    initial $readmemh(TEST_ROM_FILE, test_rom);

    assign test_case = test_rom[test_count][PAYLOAD_BITS+4:0];
    assign note = model_pkg::note_t'(test_case[PAYLOAD_BITS+2 +: 3]);

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : unpack_features
            assign features[i] = test_case[BITS*i +: BITS];
        end
    endgenerate

    full_classifier #(.BITS(BITS)) u_full_classifier (
        .clk(clk),
        .rst(rst),
        .start(start_classification),
        .note(note),
        .features(features),
        .class_index(actual_class_index),
        .valid(valid)
    );

    always_ff @(posedge clk) begin
        start_classification <= 1'b0;
        if (rst) begin
            active <= 1'b0;
            correct <= 8'd0;
            test_count <= 8'd0;
            done <= 1'b0;
        end
        else if (run_start && !active) begin
            active <= 1'b1;
            correct <= 8'd0;
            test_count <= 8'd0;
            done <= 1'b0;
            start_classification <= 1'b1;
        end
        else if (active && valid) begin
            if (actual_class_index == test_case[PAYLOAD_BITS +: 2]) begin
                correct <= correct + 8'd1;
            end
            if (test_count == TEST_COUNT - 8'd1) begin
                active <= 1'b0;
                done <= 1'b1;
            end
            else begin
                // The async ROM updates before the classifier samples start next edge.
                test_count <= test_count + 8'd1;
                start_classification <= 1'b1;
            end
        end
    end

endmodule
