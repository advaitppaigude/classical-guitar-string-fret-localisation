module multiply_accumulate(
    input logic clk,
    input logic rst,
    input logic flush,
    input logic start,
    input logic signed [4:0] feature_regs [0:7],
    input logic signed [4:0] weights [0:7],
    input logic signed [13:0] bias,
    output logic busy,
    output logic done,
    output logic signed [13:0] score
);
    
    logic [2:0] count;

    logic signed [4:0] captured_feature_regs [0:7];
    logic signed [4:0] captured_weights [0:7];

    logic signed [9:0] product;
    assign product = captured_feature_regs[count] * captured_weights[count];

    logic signed [14:0] extended_sum;

    assign extended_sum = $signed({score[13], score}) + $signed({{5{product[9]}}, product});

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst || flush) begin
            count <= 3'd7;
            score <= 14'd0;
            busy <= 1'b0;
        end
        else if (busy) begin
            if (count == 3'd0) begin
                done <= 1'b1;
                busy <= 1'b0;
            end
            else begin
                count <= count - 3'd1;
            end
            /*
            forcing saturation instead of allowing wraparound
            
            since for inference it is better to preserve approx. magnitude and sign
            than to allow full wraparound

            eg if trying to store 10000, its better to store 8191 than -6384
            */
            if (extended_sum > 15'sd8191) begin
                score <= 14'sd8191;
            end
            else if (extended_sum < -15'sd8192) begin
                score <= -14'sd8192;
            end
            else begin
                score <= extended_sum[13:0];
            end
        end
        else if (start) begin
            captured_feature_regs <= feature_regs;
            captured_weights <= weights;
            count <= 3'd7;
            score <= bias;
            busy <= 1'd1;
        end
    end

endmodule
