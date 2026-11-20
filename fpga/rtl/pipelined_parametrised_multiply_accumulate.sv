module pipelined_parametrised_multiply_accumulate #(
    parameter int BITS = 5,
    parameter int ACC_BITS = 2*BITS + 4
)(
    input logic clk,
    input logic rst,
    input logic flush,
    input logic start,
    input logic signed [BITS-1:0] feature_regs [0:7],
    input logic signed [BITS-1:0] weights [0:7],
    input logic signed [ACC_BITS-1:0] bias,
    output logic busy,
    output logic done,
    output logic signed [ACC_BITS-1:0] score
);


    logic [2:0] count;

    logic signed [BITS-1:0] captured_feature_regs [0:7];
    logic signed [BITS-1:0] captured_weights [0:7];

    logic signed [(2*BITS)-1:0] product;
    assign product = captured_feature_regs[count] * captured_weights[count];

    logic signed [ACC_BITS:0] extended_sum;

    assign extended_sum = $signed({score[ACC_BITS-1], score}) + 
        $signed({{(ACC_BITS + 1 - 2 * BITS){product_reg[(2 * BITS) - 1]}}, product_reg});

    // needed for pipelined architecture
    logic signed [(2 * BITS) -1:0] product_reg;
    logic product_valid;
    logic drain;


    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst || flush) begin
            count <= 3'd7;
            score <= {ACC_BITS{1'd0}};
            busy <= 1'b0;
            drain <= 1'b0;
            product_valid <= 1'b0;
        end
        else if (busy) begin
            if (drain) begin
                done <= 1'b1;
                drain <= 1'b0;
                busy <= 1'b0;
                product_valid <= 1'b0;
            end
            else begin
                product_reg <= product;
                product_valid <= 1'b1;
                if (count == 3'd0) begin
                    drain <= 1'b1;
                end
                else begin
                    count <= count - 3'd1;
                end
            end
            /*
            forcing saturation instead of allowing wraparound
            
            since for inference it is better to preserve approx. magnitude and sign
            than to allow full wraparound

            eg if trying to store 10000, its better to store 8191 than -6384
            */
            if (product_valid) begin
                if (extended_sum[ACC_BITS] != extended_sum[ACC_BITS-1]) begin
                    // overflow so should saturate according to actual result sign
                    score <= extended_sum[ACC_BITS] ? {1'b1, {(ACC_BITS-1){1'b0}}} :
                                                        {1'b0, {(ACC_BITS-1){1'b1}}};
                end
                else begin
                    score <= extended_sum[ACC_BITS-1:0];
                end
            end
        end
        else if (start) begin
            captured_feature_regs <= feature_regs;
            captured_weights <= weights;
            count <= 3'd7;
            score <= bias;
            busy <= 1'b1;
            product_valid <= 1'b0;
            drain <= 1'b0;
        end
    end

endmodule
