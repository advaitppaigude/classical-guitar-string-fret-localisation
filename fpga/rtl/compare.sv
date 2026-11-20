module compare #(
    parameter int BITS = 14
)(
    input logic signed [BITS-1:0] a,
    input logic signed [BITS-1:0] b,
    input logic signed [BITS-1:0] c,
    output logic [1:0] biggest
);

    always_comb begin
        if (a >= b && a >= c)
            biggest = 2'd0;
        else if (b >= c)
            biggest = 2'd1;
        else
            biggest = 2'd2;
    end

endmodule
