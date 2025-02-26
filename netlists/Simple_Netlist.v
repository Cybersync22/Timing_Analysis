module TimingAnalysisCircuit (
    input A, B, Clk,
    output Q
);
    wire C, D, D_int;

    // Logic Gates (Example: AND, OR, NOT)
    and u1 (C, A, B);  // C = A & B
    or  u2 (D, C, B);  // D = C | B
    not u3 (D_int, D); // D_int = ~D

    // D Flip-Flop
    D_FF u4 (
        .D(D_int), 
        .Clk(Clk), 
        .Q(Q)
    );

endmodule

// D Flip-Flop module
module D_FF (
    input D, Clk,
    output reg Q
);
    always @(posedge Clk)
        Q <= D;
endmodule
