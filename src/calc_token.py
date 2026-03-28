import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the exact number of tokens in text using OpenAI's tiktoken library
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo")
        
    Returns:
        int: Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Your Verilog code
verilog_code = '''
define WIDTH 32
module fsm(
    clk,
    reset_,
    in_A,
    in_B,
    in_C,
    in_D,
    in_E,
    in_F,
    in_G,
    in_H,
    in_I,
    in_J,
    in_K,
    in_L,
    in_M,
    in_N,
    in_O,
    in_P,
    fsm_out
);
    parameter WIDTH = `WIDTH;
    parameter FSM_WIDTH = 4;

    parameter S0 = 4'b0000;
    parameter S1 = 4'b0001;
    parameter S2 = 4'b0010;
    parameter S3 = 4'b0011;
    parameter S4 = 4'b0100;
    parameter S5 = 4'b0101;
    parameter S6 = 4'b0110;
    parameter S7 = 4'b0111;
    parameter S8 = 4'b1000;
    parameter S9 = 4'b1001;
    parameter S10 = 4'b1010;
    parameter S11 = 4'b1011;
    parameter S12 = 4'b1100;
    parameter S13 = 4'b1101;
    parameter S14 = 4'b1110;
    parameter S15 = 4'b1111;

    input clk;
    input reset_;
    input [WIDTH-1:0] in_A;
    input [WIDTH-1:0] in_B;
    input [WIDTH-1:0] in_C;
    input [WIDTH-1:0] in_D;
    input [WIDTH-1:0] in_E;
    input [WIDTH-1:0] in_F;
    input [WIDTH-1:0] in_G;
    input [WIDTH-1:0] in_H;
    input [WIDTH-1:0] in_I;
    input [WIDTH-1:0] in_J;
    input [WIDTH-1:0] in_K;
    input [WIDTH-1:0] in_L;
    input [WIDTH-1:0] in_M;
    input [WIDTH-1:0] in_N;
    input [WIDTH-1:0] in_O;
    input [WIDTH-1:0] in_P;
    output reg [FSM_WIDTH-1:0] fsm_out;
    reg [FSM_WIDTH-1:0] state, next_state;
    always_ff @(posedge clk or negedge reset_) begin
        if (!reset_) begin
            state <= S0;
        end else begin
            state <= next_state;
        end
    end
    always_comb begin
        case(state)
            S0: begin
                if ((in_B == 'd1)) begin
                    next_state = S3;
                end
                else if (((in_M != in_E) != in_M)) begin
                    next_state = S7;
                end
                else begin
                    next_state = S15;
                end
            end
            S1: begin
                if (((in_E ^ in_F) ^ (in_M || in_C))) begin
                    next_state = S8;
                end
                else if ((in_D && in_H)) begin
                    next_state = S0;
                end
                else if ((~&(in_E) && in_M)) begin
                    next_state = S15;
                end
                else if ((in_O ^ in_A)) begin
                    next_state = S12;
                end
                else begin
                    next_state = S3;
                end
            end
            S2: begin
                if ((in_O != 'd0)) begin
                    next_state = S12;
                end
                else begin
                    next_state = S1;
                end
            end
            S3: begin
                if (~^((in_N || in_F))) begin
                    next_state = S11;
                end
                else begin
                    next_state = S4;
                end
            end
            S4: begin
                if (~^(in_B)) begin
                    next_state = S0;
                end
                else if (((in_J ^ in_M) != 'd0)) begin
                    next_state = S5;
                end
                else begin
                    next_state = S10;
                end
            end
            S5: begin
                if ((in_P && in_I)) begin
                    next_state = S14;
                end
                else if (((in_K == 'd1) && in_L)) begin
                    next_state = S9;
                end
                else if (((in_K <= 'd0) != 'd0)) begin
                    next_state = S7;
                end
                else begin
                    next_state = S0;
                end
            end
            S6: begin
            end
            S7: begin
                if ((in_O != in_A)) begin
                    next_state = S9;
                end
                else if ((in_L || ~&(in_E))) begin
                    next_state = S13;
                end
                else begin
                    next_state = S5;
                end
            end
            S8: begin
                if (((in_D ^ in_F) != 'd1)) begin
                    next_state = S14;
                end
                else begin
                    next_state = S12;
                end
            end
            S9: begin
                if (!(in_A)) begin
                    next_state = S1;
                end
                else begin
                    next_state = S4;
                end
            end
            S10: begin
                next_state = S5;
            end
            S11: begin
                if ((~|(in_G) == 'd1)) begin
                    next_state = S9;
                end
                else if ((in_M == 'd0)) begin
                    next_state = S3;
                end
                else begin
                    next_state = S5;
                end
            end
            S12: begin
                if (~((in_B != in_O))) begin
                    next_state = S15;
                end
                else if ((in_G <= in_H)) begin
                    next_state = S6;
                end
                else begin
                    next_state = S9;
                end
            end
            S13: begin
                next_state = S12;
            end
            S14: begin
                if (~&((in_C == in_K))) begin
                    next_state = S2;
                end
                else if (((in_I && in_H) == 'd0)) begin
                    next_state = S6;
                end
                else if (((in_A != in_M) ^ in_F)) begin
                    next_state = S12;
                end
                else begin
                    next_state = S15;
                end
            end
            S15: begin
                if (((in_F && in_M) == 'd0)) begin
                    next_state = S4;
                end
                else if ((in_P && (in_N == 'd1))) begin
                    next_state = S7;
                end
                else if ((in_N != (in_O && in_B))) begin
                    next_state = S10;
                end
                else if ((in_H > 'd0)) begin
                    next_state = S3;
                end
                else if (&((in_A == 'd1))) begin
                    next_state = S14;
                end
                else begin
                    next_state = S5;
                end
            end
        endcase
    end
endmodule
''' # Rest of your Verilog code here

# Count tokens for different models
models = ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]
for model in models:
    num_tokens = count_tokens(verilog_code, model)
    print(f"Token count for {model}: {num_tokens}")

# Optional: Show actual tokens for verification
def show_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    token_bytes = [encoding.decode_single_token_bytes(token) for token in tokens]
    print("\nFirst 20 tokens:")
    for i, (token, byte_content) in enumerate(zip(tokens[:20], token_bytes[:20])):
        print(f"{i+1}. Token {token}: {byte_content}")

show_tokens(verilog_code)