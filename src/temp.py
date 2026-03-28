# # temp.py
# from fv_tools import evaluate_jg
# from saver import saver

# print = saver.log_info


# # Example test cases
# def main():
#     # Call evaluate_jg with example parameters and print the output
#     str1 = """
# Use JasperGold to check the syntax.                                                                                                              
# Context:                                                                                                                                         
# ```systemverilog                                                                                                                                 
# assert property(@(posedge clk)                                                                                                                   
#     sig_D |=> s_eventually(sig_F)                                                                                                                
# );                                                                                                                                               
# ```                                                                                                                                              
# The provided SVA code snippet uses a sequential assertion.
# """
#     result = evaluate_jg(str1)  # Replace with actual parameters
#     print('\033[94m' + 'result: ' + '\033[0m' + result)


# if __name__ == "__main__":
#     main()


from adlrchat.langchain import ADLRChat

from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat_C = ADLRChat(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, model="fvmixtral_8x7b_0802_steer_delta", stop=["extra_id_1"],
    prepare_env_args={"SERVER_COOLNAME": "aromatic-partridge"}
)

chat_C.invoke([HumanMessage(content="What is air?")])