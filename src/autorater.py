from config import FLAGS
from saver import saver
from utils_agent import initiate_chat_with_retry
import re
from autogen.agentchat.chat import ChatResult


print = saver.log_info


def extract_sva(response):
    if not isinstance(response, str):
        print(f"Warning: response is not a string. Type: {type(response)}")
        if response is None:
            return None
        try:
            response = str(response)
        except:
            print("Error: Unable to convert response to string.")
            return None

    sva_match = re.search(r'assert property\s*\((.*?)\)\s*;', response, re.DOTALL)
    if sva_match:
        return sva_match.group()
    return None


def auto_correct(agents, response, row):
    autorater = agents['autorater']
    coding_agent = agents['Coding']
    user_agent = agents['user']

    max_iterations = FLAGS.max_autorater_iter
    best_response = response
    best_rating = 0

    ICL_EXAMPLES = [
        {
            "prompt": "Create a SVA assertion that checks: that the request signal is one-hot or zero. Use the signal 'req'.",
            "sva": "asrt: assert property (@(posedge clk) disable iff (reset) !($onehot0(req)) !== 1'b1);"
        },
        {
            "prompt": "Create a SVA assertion that checks: that the grant signal is strictly one-hot. Use the signal 'gnt'.",
            "sva": "asrt: assert property (@(posedge clk) disable iff (reset) !($onehot(gnt)) !== 1'b1);"
        }, # Add tricks here, write some similar cases
    ]

    icl_examples = "\n\n".join([f"Question: {ex['prompt']}\nAnswer: {ex['sva']}" for ex in ICL_EXAMPLES])

    # Extract the actual response string
    if hasattr(response, 'chat_history') and isinstance(response.chat_history, list):
        response_str = response.chat_history[-1]['content'] if response.chat_history else ""
    else:
        response_str = str(response)

    for iteration in range(max_iterations):
        sva = extract_sva(response_str)
        if not sva:
            print(f"No SVA found in the response. Response: {response_str}")
            print("Ending auto-correction.")
            return best_response

        print(f"\nIteration {iteration + 1}")
        print(f"Current SVA:\n{sva}")

        if FLAGS.baseline_reflexion:
            autorater_input = f"""
            Natural Language Description: {row.prompt}

            Generated SVA:
            {sva}

            Here are some example SVAs for similar problems:

            {icl_examples}

            Please evaluate this SVA based on the natural language description and the following criteria:

            1. Correctness (0-4 points):
            - Does the assertion correctly check the behavior described in the prompt?

            2. Completeness (0-3 points):
            - Does the assertion cover all aspects mentioned in the description?
            - Are all necessary signals included?

            3. Syntax (0-2 points):
            - Is the SVA syntactically correct?
            - Are all parentheses and operators properly placed?

            4. Efficiency (0-1 point):
            - Is the assertion written in a clear and concise manner?

            Please provide:
            1. A numeric rating from 0 to 10 (sum of the above criteria)
            2. Brief feedback for improvement, addressing any shortcomings in the above criteria
            3. A suggested improved version of the SVA, if the current version is not perfect

            Ensure your feedback is specific and actionable, focusing on how to make the SVA match the required behavior described in the natural language description and follow the general format of the example SVAs.
            """
        else:
            autorater_input = f"""
            Natural Language Description: {row.prompt}

            Generated SVA:
            {sva}

            Here are some example SVAs for similar problems:

            {icl_examples}

            Please evaluate this SVA based on the natural language description and the following criteria:

            1. Correctness (0-4 points):
            - Does the assertion correctly check the behavior described in the prompt?
            - Is the correct function used ($onehot, $onehot0, etc.) based on the description?
            - Is the logic of the assertion sound, including the use of ! and !== 1'b1?

            2. Completeness (0-3 points):
            - Does the assertion cover all aspects mentioned in the description?
            - Are all necessary signals included?

            3. Syntax (0-2 points):
            - Is the SVA syntactically correct?
            - Does it follow the general format shown in the examples, including the "asrt:" label and !== 1'b1 at the end?
            - Are all parentheses and operators properly placed?

            4. Efficiency (0-1 point):
            - Is the assertion written in a clear and concise manner?

            Please provide:
            1. A numeric rating from 0 to 10 (sum of the above criteria)
            2. Brief feedback for improvement, addressing any shortcomings in the above criteria
            3. A suggested improved version of the SVA, if the current version is not perfect

            Ensure your feedback is specific and actionable, focusing on how to make the SVA match the required behavior described in the natural language description and follow the general format of the example SVAs.
            Pay special attention to the choice between $onehot and $onehot0 based on the problem description.
            """
        autorater_feedback = initiate_chat_with_retry(user_agent, autorater, message=autorater_input)

        # Parse the autorater's response
        rating_match = re.search(r'Rating:\s*(\d+(?:\.\d+)?)', autorater_feedback)
        feedback_match = re.search(r'Feedback:(.*?)(?:\n\n|$)', autorater_feedback, re.DOTALL)
        improved_sva_match = re.search(r'Improved SVA:(.*?)(?:\n\n|$)', autorater_feedback, re.DOTALL)

        rating = float(rating_match.group(1)) if rating_match else 0
        feedback = feedback_match.group(1).strip() if feedback_match else ""
        improved_sva = improved_sva_match.group(1).strip() if improved_sva_match else ""

        print(f"Iteration {iteration + 1} - Rating: {rating}")
        print(f"Feedback: {feedback}")
        if improved_sva:
            print(f"Suggested Improvement: {improved_sva}")

        if rating > best_rating:
            best_rating = rating
            best_response = response_str
            print(f"New best response (Rating: {best_rating}):\n{best_response}")

        if rating >= 9.5:  # Increased threshold for excellence
            print("SVA is considered excellent. Ending auto-correction.")
            break

        coding_input = f"""
        Please improve the following SVA based on this feedback:
        
        Current SVA:
        {sva}
        
        Feedback: {feedback}
        
        Natural Language Description: {row.prompt}
        
        Here are some example SVAs for similar problems:

        {icl_examples}

        Generate an improved SVA addressing the feedback and following the general format of the example SVAs.
        Ensure you include the "asrt:" label at the beginning and end with !== 1'b1.
        Pay careful attention to the choice between $onehot and $onehot0 based on the problem description.
        Only output the SVA code and nothing else.
        """

        response = initiate_chat_with_retry(user_agent, coding_agent, message=coding_input)

        # Extract the actual response string
        if hasattr(response, 'chat_history') and isinstance(response.chat_history, list):
            response_str = response.chat_history[-1]['content'] if response.chat_history else ""
        else:
            response_str = str(response)

    print(f"Finished auto-correction. Best rating: {best_rating}")

    # Post-processing step
    # final_response = post_process_sva(best_response)
    final_response = best_response

    # Create a new object with the same structure as the original response
    if hasattr(response, 'chat_history') and isinstance(response.chat_history, list):
        new_response = type(response)(chat_id=None, chat_history=[{'content': final_response, 'role': 'assistant'}])
    else:
        new_response = final_response

    return new_response

def post_process_sva(sva):
    # Remove any extra characters or formatting
    sva = re.sub(r'```systemverilog|```', '', sva).strip()
    
    # Ensure the SVA starts with "asrt:"
    if not sva.startswith("asrt:"):
        sva = "asrt: " + sva

    # Ensure the correct format for the assertion
    sva = re.sub(r'assert\s+property\s*\(\s*@\s*\(\s*posedge\s+clk\s*\)\s*disable\s+iff\s*\(\s*\w+\s*\)\s*', 
                 'assert property (@(posedge clk) disable iff (tb_reset) ', sva)

    # Ensure the assertion ends with !== 1'b1
    if '!== 1\'b1' not in sva:
        sva = re.sub(r'\)\s*;?\s*$', ') !== 1\'b1);', sva)

    # Ensure the assertion ends correctly
    if not sva.endswith(');'):
        sva = sva.rstrip(';') + ');'

    return sva