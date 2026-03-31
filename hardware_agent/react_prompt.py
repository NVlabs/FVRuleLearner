# Templates from langchain https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/structured_chat/prompt.py
# flake8: noqa
PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
# FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
# Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
#
#The only values that should be in the "action" field are: {tool_names}
#
#The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
#
#```
#{{{{
#  "action": $TOOL_NAME,
#  "action_input": $INPUT
#}}}}
#```
#
#ALWAYS use the following format:
#
#Question: the input question you must answer
#Thought: you should always think about what to do
#Action:
#```
#$JSON_BLOB
#```
#Observation: the result of the action
#... (this Thought/Action/Observation can repeat N times)
#Thought: I now know the final answer
#Final Answer: the final answer to the original input question
#"""

FORMAT_INSTRUCTIONS = """
Answer the following questions as best you can. You have access to tools provided in {tool_names}.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

SUFFIX = """Begin! Reply TERMINATE when you provide the Final Answer..
Question: {input}
"""