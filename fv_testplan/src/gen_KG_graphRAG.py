import os
import subprocess
import shutil
import json
import re
from pathlib import Path
import PyPDF2
import logging
from utils import OurTimer
from saver import saver
from config import FLAGS
print = saver.log_info


# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)



def build_KG():
    timer = OurTimer()
    try:
        input_file_path = FLAGS.input_file_path
        
        base_dir = get_base_dir(input_file_path)
        print(f"Derived base directory: {base_dir}")

        # Step 1: Process input file
        print("Step 1: Processing input file")
        graph_rag_dir = create_directory_structure(base_dir)
        
        # Clean up the input folder
        clean_input_folder(os.path.join(graph_rag_dir, 'input'))
        
        if input_file_path.lower().endswith('.jsonl'):
            num_pages, file_size = get_jsonl_stats(input_file_path)
            print(f"JSONL Statistics: {num_pages} pages, {file_size:.2f} MB")
            text_file_path = parse_jsonl_to_text(input_file_path, os.path.join(graph_rag_dir, 'input'))
        else:
            num_pages, file_size = get_pdf_stats(input_file_path)
            print(f"PDF Statistics: {num_pages} pages, {file_size:.2f} MB")
            text_file_path = parse_pdf_to_text(input_file_path, os.path.join(graph_rag_dir, 'input'))

        # Step 2: Initialize GraphRAG
        print("Step 2: Initializing GraphRAG")
        initialize_graphrag(graph_rag_dir)

        # Step 3: Update .env file
        print("Step 3: Updating .env file")
        shutil.copy(FLAGS.env_source_path, os.path.join(graph_rag_dir, '.env'))
        print(f"Copied .env from {FLAGS.env_source_path} to {graph_rag_dir}")

        # Step 4: Update settings.yaml file
        print("Step 4: Updating settings.yaml file")
        shutil.copy(FLAGS.settings_source_path, os.path.join(graph_rag_dir, 'settings.yaml'))
        print(f"Copied settings.yaml from {FLAGS.settings_source_path} to {graph_rag_dir}")

        # Step 5: Run GraphRAG indexing
        print("Step 5: Running GraphRAG indexing")
        return_code = run_graphrag_index(graph_rag_dir)
        if return_code != 0:
            timer.time_and_clear("error")
            timer.print_durations_log(print_func=print)
            raise RuntimeError("GraphRAG indexing failed")

        print("Process completed successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    timer.time_and_clear("done")
    timer.print_durations_log(print_func=print)

def get_base_dir(file_path):
    return os.path.dirname(file_path)

def get_pdf_stats(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
    return num_pages, file_size

def get_jsonl_stats(jsonl_path):
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()
        num_pages = sum(1 for line in lines if json.loads(line).get('page') is not None)
        file_size = os.path.getsize(jsonl_path) / (1024 * 1024)  # Convert to MB
    return num_pages, file_size

def parse_pdf_to_text(pdf_path, output_dir):
    design_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{design_name}.txt")

    with open(pdf_path, 'rb') as pdf_file, open(output_path, 'w', encoding='utf-8') as txt_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            txt_file.write(page.extract_text())

    return output_path


def clean_input_folder(input_dir):
    """
    Remove all files from the input directory.
    
    Args:
    input_dir (str): Path to the input directory
    """
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print(f"Cleaned up input folder: {input_dir}")


def clean_table(table_content):
    table = re.sub(r'\\text\{([^}]*)\}', r'\1', table_content)
    table = table.replace('\\\\', '\n')
    table = table.replace('&', ' | ')
    table = re.sub(r'\\[a-zA-Z]+', '', table)
    table = re.sub(r'\^\{?2\}?', '²', table)
    table = re.sub(r'\s+', ' ', table).strip()
    return table

def parse_jsonl_to_text(jsonl_path, output_dir):
    design_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    output_path = os.path.join(output_dir, f"{design_name}_processed.txt")

    with open(jsonl_path, 'r') as jsonl_file, open(output_path, 'w', encoding='utf-8') as txt_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            
            # Process normal "out" content
            if 'out' in json_obj and json_obj['out'].strip():
                txt_file.write(json_obj['out'] + "\n\n")
            
            # Process Table 1 if present in raw_out
            if 'raw_out' in json_obj:
                raw_out = json_obj['raw_out']
                if 'Table 1: Pinout description' in raw_out: # pretty hacky...
                    txt_file.write("Table 1: Pinout description\n\n")
                    table_content = re.search(r'\\begin\{array\}(.*?)\\end\{array\}', raw_out, re.DOTALL)
                    if table_content:
                        cleaned_table = clean_table(table_content.group(1))
                        txt_file.write(cleaned_table + "\n\n")
                
                # Process other raw_out content if "out" is empty
                elif not json_obj.get('out'):
                    cleaned_raw_out = re.sub(r'<[^>]+>', '', raw_out)  # Remove HTML-like tags
                    cleaned_raw_out = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', cleaned_raw_out)  # Remove LaTeX commands
                    cleaned_raw_out = cleaned_raw_out.replace('\\n', '\n').strip()
                    if cleaned_raw_out:
                        txt_file.write(cleaned_raw_out + "\n\n")

    return output_path


def create_directory_structure(base_dir):
    graph_rag_dir = os.path.join(base_dir, 'graph_rag')
    input_dir = os.path.join(graph_rag_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    return graph_rag_dir


def initialize_graphrag(graph_rag_dir):
    command = f"export PYTHONPATH='{FLAGS.graphrag_local_dir}' && python -m graphrag.index --init --root {graph_rag_dir}"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        if "Project already initialized" in result.stderr:
            print("GraphRAG project already initialized. Skipping initialization.")
        else:
            print("Error during initialization:")
            print(result.stderr)
            raise RuntimeError("GraphRAG initialization failed")
    else:
        print("GraphRAG Initialization Output:")
        print(result.stdout)

def run_graphrag_index(graph_rag_dir):
    command = f"export PYTHONPATH='{FLAGS.graphrag_local_dir}:$PYTHONPATH' && python -m graphrag.index --root {graph_rag_dir}"
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

    print(f"GraphRAG Indexing Output:")
    for line in process.stdout:
        print(f"{line.rstrip()}")  # Remove trailing newline and print

    return_code = process.wait()

    if return_code != 0:
        print(f"GraphRAG indexing failed with return code {return_code}")
    else:
        print(f"GraphRAG indexing completed successfully")

    return return_code

if __name__ == "__main__":
    build_KG()


prompt = '''
Suppose you have a PDF file representing a design spec, '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/apbi2c_spec.pdf'. You want to parse it into plain text using some tool/library, and then call GraphRAG to convert it into graphs stored on disk. Be careful with file/folder structure and locations: Print intermediate steps and results across the code. 

Step (1): Parse the PDF. Before that, print some stats about this PDF like how many page, how big is the file size (in Mb etc. with proper file size formatting). Store the resulting txt version of this PDF under the same directory as the enclosing folder of this PDF, e.g. '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/` in thie example. Specifically, create a new folder called `graph_rag/` and create a subfolder `graph_rag/input` under it with the file `graph_rag/input/<design_name>.txt` where design_name is 'apbi2c_spec' in this example. Print some stats and maybe some nice visualization of the folder structure at this point.

Step (2): Initialize Microsoft GraphRAG. Suppose I have installed the package. So somehow spawn a new subprocess/thread or so (or maybe just do it in the current thread) running `python -m graphrag.index --init --root ./home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/graph_rag`. Record and print any error message this command may have raised. If this command succeeds, we should have '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/graph_rag/.env` file and '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/graph_rag/settings.yaml`.

Step (3): Replace the content of .env with the following:

"""
GRAPHRAG_API_KEY=123
GRAPHRAG_API_BASE=http://172.17.0.1:11434/v1
# GRAPHRAG_LLM_MODEL=llama3:instruct
GRAPHRAG_LLM_MODEL=llama2
GRAPHRAG_LLM_THREAD_COUNT=4
GRAPHRAG_LLM_CONCURRENT_REQUESTS=8
GRAPHRAG_LLM_MAX_TOKENS=2048

GRAPHRAG_EMBEDDING_API_BASE=http://172.17.0.1:11434/api
GRAPHRAG_EMBEDDING_MODEL=llama2
"""


Step (4): Replace the content of settings.yaml with the following:

"""

encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat # or azure_openai_chat
  model: mistral
  model_supports_json: true # recommended if this is available for your model.
  # max_tokens: 4000
  # request_timeout: 180.0
  api_base: http://localhost:11434/v1
  # api_version: 2024-02-15-preview
  # organization: <organization_id>
  # deployment_name: <azure_model_deployment_name>
  # tokens_per_minute: 150_000 # set a leaky bucket throttle
  # requests_per_minute: 10_000 # set a leaky bucket throttle
  # max_retries: 10
  # max_retry_wait: 10.0
  # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
  # concurrent_requests: 25 # the number of parallel inflight requests that may be made

parallelization:
  stagger: 0.3
  # num_threads: 50 # the number of threads to use for parallel processing

async_mode: threaded # or asyncio

embeddings:
  ## parallelization: override the global parallelization settings for embeddings
  async_mode: threaded # or asyncio
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding # or azure_openai_embedding
    model: nomic-embed-text
    api_base: http://localhost:11434/api
    # api_version: 2024-02-15-preview
    # organization: <organization_id>
    # deployment_name: <azure_model_deployment_name>
    # tokens_per_minute: 150_000 # set a leaky bucket throttle
    # requests_per_minute: 10_000 # set a leaky bucket throttle
    # max_retries: 10
    # max_retry_wait: 10.0
    # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
    # concurrent_requests: 25 # the number of parallel inflight requests that may be made
    # batch_size: 16 # the number of documents to send in a single request
    # batch_max_tokens: 8191 # the maximum number of tokens to send in a single request
    # target: required # or optional
  


chunks:
  size: 300
  overlap: 100
  group_by_columns: [id] # by default, we don't allow chunks to cross documents
    
input:
  type: file # or blob
  file_type: text # or csv
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file # or blob
  base_dir: "cache"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

storage:
  type: file # or blob
  base_dir: "output/${timestamp}/artifacts"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

reporting:
  type: file # or console, blob
  base_dir: "output/${timestamp}/reports"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

entity_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 0

summarize_descriptions:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  # enabled: true
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 0

community_report:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false # if true, will generate node2vec embeddings for nodes
  # num_walks: 10
  # walk_length: 40
  # window_size: 2
  # iterations: 3
  # random_seed: 597832

umap:
  enabled: false # if true, will generate UMAP embeddings for nodes

snapshots:
  graphml: yes
  raw_entities: yes
  top_level_nodes: yes

local_search:
  # text_unit_prop: 0.5
  # community_prop: 0.1
  # conversation_history_max_turns: 5
  # top_k_mapped_entities: 10
  # top_k_relationships: 10
  # max_tokens: 12000

global_search:
  # max_tokens: 12000
  # data_max_tokens: 12000
  # map_max_tokens: 1000
  # reduce_max_tokens: 2000
  # concurrency: 32
"""

The reason is that we want to do the local setup of GraphRAG, i.e. running ollama with mistral instead of calling OpenAI's API. to save cost and be quick

Step (5): Execute `python -m graphrag.index --root .//home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/graph_rag/` which will take some time. Be sure to let me see the output of this command, i.e. do NOT suppress the output of this command. Show everything including error messages generated by running this command. This step may break at the every end, e.g. 
"""
___ Loading Input (text) - 1 files loaded (0 filtered) ______ 100% 0:00:_ 0:00:_
___ create_base_text_units
___ create_base_extracted_entities
___ create_summarized_entities
___ create_base_entity_graph
___ create_final_entities
__ Errors occurred during the pipeline run, see logs for more details.
"""
However, this is fine since we already reach the `create_final_entities` step. Just step there and I will examine the content stored in the folder myself for further analysis.


Write proper helper functions with documentation/annotations for readability.
'''


'''
Some note:
Working on Graph RAG but got an error similar to https://github.com/TheAiSingularity/graphrag-local-ollama/issues/14
	According to https://github.com/microsoft/graphrag/issues/370, might need some brute-force source code modification: “/home/scratch.yunshengb_avr_misc/miniconda3/envs/fv_testplan/lib/python3.10/site-packages/graphrag/llm/openai/openai_embeddings_llm.py” on pdx (scatch space):
	“””
# Yunsheng on 7/23/2024 according to https://github.com/microsoft/graphrag/issues/370

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes

import ollama


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        # embedding = await self.client.embeddings.create(
        #     input=input,
        #     **args,
        # )
        # inputs = input['input']
        # print(inputs)
        embedding_list = []
        for inp in input:
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=inp)
            embedding_list.append(embedding["embedding"])
        # return [d.embedding for d in embedding.data]
        return embedding_list


# # Copyright (c) 2024 Microsoft Corporation.
# # Licensed under the MIT License

# """The EmbeddingsLLM class."""

# from typing_extensions import Unpack

# from graphrag.llm.base import BaseLLM
# from graphrag.llm.types import (
#     EmbeddingInput,
#     EmbeddingOutput,
#     LLMInput,
# )

# from .openai_configuration import OpenAIConfiguration
# from .types import OpenAIClientTypes


# class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
#     """A text-embedding generator LLM."""

#     _client: OpenAIClientTypes
#     _configuration: OpenAIConfiguration

#     def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
#         self.client = client
#         self.configuration = configuration

#     async def _execute_llm(
#         self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
#     ) -> EmbeddingOutput | None:
#         args = {
#             "model": self.configuration.model,
#             **(kwargs.get("model_parameters") or {}),
#         }
#         embedding = await self.client.embeddings.create(
#             input=input,
#             **args,
#         )
#         return [d.embedding for d in embedding.data]
	“””
'''
