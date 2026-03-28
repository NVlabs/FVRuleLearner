from config import FLAGS, load_perflab_config
import time
import sqlite3
# from transformers import AutoTokenizer
from saver import saver
import re
import json
import httpx
import os
import copy
import shutil
from pathlib import Path
from FVEval.fv_eval import utils as utils2
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from adlrchat.langchain import ADLRChat
from openai import AzureOpenAI
from openai import APITimeoutError, APIError, APIConnectionError, InternalServerError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from hardware_agent.basic_client import BasicClient
from autogen import config_list_from_json

# Color codes for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

print = saver.log_info

from autogen.agentchat.chat import ChatResult
import tiktoken

class RateLimitError(Exception):
    pass

def check_and_fix_cache(cache_dir=".cache"):
    """
    Check for corrupted cache databases and fix them by removing corrupted files.
    Returns True if any corrupted cache was found and fixed.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False
    
    corrupted_found = False
    for cache_seed_dir in cache_path.iterdir():
        if cache_seed_dir.is_dir():
            cache_db = cache_seed_dir / "cache.db"
            if cache_db.exists():
                try:
                    # Try to open and validate the database
                    conn = sqlite3.connect(str(cache_db))
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA integrity_check;")
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result[0] != "ok":
                        print(f"{YELLOW}⚠️  Corrupted cache detected: {cache_db}{RESET}")
                        corrupted_found = True
                        # Backup and remove corrupted cache
                        backup_name = f"{cache_seed_dir.name}.corrupted.{int(time.time())}"
                        backup_path = cache_path / backup_name
                        shutil.move(str(cache_seed_dir), str(backup_path))
                        print(f"{GREEN}✓ Moved corrupted cache to: {backup_path}{RESET}")
                        
                except sqlite3.DatabaseError as e:
                    print(f"{YELLOW}⚠️  Database error in {cache_db}: {e}{RESET}")
                    corrupted_found = True
                    # Backup and remove corrupted cache
                    backup_name = f"{cache_seed_dir.name}.corrupted.{int(time.time())}"
                    backup_path = cache_path / backup_name
                    shutil.move(str(cache_seed_dir), str(backup_path))
                    print(f"{GREEN}✓ Moved corrupted cache to: {backup_path}{RESET}")
                    
                except Exception as e:
                    print(f"{YELLOW}⚠️  Unexpected error checking {cache_db}: {e}{RESET}")
    
    return corrupted_found

def is_o_series_model(model_name):
    """Check if the model is an o-series model (o1, o3, etc.)"""
    if not model_name:
        return False
    # Check for o1, o3 patterns at the start of the model name
    return bool(re.match(r'^o[134]-', model_name.lower()))

def get_tokenizer(llm_model):
    # Attempt to use the specific model for tokenization
    try:
        return tiktoken.encoding_for_model(llm_model)
    except Exception as e:
        print(f"Error loading tokenizer for {llm_model}: {e}. Falling back to 'gpt-3.5-turbo'.")
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as fallback_e:
            print(f"Error loading fallback tokenizer: {fallback_e}")
            raise fallback_e

def count_tokens(tokenizer, message):
    # Tokenize the message using tiktoken
    try:
        return len(tokenizer.encode(message))
    except Exception as e:
        print(f"Error tokenizing message: {e}")
        raise e


def convert_time_to_float(time_str):
    # Use regular expression to find the numeric part of the string
    match = re.search(r"(\d+\.\d+)", time_str)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Cannot convert {time_str} to float.")

# def initiate_chat_with_retry(agent_user, agent_coding, message, retries=FLAGS.GPT_retries, backoff_factor=FLAGS.backoff_factor, **kwargs):
#     agent_user.revalidate_llm_config()
#     agent_coding.revalidate_llm_config()
#     if not message:
#         message = kwargs.get('message')
    
#     if not message:
#         raise ValueError("Message must be provided either as a parameter or in kwargs.")
    
#     # Initialize statistics dictionary if not already done
#     if not hasattr(agent_user, 'stats'):
#         agent_user.stats = {'runtime': [], 'tokens': []}
    
#     # Get the tokenizer for the model
#     tokenizer = get_tokenizer(FLAGS.llm_model)
    
#     for attempt in range(retries):
#         start_time = time.time()
#         try:
#             # message = 'test the issue belongs to the message'
#             response = agent_user.initiate_chat(agent_coding, message=message)
#             LLM_runtime = time.time() - start_time

#             # print(f"@@@DEBUG: runtime = {type(runtime)} {runtime}")
#             tokens = count_tokens(tokenizer, message)
            
#             # Save stats
#             saver.save_stats('LLM_runtime', LLM_runtime)
#             saver.save_stats('tokens', tokens)
            
#             # Process response and return if successful
#             response_str = response.chat_history[-1]['content']
#             return response_str
        
#         except RateLimitError as e:
#             wait_time = backoff_factor ** attempt
#             print(f"@@@{FLAGS.llm_model}: Rate limit exceeded. Retrying in {wait_time} seconds...")
#             time.sleep(wait_time)
#         except Exception as e:
#             wait_time = backoff_factor ** attempt
#             print(f"@@@{FLAGS.llm_model}: An error occurred: {e}. Retrying in {wait_time} seconds...")
#             time.sleep(wait_time)

def initiate_chat_with_retry_legacy(agent_user, agent_coding, message, retries, backoff_factor = FLAGS.backoff_factor, **kwargs):
    """
    The original logic is preserved here. 
    This function is identical to your original initiate_chat_with_retry code.
    """
    agent_user.revalidate_llm_config()
    agent_coding.revalidate_llm_config()
    if not message:
        message = kwargs.get('message')
    
    if not message:
        raise ValueError("Message must be provided either as a parameter or in kwargs.")
    
    # Initialize statistics dictionary if not already done
    if not hasattr(agent_user, 'stats'):
        agent_user.stats = {'runtime': [], 'tokens': []}
    
    # Get the tokenizer for the model
    tokenizer = get_tokenizer(FLAGS.llm_model)
    
    cache_fix_attempted = False
    
    for attempt in range(retries):
        start_time = time.time()
        try:
            response = agent_user.initiate_chat(agent_coding, message=message)
            LLM_runtime = time.time() - start_time
            tokens = count_tokens(tokenizer, message)
            
            # Save stats
            saver.save_stats('LLM_runtime', LLM_runtime)
            saver.save_stats('tokens', tokens)
            
            # Process response and return if successful
            response_str = response.chat_history[-1]['content']
            return response_str
        
        # DEBUG
        except RateLimitError as e:
            wait_time = backoff_factor ** attempt
            print(f"@@@{FLAGS.llm_model}: Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except sqlite3.DatabaseError as e:
            # Handle database corruption errors specifically
            if "database" in str(e).lower() and not cache_fix_attempted:
                print(f"{YELLOW}⚠️  Database error detected. Attempting to fix cache...{RESET}")
                check_and_fix_cache()
                cache_fix_attempted = True
                print(f"{GREEN}✓ Cache check complete. Retrying immediately...{RESET}")
                continue
            else:
                wait_time = backoff_factor ** attempt
                print(f"@@@{FLAGS.llm_model}: Database error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        except Exception as e:
            # Check if this is a database-related error in the exception message
            if "database" in str(e).lower() and not cache_fix_attempted:
                print(f"{YELLOW}⚠️  Database-related error detected: {e}. Attempting to fix cache...{RESET}")
                check_and_fix_cache()
                cache_fix_attempted = True
                print(f"{GREEN}✓ Cache check complete. Retrying immediately...{RESET}")
                continue
            else:
                wait_time = backoff_factor ** attempt
                print(f"@@@{FLAGS.llm_model}: An error occurred: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # If we reach this point, no successful response was obtained
    raise RuntimeError("Failed to get a response after all retries.")


def initiate_chat_with_retry(agent_user, agent_coding, message=None, retries=FLAGS.GPT_retries, backoff_factor=2,
                             model="fvmixtral_8x7b_0802_steer_delta",
                             prepare_env_args={"SERVER_COOLNAME": "aromatic-partridge"},
                             temperature=0.0,
                             stop=["extra_id_1"],
                             callbacks=None,
                             streaming=True,
                             **kwargs):
    """
    A wrapper function that can either use the original legacy approach (via initiate_chat_with_retry_legacy)
    or the new ADLRChat approach, based on the `use_new_method` flag.
    """
    
    if not message:
        message = kwargs.get('message')
    if not message:
        raise ValueError("Message must be provided either as a parameter or in kwargs.")

    # Determine which method to use
    use_new_method = hasattr(FLAGS, 'baseline_finetune') and FLAGS.baseline_finetune
    use_mark_perflab = hasattr(FLAGS, 'LLM_gateaway') and FLAGS.LLM_gateaway in ['Mark', 'perflab']
    # breakpoint()
    
    if use_mark_perflab:
        
        # Try multiple locations where system message might be stored
        system_prompt = agent_coding.system_message
        
        # Determine which model to use
        # Priority: agent_model parameter > agent's model attribute > FLAGS.llm_model
        effective_model = model
        if effective_model is None and hasattr(agent_coding, 'llm_config') and isinstance(agent_coding.llm_config, dict):
            # Try to extract model from agent's llm_config
            config_list = agent_coding.llm_config.get('config_list', [])
            if config_list and isinstance(config_list[0], dict):
                effective_model = config_list[0].get('model')
        
        # Debug logging if enabled
        # if hasattr(FLAGS, 'debug') and FLAGS.debug and system_prompt:
        #     print(f"Extracted system prompt: {system_prompt[:100]}...")
        
        # Initialize statistics dictionary if not already done
        if not hasattr(agent_user, 'stats'):
            agent_user.stats = {'runtime': [], 'tokens': []}
        
        # Implement retry logic for Mark/perflab gateways
        for attempt in range(retries):
            try:
                # Use llm_inference for Mark/perflab gateways
                response_str = llm_inference(system_prompt, message, temperature=temperature, model=effective_model)
                return response_str
                
            except RateLimitError as e:
                wait_time = backoff_factor ** attempt
                print(f"@@@{FLAGS.llm_model}: Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except (APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError) as e:
                # These exceptions are already handled by the @retry decorator in llm_inference
                # but we can add additional handling here if needed
                raise
            except Exception as e:
                wait_time = backoff_factor ** attempt
                print(f"@@@{FLAGS.llm_model}: An error occurred: {e}. Retrying in {wait_time} seconds...")
                if attempt < retries - 1:
                    time.sleep(wait_time)
                else:
                    raise
        
        # If we reach here, no response after retries
        raise RuntimeError("Failed to get a response after all retries using Mark/perflab gateway.")
    
    elif use_new_method:
        # Using the new ADLRChat invocation
        # if callbacks is None:
        #     callbacks = [StreamingStdOutCallbackHandler()]

        # # Initialize statistics dictionary if not already done
        # if not hasattr(agent_user, 'stats'):
        #     agent_user.stats = {'runtime': [], 'tokens': []}
        
        # tokenizer = get_tokenizer(FLAGS.llm_model)

        # for attempt in range(retries):
        #     start_time = time.time()
        #     try:
        #         chat_instance = ADLRChat(
        #             streaming=streaming,
        #             callbacks=callbacks,
        #             temperature=temperature,
        #             model=model,
        #             stop=stop,
        #             prepare_env_args=prepare_env_args
        #         )
        #         response = chat_instance.invoke([HumanMessage(content=message)])
        #         response_str = response.content if hasattr(response, 'content') else str(response)

        #         LLM_runtime = time.time() - start_time
        #         tokens = count_tokens(tokenizer, message)

        #         # Save stats
        #         saver.save_stats('LLM_runtime', LLM_runtime)
        #         saver.save_stats('tokens', tokens)

        #         return response_str

        #     except RateLimitError as e:
        #         wait_time = backoff_factor ** attempt
        #         print(f"@@@{FLAGS.llm_model}: Rate limit exceeded. Retrying in {wait_time} seconds...")
        #         time.sleep(wait_time)
        #     except Exception as e:
        #         wait_time = backoff_factor ** attempt
        #         print(f"@@@{FLAGS.llm_model}: An error occurred: {e}. Retrying in {wait_time} seconds...")
        #         time.sleep(wait_time)
        
        # # If we reach here, no response after retries
        # raise RuntimeError("Failed to get a response after all retries using the new method.")
        raise ValueError("ADLR not supported")
    
    else:
        # Using the original legacy method
        return initiate_chat_with_retry_legacy(
            agent_user,
            agent_coding,
            message=message,
            retries=retries,
            backoff_factor=backoff_factor,
            **kwargs
        )

def call_Mark_llm(system_prompt: str, user_prompt: str, temperature: float = None, timeout=300, model: str = None):
    assert FLAGS.LLM_gateaway == 'Mark'
    
    # Use provided temperature or default from FLAGS
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    
    # Use provided model or default from FLAGS
    effective_model = model if model is not None else FLAGS.llm_model

    config_path = os.path.join(os.path.dirname(__file__), "OAI_CONFIG", "OAI_CONFIG_LIST_DAR")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        # config_data is a list, we need to extract the first config
        base_config = config_data[0] if isinstance(config_data, list) else config_data
    
    if effective_model == 'gpt-4o-20241120':
        model = 'gpt-4o'
    elif effective_model == 'claude-3-7-sonnet-20250219':
        model = 'claude-3-7-sonnet-20250219'
    elif effective_model == 'o1-20241217':
        model = 'o1'
    else:
        model = effective_model
    
    # Create config_list by updating the base_config with the correct model
    config = copy.deepcopy(base_config)
    config["model"] = model
    config["api_key"] = ""  # Ensure api_key is empty string
    if "gateway_chat_type" not in config:
        config["gateway_chat_type"] = "dar_mark_team"
    
    config_list = [config]

    # Build llm_config - exclude temperature for o-series models
    llm_config = {
        "config_list": config_list, 
        "timeout": timeout,
        "cache_seed": None  # Disable caching to prevent corruption
    }
    if not is_o_series_model(effective_model):
        llm_config["temperature"] = effective_temperature

    # print(f"\n{CYAN}Query:{RESET} {user_prompt[:1000]}")
    # print("-" * 50)

    # breakpoint()
    
    try:
        # Create a copy of config_list to avoid modifying the original
        llm_config_copy = copy.deepcopy(llm_config)
        # Ensure gateway_chat_type is removed from each config in the list
        # for config in llm_config_copy.get("config_list", []):
        #     if isinstance(config, dict) and "gateway_chat_type" in config:
        #         del config["gateway_chat_type"]
        
        gpt = BasicClient(llm_config=llm_config_copy)
        response = gpt.ask_client(system_prompt+"\n\n"+user_prompt)
        
        if FLAGS.debug:
            print(f"{GREEN}Response:{RESET}", response[:200] if response else 'None')

        # breakpoint()
        return response
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        raise

def call_Perflab_llm(system_prompt: str, user_prompt: str, temperature: float = None, timeout=300, model: str = None):
    assert FLAGS.LLM_gateaway == 'perflab'

    # breakpoint()
    
    # Use provided temperature or default from FLAGS
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    
    # Use provided model or default from FLAGS
    effective_model = model if model is not None else FLAGS.llm_model
    
    # Load PerfLab configuration using centralized function
    config_list = load_perflab_config(effective_model)
    
    # Create llm_config for BasicClient - exclude temperature for o-series models
    llm_config = {
        "config_list": config_list,
        "timeout": timeout,
        "cache_seed": None  # Disable caching to prevent corruption
    }
    if not is_o_series_model(effective_model):
        llm_config["temperature"] = effective_temperature
    
    # Remove gateway_chat_type if present in config_list
    for config in llm_config.get("config_list", []):
        if "gateway_chat_type" in config:
            del config["gateway_chat_type"]
    
    # Initialize BasicClient (which uses autogen's OpenAIWrapper internally)
    client = BasicClient(llm_config=llm_config)
    
    # DEBUG: Lily0923
    if FLAGS.debug:
        print(f"\n{CYAN}Query:{RESET} {user_prompt[:200]}...")
        print("-" * 50)
    
    try:
        # Create messages list with proper roles
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Use BasicClient's create method which handles messages with roles
        response_content = client.create(messages=messages, is_extract_response=True)
        
        if FLAGS.debug:
            print(f"{GREEN}Response:{RESET} {response_content[:200] if response_content else 'None'}...")
            print("-" * 50)
        
        return response_content
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, httpx.TimeoutException, httpx.ReadTimeout, InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(10),
)
def llm_inference(system_prompt: str, user_prompt: str, temperature: float = None, LLM_gateaway: str = None, model: str = None):
    """Direct LLM inference function for Mark and Perflab gateways."""
    if LLM_gateaway is None:
        LLM_gateaway = FLAGS.LLM_gateaway
    
    # Use provided temperature or default from FLAGS
    effective_temperature = temperature if temperature is not None else FLAGS.temperature
    
    # Use provided model or default from FLAGS
    effective_model = model if model is not None else FLAGS.llm_model
    
    # Track inference start time
    inference_start_time = time.time()
    
    # Get tokenizer to count tokens for stats
    tokenizer = get_tokenizer(effective_model)
    total_tokens = count_tokens(tokenizer, system_prompt + user_prompt)

    try:
        if LLM_gateaway == 'Mark':
            response = call_Mark_llm(system_prompt, user_prompt, effective_temperature, model=effective_model)
        elif LLM_gateaway == 'perflab':
            response = call_Perflab_llm(system_prompt, user_prompt, effective_temperature, model=effective_model)
        else:
            raise ValueError(f"Invalid model source: {LLM_gateaway}")
            
        # Calculate inference time and log to saver
        inference_time = time.time() - inference_start_time
        saver.save_stats('LLM_runtime', inference_time)
        saver.save_stats('tokens', total_tokens)
        
        return response
        
    except Exception as e:
        # Calculate inference time even for failed calls
        inference_time = time.time() - inference_start_time
        print(f"⚠️ LLM call failed after {inference_time:.2f}s")
        print(f"   Error: {str(e)[:200]}...")
        raise

