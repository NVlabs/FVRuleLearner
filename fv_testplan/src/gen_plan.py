import os
import subprocess
from config import FLAGS
from saver import saver
from utils import get_llm, OurTimer
import spacy
import networkx as nx
from spacy.tokens import Doc, Span
from typing import Tuple, List
from pathlib import Path
from PyPDF2 import PdfReader
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import get_openai_callback
from langchain_community.llms import Ollama
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import matplotlib.pyplot as plt

print = saver.log_info


def gen_plan():
    """
    Main function to generate test plans and SVAs from a design specification,
    and run JasperGold for verification.
    """
    timer = OurTimer()

    print("Starting the test plan generation process...")

    timer.start_timing()
    print("Step 1: Reading the PDF file...")
    file_path = FLAGS.file_path
    spec_text, pdf_stats = read_pdf(file_path)
    timer.time_and_clear("Read PDF")

    print("Step 2: Initializing the language model...")
    llm_agent = get_llm(model_name=FLAGS.llm_model, **FLAGS.llm_args)
    timer.time_and_clear("Initialize LLM")

    print("Step 3: Generating natural language test plans...")
    nl_plans = generate_nl_plans(spec_text, llm_agent)
    timer.time_and_clear("Generate NL plans")

    print("Step 4: Generating SVAs...")
    svas = generate_svas(spec_text, nl_plans, llm_agent)
    timer.time_and_clear("Generate SVAs")

    print("Step 5: Writing SVAs to file...")
    sva_file_path = write_svas_to_file(svas)
    timer.time_and_clear("Write SVAs to file")

    print("Step 6: Generating TCL script...")
    tcl_file_path = generate_tcl_script(sva_file_path)
    timer.time_and_clear("Generate TCL script")

    print("Step 7: Running JasperGold...")
    jasper_report = run_jaspergold(tcl_file_path)
    timer.time_and_clear("Run JasperGold")

    print("Step 8: Analyzing and printing results...")
    analyze_results(pdf_stats, nl_plans, svas, jasper_report)
    timer.time_and_clear("Analyze results")

    print('Test plan generation process completed.')

    # Print the durations log
    timer.print_durations_log(print_func=print)


def read_pdf(file_path: str) -> Tuple[str, dict]:
    """
    Read a PDF file and extract its content.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        Tuple[str, dict]: A tuple containing the extracted text and file statistics.
    """
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    stats = {
        "num_pages": len(pdf_reader.pages),
        "num_tokens": len(text.split()),
        "file_size": os.path.getsize(file_path),
    }

    print(f'spec_text=\n{text}')
    return text, stats


def generate_nl_plans(spec_text: str, llm_agent) -> List[str]:
    nl_gen_prompt = f"""
    Given the following design specification, generate natural language test plans:

    {spec_text}

    Use the following examples as a guide for the format and style of the test plans:

    1. that when x_data is within the range of 230 to 255, in the next cycle x_data will remain within the range of 205 to 255. Use the signals 'sclk' for the clock edge and 'x_data' for the data being checked.
    2. that the input data is within the valid range when not in reset. Use the signals 'rstn', 'sclk', and 'x_data'.
    3. that if the input data 'x_data' is within the range of 138 to 153 inclusive, then in the subsequent cycles, 'x_data' must continue to be within the range of 98 to 153 inclusive. Use the signals 'x_data' and 'sclk'.
    4. that the input data x_data has a value between 83 and 165, inclusive, 3 clock cycles after the reset signal rstn becomes deasserted. Use the signals 'rstn', 'sclk', and 'x_data'.
    5. that the input data signal 'x_data' is within the range 0 to 45 inclusive, starting from four clock cycles after the reset signal 'rstn' becomes deasserted. Use the signals 'rstn', 'sclk', and 'x_data'.

    Generate diverse test plans based on the given specification.
    """

    prompt_template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    llm_chain = LLMChain(prompt=prompt_template, llm=llm_agent)

    try:
        if isinstance(llm_agent, Ollama):
            result = llm_chain.run(prompt=nl_gen_prompt)
            print(f'Generated result using Ollama')
        else:
            with get_openai_callback() as cb:
                result = llm_chain.run(prompt=nl_gen_prompt)
                print(f'Spent a total of {cb.total_tokens} tokens')

        nl_plans = [plan.strip() for plan in result.split('\n') if plan.strip()]
        return nl_plans
    except Exception as e:
        print(f"Error generating NL description: {str(e)}")
        raise


def generate_svas(spec_text: str, nl_plans: List[str], llm_agent) -> List[str]:
    """
    Generate SVAs using LLM based on the design specification and natural language test plans.

    Args:
        spec_text (str): The design specification text.
        nl_plans (List[str]): List of natural language test plans.
        llm_agent: The language model agent.

    Returns:
        List[str]: A list of generated SVAs.
    """
    sva_gen_prompt = f"""
    Given the following design specification and natural language test plans, generate SVAs (System Verilog Assertions):

    Specification:
    {spec_text}

    Test Plans:
    {' '.join(nl_plans)}

    Generate one SVA for each of the provided natural language test plans. Each SVA should be in the following format:
    @(posedge PCLK) <condition> |-> <consequence>;

    Ensure that each SVA is a complete and valid System Verilog assertion.
    """

    prompt_template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    llm_chain = LLMChain(prompt=prompt_template, llm=llm_agent)

    try:
        with get_openai_callback() as cb:
            result = llm_chain.run(prompt=sva_gen_prompt)
            print(f'Spent a total of {cb.total_tokens} tokens')

        svas = [
            sva.strip()
            for sva in result.split('\n')
            if sva.strip() and sva.strip().startswith("@(posedge PCLK)")
        ]

        if not svas:
            print(
                "Warning: No valid SVAs were generated. Please check the output and adjust the prompt if necessary."
            )
        else:
            print(f"Generated {len(svas)} SVAs.")

        return svas
    except Exception as e:
        print(f"Error generating SVAs: {str(e)}")
        return []


def write_svas_to_file(svas: List[str]) -> str:
    """
    Write generated SVAs to a file, preserving the module interface from the original file
    and formatting the SVAs correctly.

    Args:
        svas (List[str]): List of generated SVAs.

    Returns:
        str: Path to the generated SVA file.
    """
    original_sva_path = os.path.join(FLAGS.design_dir, "property_goldmine.sva")

    # Extract the module interface from the original file
    with open(original_sva_path, "r") as f:
        original_content = f.read()

    # Find the module declaration
    module_start = original_content.find("module i_apb(")
    module_end = original_content.find(");", module_start) + 2  # +2 to include ");"

    module_interface = original_content[module_start:module_end]

    # Construct the new SVA file content
    sva_file_content = f"{module_interface}\n\n"

    for i, sva in enumerate(svas):
        # Extract only the SVA part, removing any NL descriptions
        sva_only = sva.split("NL=")[0].strip()
        if sva_only.startswith("SVA "):
            sva_only = sva_only.split(":", 1)[1].strip()

        # Format the SVA as a property
        property_name = f"a{len(svas) - i - 1}"
        sva_file_content += f"property {property_name};\n"
        sva_file_content += f"{sva_only}\n"
        sva_file_content += f"endproperty\n"
        sva_file_content += (
            f"assert_{property_name}: assert property({property_name});\n\n"
        )

    sva_file_content += "endmodule\n"

    sva_file_path = os.path.join(saver.logdir, "property_goldmine.sva")
    with open(sva_file_path, "w") as f:
        f.write(sva_file_content)

    return sva_file_path


def generate_tcl_script(sva_file_path: str) -> str:
    """
    Generate a TCL script for JasperGold.

    Args:
        sva_file_path (str): Path to the SVA file.

    Returns:
        str: Path to the generated TCL script.
    """
    design_dir = FLAGS.design_dir
    if not os.path.exists(design_dir):
        raise Exception(f"Design directory {design_dir} does not exist")

    tcl_content = f"""
# Analyze design under verification files
set ROOT_PATH {design_dir}
set RTL_PATH ${{ROOT_PATH}}
set PROP_PATH ${{ROOT_PATH}}

analyze -v2k \\
  ${{RTL_PATH}}/apb.v

# Analyze property files
analyze -sva \\
  ${{RTL_PATH}}/bindings.sva \\
  {sva_file_path}

# Elaborate design and properties
elaborate -top apb

# Set up Clocks and Resets
clock PCLK
reset PRESETn

# Get design information to check general complexity
get_design_info

# Prove properties
prove -all

# Report proof results
report
"""

    tcl_file_path = os.path.join(saver.logdir, "FPV_apb.tcl")
    with open(tcl_file_path, "w") as f:
        f.write(tcl_content)

    return tcl_file_path


def run_jaspergold(tcl_file_path: str) -> str:
    """
    Run JasperGold using the generated TCL script.

    Args:
        tcl_file_path (str): Path to the TCL script.

    Returns:
        str: JasperGold report content.
    """
    jasper_command = (
        f"/home/tools/tempusquest/jasper_2023.12/bin/jg -batch -tcl {tcl_file_path}"
    )

    try:
        result = subprocess.run(
            jasper_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=FLAGS.design_dir,
        )
        report = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running JasperGold: {e}")
        report = e.output

    report_file_path = os.path.join(saver.logdir, "jasper_report.txt")
    with open(report_file_path, "w") as f:
        f.write(report)

    return report


def analyze_results(
    pdf_stats: dict, nl_plans: List[str], svas: List[str], jasper_report: str
):
    """
    Analyze and print statistics about the generated plans, SVAs, and JasperGold results.

    Args:
        pdf_stats (dict): Statistics about the input PDF file.
        nl_plans (List[str]): List of generated natural language test plans.
        svas (List[str]): List of generated SVAs.
        jasper_report (str): JasperGold report content.
    """
    print("PDF Statistics:")
    print(f"  Number of pages: {pdf_stats['num_pages']}")
    print(f"  Number of tokens: {pdf_stats['num_tokens']}")
    print(f"  File size: {pdf_stats['file_size']} bytes")

    print("\nNatural Language Test Plans:")
    print(f"  Number of plans generated: {len(nl_plans)}")
    if nl_plans:
        print(
            f"  Average plan length: {sum(len(plan.split()) for plan in nl_plans) / len(nl_plans):.2f} words"
        )
        print("  Sample plans:")
        for i, plan in enumerate(nl_plans, 1):
            print(f"    {i}. {plan}")

    print("\nSVAs:")
    print(f"  Number of SVAs generated: {len(svas)}")
    if svas:
        print(
            f"  Average SVA length: {sum(len(sva.split()) for sva in svas) / len(svas):.2f} words"
        )
        print("  Sample SVAs:")
        for i, sva in enumerate(svas, 1):
            print(f"    {i}. {sva}")

    print("\nJasperGold Results:")
    print(f"  Report saved to: {os.path.join(saver.logdir, 'jasper_report.txt')}")
    # TODO: Add more detailed analysis of the JasperGold report here


prompt = '''
We are a research team trying to build an end-to-end pipeline from design spec to natural language (NL) test plans to SVAs. Specifically, write code to read '/home/scratch.yunshengb_cpu/fv/fv_testplan/data/apb/apbi2c_spec.pdf' into text and write a high-quality prompt (NL_gen_prompt) to ask FLAGS.llm_model (via     llm_agent = get_llm(model_name=FLAGS.llm_model) and the following code:
"""
    prompt_template = PromptTemplate(input_variables=["prompt"], template="{prompt}")

    llm_chain = LLMChain(prompt=prompt_template, llm=llm_agent)

    try:
        with get_openai_callback() as cb:
            result = llm_chain.run(prompt=prompt)
            print(f'Spent a total of {cb.total_tokens} tokens')

        nl_description = result.strip()
        if nl_description.startswith("NL:") or nl_description.startswith("NL=="):
            nl_description = nl_description.split(maxsplit=1)[1].strip()

        return nl_description
    except Exception as e:
        print(f"Error generating NL description: {str(e)}")
        raise  # Re-raise the exception to trigger a retry

""") to generate NL test plans. Then use these test plans together with the loaded design spec (SVA_gen_prompt) and call FLAGS.llm_model to generate SVAs for this design.

Make sure your 'NL_gen_prompt' use the following in-context learning (ICL) examples:
"""
that when x_data is within the range of 230 to 255, in the next cycle x_data will remain within the range of 205 to 255. Use the signals 'sclk' for the clock edge and 'x_data' for the data being checked.
that the input data is within the valid range when not in reset. Use the signals 'rstn', 'sclk', and 'x_data'.
that if the input data 'x_data' is within the range of 138 to 153 inclusive, then in the subsequent cycles, 'x_data' must continue to be within the range of 98 to 153 inclusive. Use the signals 'x_data' and 'sclk'.
that the input data x_data has a value between 83 and 165, inclusive, 3 clock cycles after the reset signal rstn becomes deasserted. Use the signals 'rstn', 'sclk', and 'x_data'.
that the input data signal 'x_data' is within the range 0 to 45 inclusive, starting from four clock cycles after the reset signal 'rstn' becomes deasserted. Use the signals 'rstn', 'sclk', and 'x_data'.
"""
These examples are machine-generated from some ground-truth SVAs.

Make sure your 'SVA_gen_prompt' use the following 5 ICL examples:
"""
SVA=@(posedge sclk) ((x_data >= 230) && (x_data <= 255)) |-> (x_data >= 205) && (x_data <= 255)	
NL=that when x_data is within the range of 230 to 255, in the next cycle x_data will remain within the range of 205 to 255. Use the signals 'sclk' for the clock edge and 'x_data' for the data being checked.
SVA=SVA=@(posedge sclk) (rstn) |-> (x_data >= 0) && (x_data <= 45)	
NL=that the input data is within the valid range when not in reset. Use the signals 'rstn', 'sclk', and 'x_data'.
@(posedge sclk) ((x_data >= 138) && (x_data <= 153)) |-> (x_data >= 98) && (x_data <= 153)	
NL=that if the input data 'x_data' is within the range of 138 to 153 inclusive, then in the subsequent cycles, 'x_data' must continue to be within the range of 98 to 153 inclusive. Use the signals 'x_data' and 'sclk'.
SVA=@(posedge sclk) (rstn ##3 1) |-> (x_data >= 83) && (x_data <= 165)	
NL=that the input data x_data has a value between 83 and 165, inclusive, 3 clock cycles after the reset signal rstn becomes deasserted. Use the signals 'rstn', 'sclk', and 'x_data'.
SVA=@(posedge sclk) (rstn ##4 1) |-> (x_data >= 0) && (x_data <= 45)	
NL=that the input data signal 'x_data' is within the range 0 to 45 inclusive, starting from four clock cycles after the reset signal 'rstn' becomes deasserted. Use the signals 'rstn', 'sclk', and 'x_data'.
"""

Eventually, print some stats lile PDF length (#pages, #tokens, file size, etc.), #NL plans generated by the LLM and some stats about these plans and the actual plans, #SVAs generated by the LLM (one SVA per plan) and some stats about these SVAs and the actual SVAs.

Use the skeleton code below:
"""
from config import FLAGS
from saver import saver
from utils import get_llm
import spacy
import networkx as nx
from spacy.tokens import Doc, Span
from typing import Tuple, List
from pathlib import Path


def gen_plan():
    print('done')

"""

Make sure the code has helper functions and documentations/annotations for readability.



'''
