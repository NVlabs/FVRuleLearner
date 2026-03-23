# python gen_assertionbench_data.py 2>&1 | tee log.txt


import os
import tarfile
import csv
import re
from typing import List, Tuple, Dict
import adlrchat
from adlrchat import ADLRChat, LLMGatewayChat, ADLRCompletion
from adlrchat.vault import prepare_env
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from copy import deepcopy
from tqdm import tqdm
import random, glob

random.seed(123)  # Set a fixed seed for reproducibility
from prettytable import PrettyTable
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import sys
from tenacity import retry, stop_after_attempt, wait_exponential
from difflib import SequenceMatcher

# correct_csv_path = None # regenerate
correct_csv_path = "/home/scratch.yunshengb_cpu/fv/fv_testplan/src/module_sva_nl.csv"  # optional
# MODE = 'normal'
MODE = 'postprocess'


def main():

    if MODE == 'postprocess':
        final_csv = "/home/scratch.yunshengb_cpu/fv/fv_testplan/src/module_sva_nl.csv"
        # Post-process the final CSV
        post_process_csv(final_csv)
        return

    root_dir = "/home/scratch.yunshengb_cpu/assertion_data_for_LLM/verified_assertions"
    initial_csv = "module_sva_initial.csv"
    sampled_csv = "module_sva_sampled.csv"
    final_csv = "module_sva_nl.csv"

    print("Step 1: Extracting all property.sva files...")
    extract_all_property_sva(root_dir)
    print("Extraction complete.")

    print("\nStep 2: Creating initial SVA dictionary...")
    try:
        svas_dict = create_initial_csv(root_dir, initial_csv)
    except Exception as e:
        print(f"Error in creating initial CSV: {str(e)}")
        return  # Exit the script if there's an error

    print("\nStep 3: Sampling SVAs...")
    sampled_svas, stats = sample_svas(svas_dict)

    print(f"\nWriting sampled SVAs to {sampled_csv}")
    with open(sampled_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["module_name", "module_interface", "SVA"])
        csv_writer.writerows(sampled_svas)

    print(
        f"Sampled CSV contains {len(set(row[0] for row in sampled_svas))} unique modules"
    )

    # Sort the sampled SVAs by module name
    sampled_svas.sort(key=lambda x: x[0])  # Sort by the first element (module_name)

    print("\nSampling Statistics:")
    print_sampling_stats(stats)

    print(f"\nWriting sampled SVAs to {sampled_csv}")
    with open(sampled_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["module_name", "module_interface", "SVA"])
        csv_writer.writerows(sampled_svas)

    target_total = 1000  # Define the target number of entries

    if correct_csv_path and os.path.exists(correct_csv_path):
        print(f"\nStep 4: Updating module interfaces in correct CSV...")
        update_module_interfaces_preserve_nl(
            sampled_csv, correct_csv_path, "temp_final.csv"
        )

        print("\nStep 5: Modifying NL descriptions that contain SVA-like parts...")
        modified_count = process_csv_modify_nl(
            "temp_final.csv", "modified_temp_final.csv"
        )
        os.remove("temp_final.csv")  # Remove the temporary file

        print(f"Modified {modified_count} entries where NL contained SVA-like parts.")

        print("\nStep 6: Merging and balancing CSV files...")
        merge_and_balance_csv(
            correct_csv_path, "modified_temp_final.csv", final_csv, target_total=1000
        )
        os.remove("modified_temp_final.csv")  # Clean up intermediate file

    else:
        print(f"\nStep 4: Adding prompts to sampled SVAs...")
        add_prompts_to_csv(sampled_csv, "module_sva_with_prompts.csv")
        print(f"CSV file with prompts created: module_sva_with_prompts.csv")

        print("\nStep 5: Generating NL descriptions...")
        generate_nl_descriptions("module_sva_with_prompts.csv", final_csv)

    post_process_csv(final_csv)

    print(f"Final CSV file created: {final_csv}")

    # Check if the final CSV contains the target number of entries
    with open(final_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        actual_entries = sum(1 for _ in reader)

    if actual_entries != 1000:
        print(
            f"Warning: Final CSV contains {actual_entries} entries, "
            f"which is different from the target of 1000."
        )
    else:
        print(f"Final CSV contains the target number of 1000 entries.")

    # Print final statistics
    print_final_statistics(final_csv)


def post_process_csv(csv_path: str):
    """
    Post-process the CSV by removing SVA-containing NL descriptions and sorting the rows by module name.
    """
    processed_rows = []

    with open(csv_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            module_name, module_interface, sva, nl = row
            modified_nl, _, _ = extract_sva_like_part(nl, sva)
            processed_rows.append([module_name, module_interface, sva, modified_nl])

    # Sort the rows by module name
    processed_rows.sort(key=lambda x: x[0])

    with open(csv_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(processed_rows)

    print(f"Post-processed and sorted CSV saved to {csv_path}")


def extract_sva_like_part(nl: str, sva: str) -> tuple:
    """
    Extract the part of the NL description that is too similar to the SVA.

    Args:
    nl (str): Natural language description
    sva (str): System Verilog Assertion

    Returns:
    tuple: (modified NL, extracted part, bool indicating if modification was made)
    """
    # Remove "assert property" and surrounding parentheses from the SVA
    sva_content = sva.strip()

    # Check if the entire SVA content is in the NL
    if sva_content.lower() in nl.lower():
        # Remove the SVA content from the NL
        pattern = re.compile(re.escape(sva_content), re.IGNORECASE)
        modified_nl = pattern.sub('', nl, count=1).strip()
        # Additionally remove "SVA: assert property();" if present
        modified_nl = re.sub(
            r'SVA: assert property\(\);\s*', '', modified_nl, flags=re.IGNORECASE
        ).strip()
        return modified_nl, sva_content, True

    # If not, don't modify the NL
    return nl, "", False


def process_csv_modify_nl(input_csv: str, output_csv: str) -> int:
    modified_count = 0
    processed_rows = []

    with open(input_csv, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            module_name, module_interface, sva, nl = row
            modified_nl, extracted_part, was_modified = extract_sva_like_part(nl, sva)
            if was_modified:
                modified_count += 1
                processed_rows.append([module_name, module_interface, sva, modified_nl])
            else:
                processed_rows.append(row)

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(processed_rows)

    print(f"Modified {modified_count} entries where NL contained SVA-like parts.")
    print(
        f"Number of unique modules after modification: {len(set(row[0] for row in processed_rows))}"
    )
    return modified_count


def update_module_interfaces_preserve_nl(
    sampled_csv: str, correct_csv: str, output_csv: str
):
    sampled_dict = {}
    with open(sampled_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            module_name, new_interface, sva = row
            if module_name not in sampled_dict:
                sampled_dict[module_name] = []
            sampled_dict[module_name].append((new_interface, sva))

    correct_entries = []
    new_entries = []
    with open(correct_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            module_name, old_interface, sva, nl = row
            if module_name in sampled_dict and sampled_dict[module_name]:
                new_interface, new_sva = sampled_dict[module_name].pop(0)
                correct_entries.append([module_name, new_interface, new_sva, nl])
            else:
                correct_entries.append(row)

    # Add any remaining sampled entries as new
    for module_name, entries in sampled_dict.items():
        for new_interface, new_sva in entries:
            new_entries.append([module_name, new_interface, new_sva, ""])

    # Combine correct_entries and new_entries
    all_entries = correct_entries + new_entries

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_entries)

    print(f"Updated CSV with preserved NL descriptions saved to {output_csv}")
    print(
        f"Number of unique modules in updated CSV: {len(set(row[0] for row in all_entries))}"
    )
    print(f"Number of entries in updated CSV: {len(all_entries)}")
    print(f"Number of new entries added: {len(new_entries)}")


def extract_all_property_sva(root_dir: str):
    """
    Extract all property.sva.tar.gz files in the directory structure.
    """
    for top_level in tqdm(
        os.listdir(root_dir), desc="Processing top-level directories"
    ):
        top_level_path = os.path.join(root_dir, top_level)
        if os.path.isdir(top_level_path):
            for secondary_level in tqdm(
                os.listdir(top_level_path), desc=f"Processing {top_level}", leave=False
            ):
                secondary_level_path = os.path.join(top_level_path, secondary_level)
                if os.path.isdir(secondary_level_path):
                    tar_gz_path = os.path.join(
                        secondary_level_path, "property.sva.tar.gz"
                    )
                    if os.path.exists(tar_gz_path):
                        try:
                            extract_tar_gz(tar_gz_path, secondary_level_path)
                        except Exception as e:
                            print(f"Error extracting {tar_gz_path}: {str(e)}")


def create_initial_csv(
    root_dir: str, output_csv: str
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Create a CSV file with module information and SVAs (without NL descriptions).
    Returns a dictionary with module names as keys and lists of (module_interface, SVA) as values.
    """
    results = {}
    for top_level in os.listdir(root_dir):
        top_level_path = os.path.join(root_dir, top_level)
        if os.path.isdir(top_level_path):
            for secondary_level in os.listdir(top_level_path):
                secondary_level_path = os.path.join(top_level_path, secondary_level)
                if os.path.isdir(secondary_level_path):
                    module_name = f"{top_level}/{secondary_level}"
                    verilog_file = find_verilog_file(
                        secondary_level_path, secondary_level
                    )
                    sva_path = find_property_sva(secondary_level_path)

                    if verilog_file and sva_path:
                        try:
                            module_interface = get_module_interface(verilog_file)
                            svas = extract_svas(sva_path)
                            results[module_name] = [
                                (module_interface, sva) for sva in svas
                            ]
                        except Exception as e:
                            print(f"Error processing files for {module_name}: {str(e)}")
                            raise  # This will cause the script to break instead of continue
                    else:
                        print(
                            f"Warning: Missing Verilog file or SVA file for {module_name}"
                        )

    # Write initial CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["module_name", "module_interface", "SVA"])
        for module_name, svas in results.items():
            for module_interface, sva in svas:
                csv_writer.writerow([module_name, module_interface, sva])

    print(f"Initial CSV created with {len(results)} unique modules")
    return results


def find_verilog_file(directory: str, module_name: str) -> str:
    """
    Search for a Verilog file in the given directory that matches the module name.
    """
    potential_files = glob.glob(os.path.join(directory, '*.v'))
    best_match = None
    highest_similarity = 0.0
    for file in potential_files:
        filename = os.path.basename(file)
        similarity = SequenceMatcher(
            None, module_name.lower(), filename.lower()
        ).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = file
    # if highest_similarity < 0.5:  # Threshold to avoid false matches
    #     best_match = None
    if best_match:
        print(
            f"Matched Verilog file '{best_match}' with module '{module_name}' (similarity: {highest_similarity:.2f})"
        )
    else:
        print(f"No suitable Verilog file found for module '{module_name}'")
    return best_match


def generate_nl_for_new_module(
    module_name: str, module_interface: str, sva: str
) -> str:
    """
    Generate NL description for a new module using the LLM.
    """
    llm_agent = get_llm(model_name='gpt-4-turbo')
    prompt = generate_nl_prompt(module_name, module_interface, sva)

    try:
        nl_description = get_nl_description_with_retry(llm_agent, prompt)
        print(f'nl_description generated: {nl_description}')
        return nl_description
    except Exception as e:
        print(
            f"Failed to generate NL description for new module {module_name}: {str(e)}"
        )
        return ""  # Return empty string if generation fails


def merge_and_balance_csv(
    correct_csv: str, new_csv: str, output_csv: str, target_total: int = 1000
):
    correct_entries = []
    new_entries = []

    # Read correct CSV
    with open(correct_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        correct_entries = list(reader)

    # Read new CSV
    with open(new_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        new_entries = list(reader)

    print(
        f"Correct CSV contains {len(set(entry[0] for entry in correct_entries))} unique modules"
    )
    print(
        f"New CSV contains {len(set(entry[0] for entry in new_entries))} unique modules"
    )

    # Identify new modules
    existing_modules = set(entry[0] for entry in correct_entries)
    new_modules = [entry for entry in new_entries if entry[0] not in existing_modules]

    print(f"Found {len(set(entry[0] for entry in new_modules))} new unique modules")

    # Generate NL for new modules if missing
    for i, entry in enumerate(new_modules):
        if len(entry) < 4 or not entry[3]:  # If NL is missing
            module_name, module_interface, sva = entry[:3]
            print(f'{i}/{len(new_modules)}')
            nl = generate_nl_for_new_module(module_name, module_interface, sva)
            new_modules[i] = entry[:3] + [nl]

    if new_modules:
        print("New modules:", set(entry[0] for entry in new_modules))
    else:
        print(
            "No new modules found. Existing modules in correct CSV:", existing_modules
        )
        print("Modules in new CSV:", set(entry[0] for entry in new_entries))

    # Prioritize including all new modules
    if new_modules:
        space_for_new = min(
            len(new_modules), target_total // 2
        )  # Reserve up to half the space for new modules
        new_modules = new_modules[:space_for_new]
        correct_entries = correct_entries[: target_total - len(new_modules)]

    # If we still have space, add more from new_entries that aren't new modules
    remaining_space = target_total - (len(correct_entries) + len(new_modules))
    if remaining_space > 0:
        additional_entries = [
            entry for entry in new_entries if entry not in new_modules
        ]
        additional_entries = random.sample(
            additional_entries, min(remaining_space, len(additional_entries))
        )
        new_modules.extend(additional_entries)

    # Combine and shuffle
    final_entries = correct_entries + new_modules
    random.shuffle(final_entries)

    # Ensure we don't exceed the target total
    final_entries = final_entries[:target_total]

    # Write to output CSV
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(final_entries)

    print(
        f"Merged CSV created with {len(final_entries)} entries, including {len(set(entry[0] for entry in new_modules))} new unique modules."
    )
    print(f"New unique modules added: {set(entry[0] for entry in new_modules)}")


def sample_svas(
    svas_dict: Dict[str, List[Tuple[str, str, str]]],
    target_total: int = 1000,
    min_per_module: int = 1,
) -> List[Tuple[str, str, str, str]]:
    sampled_results = []
    stats = []

    # Ensure at least one SVA per module
    for module_name, module_svas in svas_dict.items():
        sampled_sva = random.choice(module_svas)
        sampled_results.append((module_name, sampled_sva[0], sampled_sva[1]))
        stats.append((module_name, len(module_svas), 1))

    # If we have space, add more SVAs randomly
    remaining_slots = target_total - len(sampled_results)
    if remaining_slots > 0:
        all_svas = [
            (module, interface, sva)
            for module, svas in svas_dict.items()
            for interface, sva in svas
            if (module, interface, sva) not in sampled_results
        ]

        additional_samples = random.sample(
            all_svas, min(remaining_slots, len(all_svas))
        )
        sampled_results.extend(additional_samples)

        # Update stats
        for module, _, _ in additional_samples:
            for i, (stat_module, orig_count, sampled_count) in enumerate(stats):
                if stat_module == module:
                    stats[i] = (stat_module, orig_count, sampled_count + 1)
                    break

    unique_modules_sampled = set(result[0] for result in sampled_results)
    print(f"Sampled SVAs from {len(unique_modules_sampled)} unique modules")
    return sampled_results, stats


def print_sampling_stats(stats: List[Tuple[str, int, int]]):
    """
    Print a table showing sampling statistics for each module.
    """
    table = PrettyTable()
    table.field_names = ["Module Name", "Original SVAs", "Sampled SVAs"]
    for module_name, original_count, sampled_count in stats:
        table.add_row([module_name, original_count, sampled_count])
    print(table)


def generate_nl_descriptions(input_csv: str, output_csv: str, save_interval: int = 10):
    """
    Generate NL descriptions for each SVA in the input CSV and create a new CSV with the descriptions.
    Saves intermediate results every 'save_interval' iterations.
    """
    llm_agent = get_llm(model_name='gpt-4-turbo')
    results = []
    intermediate_csv = "intermediate_results.csv"

    # Count total number of rows
    with open(input_csv, 'r') as csvfile:
        total_rows = sum(1 for row in csv.reader(csvfile)) - 1  # Subtract 1 for header

    # Check if intermediate results exist and load them
    if os.path.exists(intermediate_csv):
        with open(intermediate_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            results = list(reader)
        print(f"Loaded {len(results)} existing results from {intermediate_csv}")

    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip header

        # Skip rows that have already been processed
        for _ in range(len(results)):
            next(csv_reader, None)

        for i, row in enumerate(
            tqdm(
                csv_reader,
                total=total_rows - len(results),
                desc="Generating NL descriptions",
                unit="SVA",
                disable=False,
                file=sys.stdout,
            )
        ):
            module_name, module_interface, sva, prompt = row
            try:
                nl_description = get_nl_description_with_retry(llm_agent, prompt)
                if nl_description:  # Only add to results if a description was generated
                    results.append((module_name, module_interface, sva, nl_description))
                    print(f"\nSVA {len(results)}/{total_rows} for {module_name}:")
                    print(f"NL Description: {nl_description}")
                    print("-------------------------")

                # Save intermediate results every 'save_interval' iterations
                if (len(results) % save_interval == 0) or (len(results) == total_rows):
                    with open(intermediate_csv, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(
                            ["module_name", "module_interface", "SVA", "NL"]
                        )
                        csv_writer.writerows(results)
                    print(f"\nSaved intermediate results to {intermediate_csv}")

            except Exception as e:
                print(
                    f"\nFailed to generate NL description for {module_name} after multiple retries:"
                )
                print(f"SVA: {sva}")
                print(f"Error: {str(e)}")
                print("-------------------------")

            sys.stdout.flush()

    # Save final results
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["module_name", "module_interface", "SVA", "NL"])
        csv_writer.writerows(results)
    print(f"\nSaved final results to {output_csv}")

    # Remove the intermediate file
    if os.path.exists(intermediate_csv):
        os.remove(intermediate_csv)
        print(f"Removed intermediate file {intermediate_csv}")


def extract_tar_gz(tar_gz_path: str, extract_path: str) -> None:
    """
    Extract a .tar.gz file to the specified path.

    Args:
    tar_gz_path (str): Path to the .tar.gz file
    extract_path (str): Path to extract the contents
    """
    try:
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully extracted {tar_gz_path} to {extract_path}")
    except Exception as e:
        print(f"Error extracting {tar_gz_path}: {str(e)}")
        raise


def find_property_sva(start_path: str) -> str:
    """
    Recursively search for 'property.sva' file starting from the given path.

    Args:
    start_path (str): Path to start the search from

    Returns:
    str: Path to the 'property.sva' file if found, None otherwise
    """
    for root, dirs, files in os.walk(start_path):
        if 'property.sva' in files:
            return os.path.join(root, 'property.sva')
    return None


class EmptyModuleInterfaceError(Exception):
    """Exception raised when a module interface is empty."""

    pass


def get_module_interface(verilog_file_path: str) -> str:
    """
    Extract the module interface from a Verilog file, preserving the module name and formatting.

    Args:
    verilog_file_path (str): Path to the Verilog file

    Returns:
    str: The extracted module interface with the module name

    Raises:
    EmptyModuleInterfaceError: If the extracted module interface is empty
    """
    with open(verilog_file_path, 'r') as file:
        content = file.read()
        # Regular expression to match the module interface
        pattern = r'module\s+(\w+)\s*(?:#\s*\([^)]*\))?\s*\(([\s\S]*?)\);\s*endmodule'
        match = re.search(pattern, content)
        if match:
            module_name = match.group(1)
            port_list = match.group(2)

            # Preserve original formatting, including comments
            lines = [line.strip() for line in port_list.split('\n')]
            formatted_ports = []
            for line in lines:
                if line.startswith('//'):
                    # Preserve full-line comments
                    formatted_ports.append(f"    {line}")
                elif '//' in line:
                    # Preserve inline comments
                    parts = line.split('//')
                    formatted_ports.append(f"    {parts[0].strip()} //{parts[1]}")
                elif line:
                    # Regular port lines
                    formatted_ports.append(f"    {line}")

            # Construct the module interface
            module_interface = f"module {module_name} (\n"
            module_interface += '\n'.join(formatted_ports)
            module_interface += "\n);"

            result = module_interface.strip()
            if not result:
                raise EmptyModuleInterfaceError(
                    f"Empty module interface extracted from {verilog_file_path}"
                )
            return result

    raise EmptyModuleInterfaceError(f"No module interface found in {verilog_file_path}")



def update_module_interfaces(sampled_csv: str, final_csv: str):
    """
    Update module interfaces in the existing CSV file without regenerating NL descriptions.
    """
    updated_rows = []
    with open(sampled_csv, 'r') as sample_file, open(final_csv, 'r') as final_file:
        sample_reader = csv.reader(sample_file)
        final_reader = csv.reader(final_file)

        next(sample_reader)  # Skip header
        header = next(final_reader)

        for sample_row, final_row in zip(sample_reader, final_reader):
            module_name, new_interface, sva = sample_row
            _, old_interface, _, nl = final_row
            updated_rows.append([module_name, new_interface, sva, nl])

    # Sort the updated rows by module name
    updated_rows.sort(key=lambda x: x[0])

    with open(final_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(updated_rows)


def extract_svas(sva_file_path: str) -> List[str]:
    """
    Extract SVAs from the property.sva file.

    Args:
    sva_file_path (str): Path to the property.sva file

    Returns:
    List[str]: List of extracted SVAs
    """
    with open(sva_file_path, 'r') as file:
        content = file.read()
        # Regular expression to match assert property statements
        pattern = r'assert property\s*\((.*?)\);'
        return re.findall(pattern, content, re.DOTALL)


def generate_nl_prompt(module_name: str, module_interface: str, sva: str) -> str:
    """
    Generate a prompt for GPT-4-turbo to create a concise yet informative NL description of the SVA.
    """
    examples = [
        (
            "that the counter does not overflow. Use the signals 'count', 'count_d1', 'jump_vld_d1', and 'tb_reset_1_cycle_pulse_shadow'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) ((count_d1 === max) && !jump_vld_d1 && ((count <= min) || (count > max)) && !tb_reset_1_cycle_pulse_shadow) !== 1'b1 );",
        ),
        (
            "that the arbiter grant signal is 0-1-hot. Use the signal 'tb_gnt'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) !($onehot0(tb_gnt)) !== 1'b1 );",
        ),
        (
            "that each of the FSM states provided as the input sequence should end in the final end state. Use the signal 'match_tracker'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) !(|(match_tracker[0])) !== 1'b1 );",
        ),
        (
            "that the FIFO does not underflow, assuming no bypass. Use the signals 'rd_pop' and 'fifo_empty'.",
            "asrt : assert property (@(posedge clk) disable iff (tb_reset) (fifo_empty && rd_pop) !== 1'b1 );",
        ),
        (
            "for forward progress: whenever there is a push, an output data pop eventually happens. Use the signals 'tb_push_cnt' and 'tb_pop_cnt'.",
            "asrt: assert property(@(posedge clk) disable iff (tb_reset) (|tb_push_cnt) |-> strong(##[0:$] |tb_pop_cnt) );",
        ),
        (
            "that the arbiter grant signal is 0-1-hot. Use the signal 'tb_gnt'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) !($onehot0(tb_gnt)) !== 1'b1 );",
        ),
        (
            "that the specified fsm_sequence is never seen. Use the signal 'tb_sequence_seen'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) (|tb_sequence_seen) !== 1'b1 );",
        ),
        (
            "that the arbiter grant signal is 0-1-hot. Use the signal 'tb_gnt'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) !($onehot0(tb_gnt)) !== 1'b1 );",
        ),
        (
            "that the FIFO does not underflow, assuming no bypass. Use the signals 'rd_pop' and 'fifo_empty'.",
            "asrt : assert property (@(posedge clk) disable iff (tb_reset) (fifo_empty && rd_pop) !== 1'b1 );",
        ),
        (
            "that the arbiter grant signal is 0-1-hot. Use the signal 'tb_gnt'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) !($onehot0(tb_gnt)) !== 1'b1 );",
        ),
        (
            "that the FIFO does not underflow, assuming no bypass. Use the signals 'rd_pop' and 'fifo_empty'.",
            "asrt : assert property (@(posedge clk) disable iff (tb_reset) (fifo_empty && rd_pop) !== 1'b1 );",
        ),
        (
            "read address is not greater than number of entries i.e. the read address is not pointing to an address not in the memory. Use the signals 're' and 'ra'.",
            "asrt: assert property (@(posedge clk) disable iff (tb_reset) (re |-> ra < entries) );",
        ),
    ]

    examples_str = "\n".join([f"NL: {nl}\nSVA: {sva}" for nl, sva in examples])

    prompt = f"""Given the following information about a hardware module and its System Verilog Assertion (SVA), 
    provide a concise yet informative natural language description of what the SVA means and what property it is checking.
    Your description should be similar in style and length to the examples provided below, but ensure it includes all necessary details for someone to reconstruct the SVA from your description alone.

    Key requirements for your description:
    1. Be concise, but don't sacrifice critical information.
    2. Include all specific values, ranges, and timing information present in the SVA.
    3. Clearly state the relationship between signals and conditions.
    4. Mention all relevant signal names used in the SVA.
    5. Start your description immediately without any prefix like "NL:" or "NL==".

    Module Name: {module_name}
    Module Interface:
    {module_interface}

    SVA:
    assert property({sva});

    Here are some examples of concise yet informative NL descriptions for SVAs:

    {examples_str}

    Now, provide a concise yet informative NL description for the given SVA in a similar style:
    """

    return prompt.strip()


def add_prompts_to_csv(input_csv: str, output_csv: str):
    """
    Read the sampled SVAs, generate prompts, and create a new CSV with prompts.
    """
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        header = next(reader)
        writer.writerow(header + ["prompt_for_NL_gen"])

        for row in tqdm(reader, desc="Generating prompts"):
            module_name, module_interface, sva = row
            prompt = generate_nl_prompt(module_name, module_interface, sva)
            writer.writerow(row + [prompt])


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_nl_description_with_retry(llm_agent, prompt: str) -> str:
    """
    Generate a natural language description for an SVA using GPT-4-turbo with retry mechanism.
    Automatically refreshes the token if it expires.

    Args:
    llm_agent: The LLM agent to use for generation
    prompt (str): The pre-generated prompt to send to the LLM

    Returns:
    str: Natural language description of the SVA
    """
    prompt_template = PromptTemplate(input_variables=["prompt"], template="{prompt}")

    try:
        llm_chain = LLMChain(prompt=prompt_template, llm=llm_agent)

        with get_openai_callback() as cb:
            result = llm_chain.run(prompt=prompt)
            print(f'Spent a total of {cb.total_tokens} tokens')

        nl_description = result.strip()
        if nl_description.startswith("NL:") or nl_description.startswith("NL=="):
            nl_description = nl_description.split(maxsplit=1)[1].strip()

        return nl_description

    except Exception as e:
        if "token has expired" in str(e):
            print(
                "Token expired. Refreshing token and creating new LLMGatewayChat object..."
            )
            new_llm_agent = refresh_token_and_create_new_agent()
            # Recursively call the function with the new agent
            return get_nl_description_with_retry(new_llm_agent, prompt)
        else:
            print(f"Error generating NL description: {str(e)}")
            raise


def refresh_token_and_create_new_agent():
    """
    Refresh the token and create a new LLMGatewayChat object.
    """
    # Use the prepare_env function to refresh the token
    HOST_UID, SERVER_COOLNAME, ROLE, OPENAI_API_KEY, OPENAI_API_BASE, CERT, su_key = (
        prepare_env(
            VAULT_TYPE="prod",
            VAULT_NAMESPACE="nemo-fastchat",
            SERVER_COOLNAME="llm_gateway",
            ROLE="adlrchat-chipnemo-llmgateway",
            # OVERWRITE=True, # according to Teo, this may cause re-authentication step and just removing this should work
        )
    )

    # Create a new LLMGatewayChat object with the refreshed token
    new_llm_agent = LLMGatewayChat(
        model_name='gpt-4-turbo',
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        openai_organization=None,
        request_timeout=600,
        # temperature=0.1,
        ssl_ca_cert=CERT,
    )

    return new_llm_agent


def library(model_name=None, name=None, **llm_kwargs):
    if model_name is None:
        if name:
            model_name = name
        else:
            model_name = os.environ.get(
                'CHIPNEMO_APP_DEFAULT_MODEL', 'chipnemo_43b_chat'
            )
    library_config = (
        _library[model_name]
        if model_name in _library
        else dict(llm_class=ADLRChat, llm_kwargs=dict(model_name=model_name))
    )
    config = deepcopy(library_config)
    config['llm_kwargs'] = {**config['llm_kwargs'], **llm_kwargs}
    return config


_library = {
    'nemo_43b_chat': dict(
        llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_43b_chat_beta')
    ),
    'chipnemo_43b': dict(
        llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_43b_beta')
    ),
    'chipnemo_43b_chat': dict(
        llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_43b_chat_delta')
    ),
    'nemo_8b_chat': dict(
        llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_8b_chat_alpha')
    ),
    'chipnemo_8b': dict(
        llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_8b_beta')
    ),
    'chipnemo_8b_chat': dict(
        llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_8b_chat_beta')
    ),
    'starcoder': dict(
        llm_class=ADLRCompletion, llm_kwargs=dict(model_name='starcoder')
    ),
    'gpt-4': dict(llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4')),
    'gpt-4-32k': dict(
        llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4-32k')
    ),
    'gpt-35-turbo': dict(
        llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-35-turbo')
    ),
    'gpt-4-turbo': dict(
        llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4-turbo')
    ),
}


def get_llm(*args, **kwargs):
    config = library(*args, **kwargs)
    print(config)
    return config['llm_class'](**config['llm_kwargs'])


def print_final_statistics(final_csv: str):
    """
    Print final statistics for the CSV file.
    """
    with open(final_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        rows = list(csv_reader)

    unique_modules = set(row[0] for row in rows)
    svas_per_module = {}
    for row in rows:
        svas_per_module[row[0]] = svas_per_module.get(row[0], 0) + 1

    print(f"\nFinal Statistics:")
    print(f"Total number of SVAs processed: {len(rows)}")
    print(f"Number of unique modules: {len(unique_modules)}")
    print(f"Maximum SVAs in a single module: {max(svas_per_module.values())}")
    print(f"Minimum SVAs in a module with SVAs: {min(svas_per_module.values())}")
    print(
        f"Average SVAs per module: {sum(svas_per_module.values()) / len(svas_per_module):.2f}"
    )


if __name__ == "__main__":
    random.seed(123)  # Set the seed again just before running main, for extra certainty
    main()

# Below is the prompt used to generate this file.
prompt_ued_to_generate_this_script = '''
We have a folder "/home/scratch.yunshengb_cpu/assertion_data_for_LLM/verified_assertions" with the following structure:
    verified_assertions --> arithmetic_core_2d_fht --> fht_bfly, 
    where "verified_assertions/arithmetic_core_2d_fht/fht_bfly" contains a design (in Verilog) called "fht_bfly.v" and a compressed file "property.sva.tar.gz". Globally, there are many design modules, all in this format/folder structure "verified_assertions/<top_level_name>/<secondary_level_name>/". You need to loop through all of them, and for each one, you need to extract/uncompress the "property.sva.tar.gz" file so you get a file called "property.sva". Write python code (with main function and helper functions each with proper documentation/annotation for readability) to loop through all design modules in the "verified_assertions" folder, and for each design module, uncompress the file and put the "property.sva" in the same folder. 
    
    Then generate a csv file containing the following information per row: "module_name", "module_interface", "SVA", "NL":
    
        (1) For the "module_name", it should be "arithmetic_core_2d_fht/fht_bfly". For another module, the name could be different such as  "communication_controller_can_protocol_controller/can_acf".

        (2) For the "module_interface", the interface should be extracted by parsing "fht_bfly.v" (or whatever the secondary module name is, e.g. for "can_acf", try to parse "can_acf.v") and extracting the interface such as """module my_module (
                                input clk,
                                input reset,
                                output data_out
                                );

                                // module implementation omitted

                                endmodule""".
        You can assume the verilog file such as "fht_bfly.v" is well-written and you can just parse the file to get the module interface. Here is a concrete example "counter.v":
        """
            //////////////////////////////////////////////////////////////////////
            ////                                                              ////
            //// MODULE NAME: counter                                         ////
            ////                                                              ////
            //// DESCRIPTION: 8bit counter                                    ////
            ////                                                              ////
            ////                                                              ////
            //// This file is part of the 10 Gigabit Ethernet IP core project ////
            ////  http://www.opencores.org/projects/ethmac10g/                ////
            ////                                                              ////
            //// AUTHOR(S):                                                   ////
            //// Zheng Cao                                                    ////
            ////                                                              ////
            //////////////////////////////////////////////////////////////////////
            ////                                                              ////
            //// Copyright (c) 2005 AUTHORS.  All rights reserved.            ////
            ////                                                              ////
            //// This source file may be used and distributed without         ////
            //// restriction provided that this copyright statement is not    ////
            //// removed from the file and that any derivative work contains  ////
            //// the original copyright notice and the associated disclaimer. ////
            ////                                                              ////
            //// This source file is free software; you can redistribute it   ////
            //// and/or modify it under the terms of the GNU Lesser General   ////
            //// Public License as published by the Free Software Foundation; ////
            //// either version 2.1 of the License, or (at your option) any   ////
            //// later version.                                               ////
            ////                                                              ////
            //// This source is distributed in the hope that it will be       ////
            //// useful, but WITHOUT ANY WARRANTY; without even the implied   ////
            //// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ////
            //// PURPOSE.  See the GNU Lesser General Public License for more ////
            //// details.                                                     ////
            ////                                                              ////
            //// You should have received a copy of the GNU Lesser General    ////
            //// Public License along with this source; if not, download it   ////
            //// from http://www.opencores.org/lgpl.shtml                     ////
            ////                                                              ////
            //////////////////////////////////////////////////////////////////////
            //
            // CVS REVISION HISTORY:
            //
            // $Log: not supported by cvs2svn $
            // Revision 1.2  2006/06/06 05:02:11  Zheng Cao
            // no message
            //
            // Revision 1.1  2005/12/25 16:43:10  Zheng Cao
            // 
            // 
            //
            //////////////////////////////////////////////////////////////////////
            //
            `timescale 100ps / 10ps
            module counter(clk, reset, load, en, value);
                input clk;
                input reset;
                input load;
                input en;

                parameter WIDTH = 8;
                output[WIDTH-1:0] value;

                reg [WIDTH-1:0] value;
            
                always @(posedge clk or posedge reset)
                if (reset)	 
                    value <= 0;
                else begin
                if (load) 
                        value <= 0;
                    else if (en)
                        value <= value + 1;
                end

            endmodule

        """.

        (3) The "SVA" should be extracted from the "property.sva" file: """module i_apb(
			input PRESETn,
            );

            assert property(@(posedge PCLK) ((PRDATA >= 201563416) && (PRDATA <= 4288916223)) |-> (PRDATA >= 201563416) && (PRDATA <= 4288916223));
            assert property(@(posedge PCLK) ((READ_DATA_ON_RX >= 201563416) && (READ_DATA_ON_RX <= 4288916223)) |-> (PRDATA >= 201563416) && (PRDATA <= 4288916223));
            ...""". Extract all the SVAs in this file. Suppose there are N SVAs for this module. Then there should 100 rows in the final csv file, i.e. one SVA per row. 

        (4) The "NL" should be generated per SVA describing in natrual language what the SVA means. We are a research team trying to study whether LLMs/agents can generate SVAs from NL descriptions of SVAs, so we need to create a benchmark dataset consisting of such (module_name, module_interface, SVA, NL) pairs. The NL should be generated by "GPT-4-turbo". Specifically, write Python code to make such LLM calls. Use the following code as an example to see how to make such call:
        """
            import os
            from copy import deepcopy
            from adlrchat import ADLRChat, ADLRCompletion, LLMGatewayChat, LLMGatewayCompletion
            default_llms = ['chipnemo_8b','nemo_8b']
            def library(model_name=None, name=None, **llm_kwargs):
                if model_name is None:
                    if name:
                        model_name = name
                    else:
                        model_name = os.environ.get('CHIPNEMO_APP_DEFAULT_MODEL', 'chipnemo_43b_chat')
                library_config = _library[model_name] if model_name in _library else dict( llm_class=ADLRChat, llm_kwargs=dict(model_name=model_name))
                config = deepcopy(library_config)
                config['llm_kwargs'] = {**config['llm_kwargs'], **llm_kwargs}
                return config
            _library = {
                'nemo_43b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_43b_chat_beta')),
                'chipnemo_43b' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_43b_beta')),
                'chipnemo_43b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_43b_chat_delta')),
                'nemo_8b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_8b_chat_alpha')),
                'chipnemo_8b' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_8b_beta')),
                'chipnemo_8b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_8b_chat_beta')),
                'starcoder' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='starcoder')),
                'gpt-4' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4')),
                'gpt-4-32k' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4-32k')),
                'gpt-35-turbo' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-35-turbo'))
            }
            def get_llm(*args, **kwargs):
                config = library(*args, **kwargs)
                print(config)
                return config['llm_class'](**config['llm_kwargs'])
            import sys
            llm_agent = get_llm(model_name='gpt-4', temperature=0.1)
            from langchain.agents import initialize_agent, AgentType
            from langchain.callbacks import get_openai_callback
            from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
            from langchain import PromptTemplate
            # Count the langchain token
            def count_tokens(chain, query):
                with get_openai_callback() as cb:
                    result = chain.run(query)
                    print(cb)
                    print(f'Spent a total of {cb.total_tokens} tokens')
                    # if cb.total_tokens > 14000:
                    #    exit(1)
                return result

            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm_agent)
            result = count_tokens(llm_chain, query="Hi")
            print('\n\n=== AI: response === \n', result)
        """.
        Definitely modify the code above to make the proper call. Specifcally, write some high-quality detailed clear prompt asking GPT-4-turbo to generate NL description for each SVA. It is like given the SVA and the module name and interface, generate a high-quality clear NL description or something like that. Prepare a good prompt and write it in the code you generate. And send that prompt to GPT-4-turbo using the example code above.

    Finally, print some stats for this big csv file and store to disk.

'''
