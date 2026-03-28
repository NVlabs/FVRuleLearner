import subprocess
import time
import datetime
import os
from typing import List, Tuple
import sys
import socket
import psutil
import paramiko  # Add this import
import re, pytz

MACHINES_AVAILABLE = []
DATASETS = [['nl2sva_human']]
# DATASETS = [['nl2sva_machine']]
MODELS = ["gpt-4-turbo","gpt-4"]
# MODELS = ["mixtral_8x7b"]
NUM_GROUP = 4
GROUP_IDS = [0,1]
# GROUP_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
RUN_COMMAND = 'cd /home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src && conda activate agent && python main.py'

DELETE_ALL = False
# DELETE_ALL = True


def get_current_machine_info() -> Tuple[int, int]:
    """Get the total and available CPU cores for the current machine."""
    total_cores = psutil.cpu_count(logical=True)
    available_cores = psutil.cpu_count(logical=True) - psutil.getloadavg()[0]
    return total_cores, int(available_cores)


def print_machine_info(machines: List[str]):
    """Print a table of machine information."""
    print("Machine Information:")
    print("IP Address\tTotal Cores\tAvailable Cores")
    print("-" * 50)
    for machine in machines:
        total, available = get_current_machine_info()
        print(f"{machine}\t{total}\t\t{available}")


import subprocess
import shlex


def run_tmux_command(command: List[str]) -> Tuple[bool, str, str]:
    """Run a tmux command and return success status, stdout, and stderr."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def create_tmux_session(session_name: str) -> Tuple[bool, str]:
    """Create a new tmux session."""
    create_command = ['tmux', 'new-session', '-d', '-s', session_name]
    success, stdout, stderr = run_tmux_command(create_command)
    if not success:
        return False, f"Failed to create session. Stdout: {stdout}, Stderr: {stderr}"

    time.sleep(1)  # Wait a bit to ensure the session is fully created
    return True, "Session created successfully"


def sanitize_window_name(name: str) -> str:
    """Sanitize the window name by replacing dots with underscores."""
    return name.replace('.', '_')


def create_tmux_window(session_name: str, window_name: str) -> Tuple[bool, str, str]:
    """Create a new tmux window within a session."""
    sanitized_window_name = sanitize_window_name(window_name)
    success, stdout, stderr = run_tmux_command(['tmux', 'new-window', '-t', session_name, '-n', sanitized_window_name])
    if not success:
        print(f"Failed to create window {sanitized_window_name}. Stdout: {stdout}, Stderr: {stderr}")
    time.sleep(0.5)  # Add a small delay between window creations
    return success, stdout, stderr


def get_src_path():
    """Get the source path of the project."""
    return os.path.dirname(os.path.abspath(__file__))


def get_current_ts(zone='US/Pacific'):
    """Get current timestamp in the specified timezone."""
    return datetime.datetime.now(pytz.timezone(zone)).strftime('%Y-%m-%dT%H-%M-%S.%f')


def get_short_ts():
    """Get a short timestamp for tmux session names."""
    return datetime.datetime.now().strftime('%m%d_%H%M%S')


def create_logdir(dataset, model, group_id, machine_name, username):
    """Create a detailed logdir with current timestamp."""
    timestamp = get_current_ts()
    logdir = os.path.join(get_src_path(), 'logs', f'{dataset}_{model}_{group_id}_{timestamp}_{machine_name}_{username}')
    return logdir


def run_experiment(
    session_name: str, window_name: str, dataset: str, model: str, group_id: int, machine_name: str, username: str
) -> Tuple[bool, str]:
    """Run the experiment in a tmux window."""
    logdir = create_logdir(dataset, model, group_id, machine_name, username)
    command = f"{RUN_COMMAND} --src_examples='{dataset}' --llm_model={model} --group_id={group_id} --logdir={logdir}"
    sanitized_window_name = sanitize_window_name(window_name)
    success, stdout, stderr = run_tmux_command(
        ['tmux', 'send-keys', '-t', f'{session_name}:{sanitized_window_name}', command, 'C-m']
    )
    if not success:
        print(f"Failed to send command to window {sanitized_window_name}. Stdout: {stdout}, Stderr: {stderr}")
        return False, ""
    return True, logdir


def get_ssh_client(ip: str) -> paramiko.SSHClient:
    """Create and return an SSH client for the given IP address."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=os.getenv('USER'))
    return client


def kill_remote_tmux_sessions(ip: str, current_session: str):
    """Kill all tmux sessions on a remote machine except the current one."""
    try:
        client = get_ssh_client(ip)
        stdin, stdout, stderr = client.exec_command(
            f"tmux list-sessions -F '#{{session_name}}' | grep -v '{current_session}' | xargs -I{{}} tmux kill-session -t {{}}"
        )
        print(f"Killed tmux sessions on {ip}: {stdout.read().decode().strip()}")
        client.close()
    except Exception as e:
        print(f"Error killing tmux sessions on {ip}: {str(e)}")


def is_timestamp_session(session_name: str) -> bool:
    """Check if the session name matches our timestamp format."""
    pattern = r'\d{4}_\d{6}'  # Matches format like '0722_153312'
    return bool(re.match(pattern, session_name))


def kill_local_tmux_sessions(current_session: str):
    """Kill tmux sessions that match our timestamp format, excluding the current one."""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True, check=True
        )
        sessions = result.stdout.strip().split('\n')
        current_script_session = os.environ.get('TMUX', '').split(',')[0].split('/')[-1]

        killed_sessions = []
        for session in sessions:
            if is_timestamp_session(session) and session != current_session and session != current_script_session:
                success, stdout, stderr = run_tmux_command(['tmux', 'kill-session', '-t', session])
                if success:
                    killed_sessions.append(session)
                else:
                    print(f"Failed to kill session {session}. Stderr: {stderr}")

        if killed_sessions:
            print(f"Killed tmux sessions: {', '.join(killed_sessions)}")
        else:
            print("No matching tmux sessions to kill.")
    except subprocess.CalledProcessError:
        print("Failed to list tmux sessions.")
    except Exception as e:
        print(f"Error killing tmux sessions: {str(e)}")


def main():
    session_name = get_short_ts()
    current_machine = socket.gethostname()
    machines = MACHINES_AVAILABLE + [current_machine]
    username = os.getenv('USER')

    try:
        if DELETE_ALL:
            print("Cleaning up tmux sessions...")
            for machine in MACHINES_AVAILABLE:
                kill_remote_tmux_sessions(machine, session_name)
            kill_local_tmux_sessions(session_name)

            # List remaining sessions after cleanup
            success, stdout, stderr = run_tmux_command(['tmux', 'list-sessions'])
            if success:
                print(f"Remaining tmux sessions after cleanup:\n{stdout}")
            else:
                print(f"Failed to list tmux sessions after cleanup. Stderr: {stderr}")

            return

        print_machine_info(machines)

        print(f"Creating tmux session: {session_name}")
        success, message = create_tmux_session(session_name)
        if not success:
            raise Exception(f"Failed to create tmux session {session_name}. Error: {message}")

        print("Tmux session created successfully. Proceeding with experiments.")

        # List all sessions to verify
        success, stdout, stderr = run_tmux_command(['tmux', 'list-sessions'])
        if success:
            print(f"Current tmux sessions:\n{stdout}")
        else:
            print(f"Failed to list tmux sessions. Stderr: {stderr}")

        experiments = []
        for dataset in DATASETS:
            for model in MODELS:
                for group_id in GROUP_IDS:
                    window_name = f"{dataset[0]}_{model}_{group_id}"
                    success, stdout, stderr = create_tmux_window(session_name, window_name)
                    if not success:
                        print(
                            f"Failed to create window {window_name}, skipping this experiment. Stdout: {stdout}, Stderr: {stderr}"
                        )
                        continue

                    success, logdir = run_experiment(
                        session_name, window_name, dataset[0], model, group_id, current_machine, username
                    )
                    if not success:
                        print(
                            f"Failed to run experiment in window {window_name}, skipping. Command may not have been sent properly."
                        )
                        continue

                    print(f"Successfully started experiment in window {window_name}")
                    experiments.append((f"{dataset[0]}, {model}, {group_id}", logdir))

        print("\nExperiment Summary:")
        print("exp_name\tlogdir")
        print("-" * 50)
        for exp_name, logdir in experiments:
            logdir_basename = os.path.basename(logdir)
            print(f"{exp_name:<30}\t{logdir_basename}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        run_tmux_command(['tmux', 'kill-session', '-t', session_name])


if __name__ == "__main__":
    main()
prompt = '''
Write a python script that does the following:

(1) Define parameters we can tune:

MACHINES_AVAILABLE = ['<ip_address_1>', '<ip_address_12', ...]
DATASETS = ['human', 'machine']
MODELS = ['gpt-3.5', 'llama3']
NUM_GROUP = 10
GROUP_IDS = [0,1,2,3]
RUN_COMMAND = 'cd /home/scratch.yunshengb_cpu/fv/fv_testplan/src && conda activate fv && python main.py'

DELETE_ALL = False # dangerous; this will tmux kill-session all experiments on MACHINES_AVAILABLE!

(2) Loop through DATASETS, and then loop through MODELS, and then loop through GROUP_IDS. For each (dataset, model, group_id) combination (i.e. an experiment), find the best machine to run it and send a tmux command to trigger the run. Next, I will provide some description on how to do that.

(3) Find the best suitable machine/server: Somehow ssh into these machines (MACHINES_AVAILABLE plus the current machine which you don't need ssh into of course, so even if MACHINES_AVAILABLE=[], you can still do the rest of steps on the current machine), and provide a summary of how many cores/CPU threads are there and how many are empty (that we can use) at the moment. Print the results nicely in a table. Tip: You may need to ssh into it using the ssh public key stored on the current machine (and the current server is a linux server).

(4) Based on the information you collected in the previous step, for each experiment, allocate one machine for it to run. To actually run it, you need to ssh into that designated machine, and send a tmux command creating a new session (or reusing the session if it is already created) named as "<timestamp>" which is something like "2024-07-17T10-53-54.564138" (time in PDT time zone). This means for all the experiments that you deploy in this script, they all share the same session name across experiments and across machines. This allows us to manually ssh into these machines and check the status later.

(5) Once you create or enter into the tmux session, for each experiment, create a new tmux window. This means if multiple (e.g. 10 ) experiments are allocated to one machine, that machine should have one session named as "<timestamp>" and within that tmux session there should be multiple (i.e. 10) windows (one per experiment).

(6) At this point the experiment (dataset, model, group_id) should have its dedicated window. Now run the program by executing f'{RUN_COMMAND} --dataset={dataset} --model={model} --group_id={group_id} --logdir={timestramp}_{machine_name}_{username}', i.e. appending the command line arguments to the `python main.py`. This will trigger the main python program. Notice the machine_name and user_name part in the logdir. These allow us to manually check who deployed these experiments and on which machines. 

(7) After deploying all the experiments, print a nice summary in a copy-and-paste-able format:
"""
exp_name                   logdir
<dataset>, <model>, <group_id>)                   <logdir>
"""
For example, for the parameters above, you should deploy 2*2*3=12 (#datasets * #models * $groups) experiments in total and 12 rows in this table. We will copy and paste this table into Google Sheet for our record and further analysis.

(8) Also write a function that kills all tmux sessions on MACHINES_AVAILABLE (except for the current tmux session which you do NOT want to kill) and execute it if DELETE_ALL=True. If DELETE_ALL=False, don't do this step and instead just do steps (1)-(7) above. The purpose of this function/step is to ensure we can run this script to clean all experiments we have deployed.


At any step, if there is an error, you should catch the error and print it and then break the whole program. For example, if RUN_COMMAND is set to something like "cdwfjwf" which is not executable (i.e. an error will be returned by bash), you should print an informative error message for us to fix. Also clean up/delete the created tmux sessions corresponding to "<timestamp>".

The whole script should have a main function and a lot of helper functions with documentation/annotations for readability. Also be sure to print intermediate steo messages for debugging this script and letting us see progress.
'''
