########################################################
# NVIDIA License
#
# 1. Definitions
#
# “Licensor” means any person or entity that distributes its Work.
# “Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
# Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.
#
# 2. License Grant
#
# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.
#
# 3. Limitations
#
# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.
#
# 3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.
#
# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.
#
# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.
#
# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.
#
# 3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.
#
# 4. Disclaimer of Warranty.
#
# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 
#
# 5. Limitation of Liability.
#
# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
########################################################
import errno
import gzip
import json
import os
import re
import numpy as np
from typing import Iterable, Dict, List
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_error(header: str, body: str):
    print(f"{bcolors.FAIL}{header}:{bcolors.ENDC}\n{body}")


def print_lm_response(header: str, body: str):
    print(
        f"{bcolors.OKCYAN}{header}:{bcolors.ENDC}\n{bcolors.BOLD}{body}{bcolors.ENDC}"
    )


def print_user_prompt(header: str, body: str):
    body = body.split("Question:")[-1]
    body = "Question:" + body
    print(
        f"{bcolors.OKBLUE}{header}:{bcolors.ENDC}\n{bcolors.BOLD}{body}{bcolors.ENDC}"
    )


def mkdir_p(path):
    """mkdir -p in python
    Args:
        path: directory path
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    Skipping None in data
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if x:
                        gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if x:
                    fp.write((json.dumps(x) + "\n").encode("utf-8"))


# def parse_code_response(lm_response_str) -> str:
#     code_tags = re.findall(r"```systemverilog(.*?)```", lm_response_str, re.DOTALL)
#     if len(code_tags) > 0:
#         # remove all substrings before and after code_tag
#         for code in code_tags:
#             lm_response_str = lm_response_str.replace(f"```systemverilog{code}```", code)
#     code_tags = re.findall(r"```systemverilog(.*?)", lm_response_str, re.DOTALL)
#     if len(code_tags) > 0:
#         for code in code_tags:
#             lm_response_str = lm_response_str.replace(f"```systemverilog{code}", code)
#     return lm_response_str


def parse_code_response(lm_response_str) -> str:
    if "```systemverilog" in lm_response_str:
        lm_response_str = lm_response_str.split("```systemverilog")[-1]
    if "```" in lm_response_str:
        lm_response_str = lm_response_str.split("```")[0]
    return lm_response_str.strip()

def parse_code_response_verilog(lm_response_str) -> str:
    if "```verilog" in lm_response_str:
        lm_response_str = lm_response_str.split("```verilog")[-1]
    if "```" in lm_response_str:
        lm_response_str = lm_response_str.split("```")[0]
    return lm_response_str.strip()

def parse_code_response_json(lm_response_str) -> str:
    if "```json" in lm_response_str:
        lm_response_str = lm_response_str.split("```json")[-1]
    if "```" in lm_response_str:
        lm_response_str = lm_response_str.split("```")[0]
    return lm_response_str.strip()

def parse_code_response_multi(lm_response_str: str) -> list[str]:
    code_blocks = []
    parts = lm_response_str.split("```")
    for i in range(1, len(parts), 2):
        if parts[i].strip().startswith("systemverilog"):
            code_blocks.append(parts[i].strip()[13:].strip())
        elif not parts[i].strip().startswith("```"):
            code_blocks.append(parts[i].strip())
    return code_blocks

def format_systemverilog_response(response_str: str) -> str:
    """
    Format a string as a SystemVerilog code block by adding
    the appropriate markdown code fence markers.

    Args:
    response_str (str): The input string containing SystemVerilog code.

    Returns:
    str: The formatted string with SystemVerilog code block markers.
    """
    # Strip any leading or trailing whitespace
    formatted_response = response_str.strip()
    
    # Add the SystemVerilog code block markers
    formatted_response = f"```systemverilog\n{formatted_response}\n```"
    
    return formatted_response

def pass_at_k(x, n, k):
    pass_at_k_values = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        c = x[i]
        if n - c < k:
            pass_at_k_values[i] = 1.0
        else:
            range_values = np.arange(n - c + 1, n + 1)
            pass_at_k_values[i] = 1.0 - np.prod(1.0 - k / range_values)

    return pass_at_k_values


# def include_system_message(message):
#     """Custom transform method to include the system message in the output."""
#     if 'system_message' in message:
#         message['content'] = (
#             f"System: {message['system_message']}\n\n{message['content']}"
#         )
#     return message


# def extract_code_snippet(text):
#     # Define the regular expression pattern to match content between <CODE> and </CODE> inclusive
#     pattern = re.compile(r'(<CODE>.*?</CODE>)', re.DOTALL)

#     # Search for the pattern in the provided text
#     match = pattern.search(text)

#     # Return the matched content or None if not found
#     if match:
#         return match.group(1)
#     else:
#         return '<CODE>\n' + text + '</CODE>'

def extract_assertion_formula(assertion_text):
    # Remove newlines and extra spaces
    assertion_text = assertion_text.replace("\n", " ").strip()
    
    # # Pattern for assert property(...);
    property_pattern = r'assert\s+property\s*\(\s*@\s*\(\s*(?:posedge|negedge)\s+\w+\s*\)\s*(.*?)\s*\)\s*;'
    
    # # Pattern for assert(@...);
    direct_pattern = r'assert\s*\(\s*@\s*\(\s*(?:posedge|negedge)\s+\w+\s*\)\s*(.*?)\s*\)\s*;'
    
    # Updated pattern for assert property(...);
    # property_pattern = r'assert\s+property\s*@\s*\(\s*(?:posedge|negedge)\s+\w+\s*\)\s*\((.*?)\)\s*;'
    
    # Updated pattern for assert(@...);
    # direct_pattern = r'assert\s*@\s*\(\s*(?:posedge|negedge)\s+\w+\s*\)\s*\((.*?)\)\s*;'
    
    # Try to match assert property pattern
    match = re.search(property_pattern, assertion_text)
    
    if not match:
        # If not found, try to match direct assert pattern
        match = re.search(direct_pattern, assertion_text)
    
    if match:
        return match.group(1).strip()
    else:
        return assertion_text

def extract_assertion_statements(assertion_text: str) -> list[str]:
    # Remove newlines and extra spaces for cleaner processing
    assertion_text = assertion_text.replace("\n", " ").strip()
    
    # Pattern for assert property(...);
    property_pattern = r'assert\s+property\s*\(\s*@\s*\(\s*(?:posedge|negedge)\s+\w+\s*\)\s*.*?\s*\)\s*;'
    
    # Pattern for assert(@...);
    direct_pattern = r'assert\s*\(\s*@\s*\(\s*(?:posedge|negedge)\s+\w+\s*\)\s*.*?\s*\)\s*;'
    
    # Find all matches for both patterns
    property_matches = re.findall(property_pattern, assertion_text)
    direct_matches = re.findall(direct_pattern, assertion_text)
    
    # Combine all matches into a single list, retaining the full assertion statement
    all_assertions = [match.strip() for match in property_matches + direct_matches]
    
    # Return the list of complete assertion statements
    return all_assertions

def safe_rmtree(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))
    shutil.rmtree(dir_path)

def extract_signals_from_module(verilog_code, task_id: str = None):
    # Regular expression to match signals inside the module interface (not internal signals or parameters)
    if task_id == "module_i2c":
        return ["PCLK", "PRESETn", "fifo_tx_f_full", "fifo_tx_f_empty", "fifo_tx_data_out", "fifo_rx_f_full", "fifo_rx_f_empty", "fifo_rx_wr_en", "fifo_rx_data_in", "DATA_CONFIG_REG", "TIMEOUT_TX", "fifo_tx_rd_en", "TX_EMPTY", "RX_EMPTY", "ERROR", "ENABLE_SDA", "ENABLE_SCL", "SDA", "SCL"]
    elif task_id == "apb":
        return ["PRESETn", "READ_DATA_ON_RX", "ERROR", "PCLK", "INTERNAL_I2C_REGISTER_TIMEOUT", "PWDATA", "INT_TX", "WRITE_DATA_ON_TX", "WR_ENA", "PRDATA", "RD_ENA", "INTERNAL_I2C_REGISTER_CONFIG", "RX_EMPTY", "PREADY", "PADDR", "PWRITE", "PENABLE", "INT_RX", "PSELx", "TX_EMPTY", "PSLVERR"]
    elif task_id == "counter":
        return ["en", "clk", "reset", "value", "load"]
    elif task_id == "rxLenTypChecker":
        return ["get_terminator", "length_error", "length_65_127", "length_128_255", "length_256_511", "frame_cnt", "small_error", "vlan_enable", "jumbo_frame", "length_512_1023", "rxclk", "tagged_frame", "length_1024_max", "location_input", "padded_frame", "terminator_location", "reset", "large_error", "location_reg", "jumbo_enable"]
    elif task_id == "rxStatModule":
        return ["rxStatRegPlus_tmp", "jumbo_frame", "multi_valid", "padded_frame", "get_error_code", "crc_check_invalid", "broad_valid", "receiving", "length_256_511", "length_512_1023", "good_frame_get", "large_error", "rxclk", "reset", "length_65_127", "length_128_255", "pause_frame", "rxStatRegPlus", "length_1024_max", "small_error"]
    elif task_id == "rxLinkFaultState":
        return ["get_one_fault", "last_seq_type", "linkstate", "linkstate_next", "seq_cnt", "reset_col_cnt", "seq_cnt_3", "local_fault", "fault_type", "remote_fault", "col_cnt_64", "rxclk", "seq_type", "no_new_type", "reset", "col_cnt", "link_fault"]
    elif task_id == "ack_counter":
        return ["ready", "tx_ack", "counter", "tx_start", "start_count", "max_count", "clock", "reset", "start_count_del"]
    elif task_id == "eth_transmitcontrol":
        return ["write_fifo", "TxFlow", "TxCtrlEndFrm", "ControlData", "ControlEnd_q", "TxDoneIn", "ResetByteCnt", "clear", "WillSendControlFrame", "TxEndFrmOut_uc", "TxUsedDataIn_q", "TxCtrlStartFrm", "TxReset", "TxData_wrapped_out_input", "TxBufferEmpty", "CtrlMux", "MAC", "EnableCnt", "TxBufferAlmostFull", "TxUsedDataIn", "StateCount", "StateLeftinQ", "TxBufferFull", "PreNib15State", "TxData_wrapped_out", "MTxClk", "TxUsedDataOutDetected", "DlyCrcCnt", "SendingCtrlFrm", "TxEndFrmIn", "TPauseRq", "IncrementByteCntBy2", "TxDataIn", "IncrementDlyCrcCnt", "TxStartFrmIn", "MuxedCtrlData", "ByteCnt", "IncrementByteCnt", "read_fifo", "BlockTxDone", "txfifo_cnt", "Divided_2_MTxClk", "TxBufferAlmostEmpty", "TxAbortIn", "TxUsedDataOut", "ControlEnd", "DlyCrcEn", "TxCtrlStartFrm_q", "TxPauseTV", "DMAC"]
    elif task_id == "eth_rxethmac":
        return ["Enable_Crc", "Data_Crc", "MRxClk", "AddressMiss", "StateSFD", "MRxDEq5", "Crc", "CrcHash", "GenerateRxStartFrm", "r_HASH0", "DelayData", "GenerateRxEndFrm", "RxData_d", "StartCalc", "ByteCntEq5", "ByteCntEq1", "ByteCntEq6", "ByteCntSmall7", "GenerateRxValid", "StateDrop", "RxValid", "CrcError", "r_HASH1", "ByteCntMaxFrame", "MaxFL", "StateData", "r_Pro", "prev_latched_Rx1", "Reset", "LatchedByte", "ByteCntEq3", "Divided_2_MRxClk", "StatePreamble", "RxValid_d", "PassAll", "CrcNext", "MRxD", "ControlFrmAddressOK", "Multicast", "RxData", "DribbleRxEndFrm", "RxAbort", "CSready", "RxStartFrm_d", "ByteCntEq2", "ByteCntEq4", "MRxDEqD", "MRxDV", "Initialize_Crc", "ByteCnt", "RxEndFrm", "DlyCrcCnt", "prev_latched_Rx", "Transmitting", "IFGCounterEq24", "Broadcast", "CrcHashGood", "ByteCntEq7", "MAC", "r_IFG", "DlyCrcEn", "CheckSum", "RxEndFrm_d", "Sum", "ByteCntGreat2", "Divided_4_MRxClk", "HugEn", "StateIdle", "ByteCntEq0", "ByteCntDelayed", "ByteCntOut", "r_Bro", "Enable", "Initialize", "Data"]
    elif task_id == "MAC_rx_FF":
        return ["Add_rd_pl1", "Add_wr_jump_tmp", "Fifo_data", "Add_wr_pluse4", "Add_wr_gray", "Packet_number_add", "Fifo_data_en", "Rx_mac_ra", "Add_wr_reg", "Empty", "Current_state_SYS", "Add_wr_jump_tmp_pl1", "Fifo_data_byte1", "Dout", "Add_wr_jump_rd_pl1", "Rx_Lwmark_pl", "Addr_freshed_ptr", "Din", "RX_APPEND_CRC", "Packet_number_add_dl2", "Clk_SYS", "Add_rd", "Next_state_SYS", "Next_state", "Packet_number_add_edge", "Packet_number_sub", "Fifo_data_dl1", "Fifo_data_byte2", "Add_wr_pluse3", "Fifo_data_err", "Rx_mac_sop_tmp", "Fifo_full", "Rx_mac_data", "Wr_en", "Packet_number_add_tmp_dl1", "Almost_full", "Din_tmp", "Wr_en_tmp", "Fifo_data_count", "Fifo_data_end", "Rx_Hwmark", "Add_rd_ungray", "Add_wr", "Rx_mac_BE", "Add_rd_gray", "Fifo_data_byte0", "Add_rd_gray_dl1", "Rx_mac_pa", "Add_wr_pluse", "Add_wr_ungray", "Wr_en_ptr", "Reset", "Full", "Clk_MAC", "Add_wr_gray_dl1", "Packet_number_inFF", "Add_wr_pluse2", "Fifo_data_en_dl1", "Packet_number_add_tmp_dl2", "Fifo_data_byte3", "Rx_mac_rd", "Add_wr_jump", "Rx_mac_sop_tmp_dl1", "Dout_dl1", "Packet_number_add_dl1", "Packet_number_add_tmp", "Rx_Hwmark_pl", "Current_state", "Rx_mac_pa_tmp", "Rx_mac_sop", "Rx_Lwmark", "Rx_mac_eop", "Din_tmp_reg", "i", "Din_tmp_input", "Add_wr_input"]
    elif task_id == "eth_cop":
        return ["s1_wb_ack_i", "s1_wb_err_i", "m1_in_progress", "m1_addressed_s2", "m1_addressed_s1", "m1_wb_cyc_i", "m1_wb_stb_i", "m1_wb_we_i", "s1_wb_sel_o", "m_wb_access_finished", "m2_req", "m1_wb_adr_i", "m1_wb_dat_i", "m2_wb_cyc_i", "m2_wb_stb_i", "m2_wb_we_i", "m1_wb_dat_o", "m2_addressed_s1", "s1_wb_we_o", "s1_wb_cyc_o", "s1_wb_stb_o", "m2_wb_adr_i", "m2_wb_dat_i", "s2_wb_adr_o", "s2_wb_dat_o", "s1_wb_dat_i", "s2_wb_we_o", "s2_wb_cyc_o", "s2_wb_stb_o", "wb_clk_i", "wb_rst_i", "s2_wb_ack_i", "s2_wb_err_i", "m2_wb_dat_o", "s2_wb_sel_o", "m2_addressed_s2", "m1_req", "m1_wb_sel_i", "m2_wb_err_o", "m2_wb_ack_o", "s2_wb_dat_i", "m1_wb_ack_o", "m1_wb_err_o", "m2_in_progress", "s1_wb_adr_o", "m2_wb_sel_i", "s1_wb_dat_o"]
    elif task_id == "eth_macstatus":
        return ["LatchedCrcError", "LoadRxStatus", "DeferIndication", "RxColWindow", "r_FullD", "InvalidSymbol", "r_HugEn", "LateCollision", "RxCrcError", "RxStateIdle", "TxStartFrm", "r_MinFL", "ReceivedLengthOK", "RxStateSFD", "ReceiveEnd", "TakeSample", "Collision", "CollValid", "ReceivedPacketGood", "StatePreamble", "RetryLimit", "r_RecSmall", "StartTxDone", "RetryCntLatched", "RxByteCnt", "Loopback", "DeferLatched", "DribbleNibble", "LateCollLatched", "CarrierSenseLost", "MTxClk", "TxUsedData", "LatchedMRxErr", "MRxClk", "RxByteCntMaxFrame", "MaxCollisionOccured", "CarrierSense", "ReceivedPacketTooBig", "ShortFrame", "MRxD", "RxByteCntEq0", "MRxDV", "RxLateCollision", "StartTxAbort", "RstDeferLatched", "r_MaxFL", "RxByteCntGreat2", "RxStateData", "RetryCnt", "MRxErr", "SetInvalidSymbol", "StateData", "Reset", "RxStatePreamble", "Transmitting"]
    elif task_id == "ge_1000baseX_rx":
        return ["ebi_rxd_d2", "ebi_rxd_d1", "receiving", "ability_match", "xmit", "early_end_cnt", "idle_cnt", "idle_cnt_m_inc", "idle_cnt_m_clr", "xmit_DATA", "xmit_nDATA", "xmit_DATA_CD", "xmit_DATA_nCD", "rx_frame_pulse", "check_end_R_R_K28_5_cnt", "CE_match", "SPD_match", "EPD_match", "early_end", "xmit_DATA_CD_SPD", "xmit_DATA_CD_nSPD", "xmit_DATA_CD_nSPD_nK28_5", "rudi", "rx_config", "rx_data_cnt", "ebi_rxd_d1_X", "ebi_rxd_d1_Y", "check_end_R_R_R", "gmii_rxd_false_carrier_m_set", "gmii_rxd_preamble_m_set", "gmii_rxd_ext_err_m_set", "gmii_rx_er_m_set", "gmii_rx_er_m_clr", "soft_reset", "check_end_R_R_S", "rx_config_tmp", "sync_status", "ge_x_pcs_rx_stats_inc", "gmii_rx_dv_m_set", "gmii_rx_dv_m_clr", "idle_match", "ebi_K_d1", "ebi_K_d2", "ebi_K_d3", "ebi_K", "check_end_R_R_R_cnt", "gmii_rxd_packet_burst_m_set", "gmii_rxd_trr_extend_m_set", "gmii_rxd_m_set", "pcs_rx_present", "pcs_rx_next", "acknowledge_match", "check_end_T_R_R", "early_end_idle", "ability", "ability_d1", "ability_d2", "check_end_R_R_S_cnt", "ebi_rxd_d2_X", "ebi_rxd_d2_Y", "rudi_INVALID_m_set", "rudi_IDLE_m_set", "rudi_CONF_m_set", "mr_main_reset", "early_end_config", "ability_matched_input", "rx_config_lo_read", "rx_config_hi_read", "check_end_T_R_K28_5_cnt", "K28_5_match", "D2_2_match", "D21_5_match", "D5_6_match", "D16_2_match", "gmii_rx_er", "rx_config_cnt", "sync_status_d1", "sync_status_d2", "sync_status_d3", "ebi_rxd_d3_X", "ebi_rxd_d3_Y", "carrier_detect", "rx_config_set", "gmii_rx_dv", "gmii_rxd", "check_end_R_R_K28_5", "check_end_T_R_K28_5", "signal_detect", "rx_config_cnt_m_inc", "rx_config_cnt_m_rst", "rx_config_cnt_done", "carrier_detect_d1", "carrier_detect_d2", "carrier_detect_d3", "check_end_T_R_R_cnt", "consistency_match", "rx_frame_cnt", "ebi_rxd_d3", "receiving_m_set", "receiving_m_clr", "rx_even_d1", "rx_even_d2", "rx_even_d3", "ebi_rxd", "rx_even", "rx_config_d1", "rx_config_d2", "rx_config_lo", "ck", "reset", "ebi_rxd_X", "ebi_rxd_Y", "ability_matched1", "ability_matched2", "rx_config_match", "rx_config_match1", "ability_matched_reg", "rx_config_match2", "ability_matched"]
    elif task_id == "ge_1000baseX_sync":
        return ["running_disparity", "ebi_K", "sync_status", "rx_even_m_init", "rx_even_m_set", "rx_even_m_clr", "rx_even_m_toggle", "ebi_rxd_out", "VALID", "K28_1_RX", "K28_5_RX", "K28_7_RX", "COMMA_RX", "COMMA_match", "cggood", "cgbad", "INVALID", "good_cgs_m_init", "good_cgs_m_inc", "good_cgs_m_cnt", "running_disparity_positive_m_set", "pcs_sync_present", "pcs_sync_next", "decoder_disparity_err", "startup_enable", "sync_m_acquired", "sync_m_lost", "ebi_K_out", "ebi_rxd_d1", "ebi_K_d1", "running_disparity_negative_m_set", "rx_even", "ck", "reset", "loopback", "ebi_rxd_X", "ebi_rxd_Y", "good_cgs", "ebi_rxd", "decoder_coding_err", "signal_detect"]
    elif task_id == "omsp_frontend":
        return ["cpu_halt_st", "decode_noirq", "e_state", "exec_done", "inst_ad", "inst_as", "inst_alu", "inst_bw", "inst_dest", "inst_dext", "inst_irq_rst", "inst_jmp", "inst_mov", "inst_sext", "inst_so", "inst_src", "inst_type", "irq_acc", "mab", "mb_en", "mclk_dma_enable", "mclk_dma_wkup", "mclk_enable", "mclk_wkup", "nmi_acc", "pc", "pc_nxt", "cpu_en_s", "cpu_halt_cmd", "cpuoff", "dbg_reg_sel", "dma_en", "dma_wkup", "fe_pmem_wait", "gie", "irq", "mclk", "mdb_in", "nmi_pnd", "nmi_wkup", "pc_sw", "pc_sw_wr", "puc_rst", "scan_enable", "wdt_irq", "wdt_wkup", "wkup"]
    elif task_id == "top":
        return ["clk", "rst", "busy", "data_in", "req_valid","wr_ready", "data_out", "credit_valid"]
    elif task_id == "control_unit":
        return ["sbox_sel", "rk_sel", "key_out_sel", "col_sel", "key_en", "col_en", "round", "bypass_rk", "bypass_key_en", "key_sel", "iv_cnt_en", "iv_cnt_sel", "key_derivation_en", "end_comp", "key_init", "key_gen", "mode_ctr", "mode_cbc", "last_round", "encrypt_decrypt", "operation_mode", "aes_mode", "start", "disable_core", "clk", "rst_n"]
    elif task_id == "uartRec":
        return ["bReg", "bNext", "stateReg", "stateNext", "rxDoneTick", "sReg", "sNext", "clk", "reset", "sTick", "rx", "dOut", "nReg", "nNext"]
    elif task_id == "host_interface":
        return ["HSElx", "buffer_write_en", "HREADYOUT", "HRESP", "rev_out_type", "HRDATA", "crc_poly_out", "buffer_full", "read_wait", "crc_idr_sel", "crc_poly_sel", "HADDR", "HCLK", "crc_poly_size", "buffer_read_en", "crc_cr_sel", "ahb_enable", "write_en", "crc_init_out", "crc_poly_en", "reset_pending", "bus_size", "crc_init_sel", "rev_in_type", "hsize_pp", "sample_bus", "crc_idr_out", "HWDATA", "crc_dr_sel", "HREADY", "bus_wr", "HSIZE", "crc_out", "crc_idr_en", "crc_init_en", "HRESETn", "hselx_pp", "read_en", "crc_cr_ff", "htrans_pp", "HTRANS", "haddr_pp", "hwrite_pp", "HWRITE", "reset_chain", "crc_cr_en", "DEFAULT_CLOCK", "DEFAULT_RESET", "crc_cr_rd"]
    elif task_id == "fpu_exceptions":
        return ["clk", "rst", "enable", "rmode", "opa", "opb", "in_except", "exponent_in", "mantissa_in", "fpu_op", "out", "ex_enable", "underflow", "overflow", "inexact", "exception", "invalid"]
    elif task_id == "FPV_MAC_rx_ctrl":
        return ["Rx_pkt_type_rmon", "Pause_next", "Too_short", "broadcast_ptr", "Frame_length_counter", "pause_quanta_h", "pause_quanta", "Fifo_data_end", "Fifo_full", "Pause_current", "MRxD", "Next_state", "IFG_counter", "MRxErr", "RxErr", "pause_quanta_val_tmp", "pause_frame_ptr", "RxD_dl1", "Rx_apply_rmon_tmp", "Fifo_data", "Fifo_data_en", "RX_IFG_SET", "RX_MIN_LENGTH", "RX_MAX_LENGTH", "CRC_en", "RxD", "Rx_pkt_err_type_rmon", "MCrs_dv", "Rx_apply_rmon", "CRC_init", "Too_long", "Fifo_data_err", "Crs_dv", "pause_quanta_val", "Rx_pkt_length_rmon", "MAC_add_en", "Rx_apply_rmon_tmp_pl1", "Reset", "Clk", "CRC_err", "MAC_rx_add_chk_err", "broadcast_drop", "Current_state"]
    elif task_id == "cavlc_read_levels":
        return ["level_4", "i", "level_14", "level_5", "TotalCoeff", "level_prefix_comb", "tmp1", "t1s_sel", "len_comb", "level_3", "prefix_sel", "clk", "level_1", "level_15", "level_abs_refresh", "first_level", "ena", "level_9", "level_6", "level_8", "rst_n", "level_2", "level_suffix", "rbsp", "level", "suffix_sel", "TrailingOnes", "level_12", "level_13", "level_10", "level_prefix", "level_abs", "level_0", "rbsp_internal", "level_7", "level_code_tmp", "calc_sel", "level_11", "suffixLength"]
    elif task_id == "can_fifo":
        return ["overrun_info", "rd_info_pointer", "rst", "wr_pointer", "wr", "release_buffer", "latch_overrun", "mbist_ctrl_i", "length_info", "wr_q", "fifo_full", "len_cnt", "data_in", "read_address", "fifo", "mbist_s_0", "info_full", "overrun", "fifo_cnt", "info_empty", "reset_mode", "initialize_memories", "clk", "write_length_info", "fifo_selected", "mbist_so_o", "length_fifo", "addr", "rd_pointer", "mbist_si_i", "info_cnt", "data_out", "fifo_empty", "wr_info_pointer", "extended_mode"]
    elif task_id == "fpu_exceptions":
        return ["out_1", "inexact_trigger", "subtract", "rmode", "mantissa_in", "in_et_zero", "sub_inf", "round_nearest", "out_0", "exp_2046", "inexact", "a_NaN", "inf_round_down", "opb_inf", "mul_inf", "div_inf", "clk", "overflow_trigger", "in_except", "SNaN_trigger", "out_inf_trigger", "out", "overflow", "div_by_0", "opa_QNaN", "opb", "multiply", "NaN_input_0", "divide", "addsub_inf", "opb_SNaN", "addsub_inf_invalid", "rst", "opb_et_zero", "underflow_trigger", "underflow", "div_by_inf", "exp_2047", "SNaN_input", "opb_pos_inf", "mul_0_by_inf", "enable", "fpu_op", "out_2", "add", "add_inf", "opa_neg_inf", "opa", "div_inf_by_inf", "out_neg_inf", "opa_pos_inf", "round_to_neg_inf", "invalid", "opa_et_zero", "round_to_pos_inf", "NaN_out_trigger", "mantissa_max", "ex_enable", "exponent_in", "invalid_trigger", "out_inf", "opb_QNaN", "opa_SNaN", "opb_neg_inf", "exception", "enable_trigger", "div_uf", "mul_uf", "round_to_zero", "input_et_zero", "out_pos_inf", "div_0_by_0", "opa_inf", "inf_round_down_trigger", "except_trigger", "NaN_input", "NaN_output_0", "NaN_output"]
    elif task_id == "PSGBusArb":
        return ["req3", "sel4", "sel3", "sel6", "sel1", "req4", "req2", "req6", "ack", "req1", "sel7", "req0", "sel2", "req5", "rst", "req7", "ce", "clk", "sel5", "seln", "sel0"]
    # elif task_id == "can_fifo":
    #     return []
    else:
        pattern = r"module\s+\w+\s*\((.*?)\);"
        
        # Find the module signal list between the parentheses
        match = re.search(pattern, verilog_code, re.DOTALL)
        
        if match:
            # Extract the signals declared in the module interface
            signals_list = match.group(1)
            
            # Split the signal list by commas and clean up any whitespace
            signals = [signal.strip() for signal in signals_list.split(",")]
            
            # Post-process to remove comments from the extracted signals
            pruned_signals = [re.sub(r'//.*', '', signal).strip() for signal in signals]
            
            # Return non-empty signals after pruning
            return [signal for signal in pruned_signals if signal]
        else:
            return []

def extract_module_name(verilog_code):
    # Regular expression to capture the module name
    match = re.search(r'\bmodule\s+(\w+)', verilog_code)
    if match:
        return match.group(1)
    else:
        return None

def parse_list_response(response):
    """
    Parses a numbered or bulleted list from the response string and returns a list of actions.
    
    Args:
        response (str): The raw text response from the LLM.
        
    Returns:
        List[str]: A list of actions extracted from the response.
    """
    # Initialize an empty list to store actions
    actions = []
    
    # Split the response into lines
    lines = response.strip().split('\n')
    
    # Regular expressions to match list items
    list_item_patterns = [
        r'^\s*\d+\.\s+(.*)',      # Matches numbered lists like "1. Action"
        r'^\s*-\s+(.*)',          # Matches bullet points like "- Action"
        r'^\s*\*\s+(.*)',         # Matches bullet points like "* Action"
        r'^\s*\d+\)\s+(.*)',      # Matches numbered lists like "1) Action"
    ]
    
    for line in lines:
        line = line.strip()
        action = None
        for pattern in list_item_patterns:
            match = re.match(pattern, line)
            if match:
                action = match.group(1).strip()
                break
        if action:
            actions.append(action)
        else:
            # If the line doesn't match any pattern but is non-empty, consider it as an action
            if line:
                actions.append(line)
    
    return actions

# task = "design2sva_fsm"
# task = "design2sva_pipeline"
# task = "design2sva_training_docs"
def package_testbench(row, lm_response: str, task) -> str:
    if task in ["design2sva_fsm", "design2sva_pipeline", "design2sva_training_docs"]:
        testbench_text_prefix = row.testbench
        testbench_text_prefix = testbench_text_prefix.split("assign tb_reset")[0]
        if task == "design2sva_fsm" or task == "design2sva_pipeline":
            testbench_text_prefix += "assign tb_reset = (reset_ == 1'b0);\n"
        elif task == "design2sva_training_docs":
            testbench_text_prefix += "assign tb_reset = (rst == 1'b0);\n"
        testbench_text_postfix = "endmodule\n" + row.testbench.split("endmodule")[-1]
        lm_response = parse_code_response(lm_response)
        packaged_tb_text = (
            testbench_text_prefix + "\n" + lm_response + "\n" + testbench_text_postfix
        )
        return packaged_tb_text
    elif task in ["design2sva_assertionbench"]:
        testbench_text_prefix = row.testbench.split("endmodule")[0]
    
        # Parse the lm_response to ensure valid code (assertions)
        # lm_response = parse_code_response(lm_response)
        
        # Reassemble the testbench by inserting the assertions before "endmodule"
        testbench_text_postfix = row.testbench.split("endmodule")[-1]
        
        # Combine the prefix, assertions (lm_response), and the postfix
        packaged_tb_text = (
            testbench_text_prefix + "\n" + lm_response + "\nendmodule\n" + testbench_text_postfix
        )
    
        return packaged_tb_text
    
def prune_and_deduplicate_assertions(assertions):
    def remove_comments(assertion):
        # Split lines and filter out comment lines
        cleaned_lines = [line for line in assertion.split('\n') if not line.strip().startswith('//')]
        return '\n'.join(cleaned_lines).strip()

    # Step 1: Create a dictionary to store unique assertions
    # Key is the assertion without comments, value is the original assertion with comments
    unique_assertions = {}
    
    for assertion in assertions:
        # Clean the assertion for comparison by removing comments
        cleaned_assertion = remove_comments(assertion)
        if cleaned_assertion not in unique_assertions:
            # Keep the original assertion (with comments) if it's unique
            unique_assertions[cleaned_assertion] = assertion
    
    # Step 2: Convert the dictionary values (original assertions) back to a list
    all_valid_assertions = list(unique_assertions.values())
    
    return all_valid_assertions