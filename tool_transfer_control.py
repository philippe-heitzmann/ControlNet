# Refactoring https://github.com/lllyasviel/ControlNet/blob/main/tool_transfer_control.py

'''
Example call:
python script.py './models/v1-5-pruned.ckpt' './models/control_sd15_openpose.pth' './models/anything-v3-full.safetensors' './models/control_any3_openpose.pth'
'''

import argparse
import os
import torch
from share import *
from cldm.model import load_state_dict


def get_node_name(name: str, parent_name: str) -> tuple[bool, str]:
    """
    Extracts the node name from the given full name using the parent name.

    Args:
        name (str): Full name of the node.
        parent_name (str): Parent name to extract the node name.

    Returns:
        tuple[bool, str]: A tuple containing a boolean value indicating whether
                          the node is a first stage or conditional stage, and
                          the extracted node name.
    """
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


def transfer_model(path_sd15: str, path_sd15_with_control: str, path_input: str, path_output: str) -> None:
    """
    Transfers the model weights based on the provided paths and saves the transferred model.

    Args:
        path_sd15 (str): Path to the SD15 model weights.
        path_sd15_with_control (str): Path to the SD15 model with control weights.
        path_input (str): Path to the input model weights.
        path_output (str): Path to save the transferred model weights.
    """
    assert os.path.exists(path_sd15), 'Input path_sd15 does not exist!'
    assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exist!'
    assert os.path.exists(path_input), 'Input path_input does not exist!'
    assert os.path.exists(os.path.dirname(path_output)), 'Output folder does not exist!'

    sd15_state_dict = load_state_dict(path_sd15)
    sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
    input_state_dict = load_state_dict(path_input)

    keys = sd15_with_control_state_dict.keys()

    final_state_dict = {}
    for key in keys:
        is_first_stage, _ = get_node_name(key, 'first_stage_model')
        is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
        if is_first_stage or is_cond_stage:
            final_state_dict[key] = input_state_dict[key]
            continue
        p = sd15_with_control_state_dict[key]
        is_control, node_name = get_node_name(key, 'control_')
        if is_control:
            sd15_key_name = 'model.diffusion_' + node_name
        else:
            sd15_key_name = key
        if sd15_key_name in input_state_dict:
            p_new = p + input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
            # print(f'Offset clone from [{sd15_key_name}] to [{key}]')
        else:
            p_new = p
            # print(f'Direct clone to [{key}]')
        final_state_dict[key] = p_new

    torch.save(final_state_dict, path_output)
    print('Transferred model saved at ' + path_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer model weights with control')
    parser.add_argument('path_sd15', type=str, help='Path to the SD15 model weights')
    parser.add_argument('path_sd15_with_control', type=str, help='Path to the SD15 model with control weights')
    parser.add_argument('path_input', type=str, help='Path to the input model weights')
    parser.add_argument('path_output', type=str, help='Path to save the transferred model weights')

    args = parser.parse_args()

    transfer_model(args.path_sd15, args.path_sd15_with_control, args.path_input, args.path_output)
