from .cellpose.image_processing import *

def get_modes():
    cellpose_description = "Runs Cellpose image segmentation (either cytoplasm or nuclei) on user images using pretrained models"
    mode_dict = {}
    for mode, channel_name, response_function, description in [[CellposeTask, 'Cellpose Image Analyzer', cellpose_get_response, cellpose_description]]:
        mode_dict[channel_name] = {
            'mode': mode,
            'response_function': response_function,
            'description': description
        }
    return mode_dict