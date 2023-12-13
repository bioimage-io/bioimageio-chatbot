import os
import asyncio
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum
import cv2
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.provider.openai_api import retry


import matplotlib.pyplot as plt

from imjoy_rpc.hypha import connect_to_server

class TaskChoice(str, Enum):
    """The best guess for the image segmentation task. Either 'cyto' or 'nuclei'"""
    cyto = "cyto"
    nuclei = "nuclei"
    unknown = "unknown"

class CellposeTask(BaseModel):
    """Take the user's uploaded image and run it through cellpose"""
    request: str = Field(description="The user's request")
    task: TaskChoice = Field(description="The best guess for the image analysis task. Either 'cyto' (cytoplasm segmentation) or 'nuclei' (nuclei segmentation). If not known, set to `unknown`")
    description: Optional[str] = Field(description="A description of the image if provided by the user, otherwise `unknown`")

class AxisGuess(BaseModel):
    """The best guess for what each axis in the image corresponds to. The largest dimensions will be 'x' and 'y'. The 'c' dimension will not be larger than 5. The numbers of dimensions in the shape tuple MUST match the number of axis labels"""
    labels : list[str] = Field(description = f"The axis label for each dimension in the image's shape.")

class UnlabeledImage(BaseModel):
    """An input image"""
    shape : list[int] = Field(description="The image's shape")

class LabeledImage(UnlabeledImage):
    """An image whose axes have been labeled according to its shape e.g. ['b','c','y','x']. The labels should intuitively make sense. The length of this list MUST exactly match the number of dimensions in the image's shape. You should avoid using the label 't' if possible. The 'c' dimension should not be larger than 5"""
    axes : AxisGuess = Field(description = "A list representing the axes of the image (for example ['c','y','x'], ['b','x','y','c']). Each entry corresponds to the axis label of the image along the corresponding dimension. 't' is the least likely axis to appear in any image. The length of this string MUST match the number of dimensions in the image's `shape`. If in doubt between labeling a dimension as 'z' or 'c', 'c' should be assigned to the smaller dimension.")
    @validator('axes')
    def validate_axes_length(cls, v, values):
        if 'shape' in values:
            if len(v.labels) != len(values['shape']):
                raise ValueError(f"The number of characters in the axes label string MUST exact match the number of dimensions in the image's shape. The number of dimensions in {values['shape']} is {len(values['shape'])} but the number of characters in the axes label string ({v}) is {len(v.labels)}")
            if 'c' in v.labels and values['shape'][v.labels.index('c')] > 5:
                raise ValueError(f"Error, the number of channels ('c' dimension) should not be greater than 5.")
        for c in v.labels:
            if c not in ['b', 'x', 'y', 'z', 'c', 't']:
                raise ValueError("Please confine your axis labels to the characters 'b', 'x', 'y', 'z', 'c', 't'")
        if 't' in v.labels and np.any([c not in v.labels for c in ['z', 'b', 'x', 'y']]):
            raise ValueError("Please prioritize using 'z', 'b', 'x', or 'y' over 't' as an axis label. The label 't' should only be used as as last resort.")
        if np.any([v.labels.count(c) > 1 for c in v.labels]):
            raise ValueError("Every unique character can be used only once in the axes labels")
        return v
    
class UnlabeledImages(BaseModel):
    """A list of unlabeled images"""
    unlabeled_images : list[UnlabeledImage] = Field(description="The unlabled images")

class LabeledImages(BaseModel):
    """A list of images whose axes have been labeled"""
    labeled_images : list[LabeledImage] = Field(description="The labeled images")

async def agent_guess_image_axes(image : UnlabeledImage, role : Role = None) -> LabeledImage:
    """Guesses the axis labels based on the image shape. The largest dimensions will be 'x' and 'y'. The 'c' dimension will not be larger than 5. The numbers of dimensions in the shape tuple must the number of axis labels"""
    # response = await role.aask(image, LabeledImage)
    response = role.aask(image.shape, AxisGuess)
    labeled_image = LabeledImage(shape = image.shape, axes = response.axes)
    return labeled_image

async def retry_aask(role, ui, output_type):
    @retry(5)
    async def inner():
        return await role.aask(ui, output_type)
    return await inner()

async def agent_guess_all_axes(unlabeled_images : UnlabeledImages, role : Role = None) -> LabeledImages:
    """Labels the axes in all images in the input list of unlabeled images"""
    labeled_images = []
    for unlabeled_image in unlabeled_images.unlabeled_images:
        labeled_image = await role.aask(unlabeled_image, LabeledImage)
        labeled_images.append(labeled_image)
    
    guessing_tasks = (retry_aask(role, ui, LabeledImage) for ui in unlabeled_images.unlabeled_images)
    labeled_images = await asyncio.gather(*guessing_tasks)
    labeled_images = LabeledImages(labeled_images=labeled_images)
    return labeled_images


def resize_image(input_image : np.ndarray, input_format : str, output_format : str, output_dims_xy = tuple[int,int], grayscale : bool = False, output_type = np.float32):
    current_format = input_format.lower()
    output_format = output_format.lower()
    inter_format = "yxc"
    rearranged = input_image.copy()
    assert sorted(current_format) == sorted(output_format) == ['c', 'x', 'y']
    transposed = np.transpose(rearranged, [current_format.index(c) for c in inter_format])
    current_format = inter_format
    resized = cv2.resize(transposed, output_dims_xy, interpolation = cv2.INTER_AREA)
    if grayscale:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        current_format = "yxc"
    resized = resized.astype(output_type)
    resized = np.transpose(resized, [current_format.index(c) for c in output_format])
    return(resized)

async def run_cellpose(img, server_url : str = "https://ai.imjoy.io", diameter = None, model_type = 'cyto', method_timeout = 30, server = None): # model_type = 'cyto' or 'nuclei')
    params = {'diameter' : diameter, 'model_type' : model_type}
    img_input =img.copy()
    if server is None:
        cellpose_server = await connect_to_server({"name": "client", "server_url": server_url, "method_timeout": method_timeout})
    else:
        cellpose_server = server
    triton = await cellpose_server.get_service("triton-client")
    results = await triton.execute(inputs=[img_input,params], model_name = "cellpose-python", decode_bytes=True)
    return results

async def guess_image_axes(input_files : list | str):
    single_input = isinstance(input_files, str)
    if single_input:
        input_files = [input_files]
    axis_guesser = Role(name = "AxisGuesser",
                profile = "Axis Guesser",
                goal="Your goal as AxisGuesser is read the shapes of input images and guess their axis labels using common sense.",
            constraints=None,
            actions=[agent_guess_image_axes, agent_guess_all_axes])
    event_bus = axis_guesser.get_event_bus()
    event_bus.register_default_events()
    message_input = UnlabeledImages(unlabeled_images = [UnlabeledImage(shape = read_image(fname).shape) for fname in input_files])
    m = Message(content = 'guess the image axes for each image in the list', data = message_input, role = 'User')
    responses = await axis_guesser.handle(m)
    guessed_axes = [''.join(x.axes.labels) for x in responses[0].data.labeled_images]
    if single_input:
        return guessed_axes[0]
    else:
        return guessed_axes

def read_image(input_image_path) -> np.ndarray:
    if input_image_path.endswith('.npy'):
        input_image = np.load(input_image_path)
    else:
        input_image = cv2.imread(input_image_path)
    return input_image