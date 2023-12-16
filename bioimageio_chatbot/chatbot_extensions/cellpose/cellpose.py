import os
import asyncio
import base64
from enum import Enum
from imjoy_rpc.hypha import connect_to_server
import cv2
import torch
import numpy as np
from schema_agents.provider.openai_api import retry
from pydantic import BaseModel, Field, validator
from schema_agents.role import Role
from schema_agents.schema import Message
import matplotlib.pyplot as plt
from typing import Optional
from bioimageio_chatbot.chatbot_extensions.cellpose.results_display import create_results_page
from bioimageio_chatbot.jsonschema_pydantic import jsonschema_to_pydantic, JsonSchemaObject

class TaskChoice(str, Enum):
    """The best guess for the image segmentation task. Either 'cyto' or 'nuclei'"""
    cyto = "cyto"
    nuclei = "nuclei"
    unknown = "unknown"

class CellposeTask(BaseModel):
    """Take the user's uploaded image and run it through Cellpose segmentation making sure that the user has both uploaded an image and has specified a task (either cytoplasm or nuclei segmentation)"""
    request: str = Field(description="The user's request")
    task: TaskChoice = Field(description="The best guess for the image analysis task. Either 'cyto' (cytoplasm segmentation) or 'nuclei' (nuclei segmentation). If not known, set to `unknown`")
    diameter: Optional[float] = Field(description="The diameter of the cells in pixels if specified by the user.")

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

async def guess_image_axes(input_files : list | str):
    image_processor = ImageProcessor()
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
    message_input = UnlabeledImages(unlabeled_images = [UnlabeledImage(shape = image_processor.read_image(fname).shape) for fname in input_files])
    m = Message(content = 'guess the image axes for each image in the list', data = message_input, role = 'User')
    responses = await axis_guesser.handle(m)
    guessed_axes = [''.join(x.axes.labels) for x in responses[0].data.labeled_images]
    if single_input:
        return guessed_axes[0]
    else:
        return guessed_axes
    
def decode_base64(encoded_data):
    header, encoded_content = encoded_data.split(',')
    decoded_data = base64.b64decode(encoded_content)
    file_extension = header.split(';')[0].split('/')[1]
    return decoded_data, file_extension

def encode_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    # format it for the html img tag
    encoded_string = f"data:image/png;base64,{encoded_string.decode('utf-8')}"
    return encoded_string

class ImageProcessor():
        
    def resize_image(self, input_image : np.ndarray, input_format : str, output_format : str, output_dims_xy = tuple[int,int] | None, grayscale : bool = False, output_type = np.float32):
        current_format = input_format.lower()
        output_format = output_format.lower()
        inter_format = "yxc"
        rearranged = input_image.copy()
        assert sorted(current_format) == sorted(output_format) == ['c', 'x', 'y']
        transposed = np.transpose(rearranged, [current_format.index(c) for c in inter_format])
        current_format = inter_format
        if output_dims_xy is not None:
            resized = cv2.resize(transposed, output_dims_xy, interpolation = cv2.INTER_AREA)
        if grayscale:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            current_format = "yxc"
        resized = resized.astype(output_type)
        resized = np.transpose(resized, [current_format.index(c) for c in output_format])
        return(resized)

    def read_image(self, input_image_path) -> np.ndarray:
        if input_image_path.endswith('.npy'):
            input_image = np.load(input_image_path)
        else:
            input_image = cv2.imread(input_image_path)
        return input_image

    def get_torch_image(self, input_image_path, input_axes, grayscale : bool = True):
        input_image = self.read_image(input_image_path)
        resized_image = self.resize_image(input_image, input_axes, grayscale=grayscale, output_format = "bcyx")
        resized_image = resized_image.astype(np.float32)
        torch_image = torch.from_numpy(resized_image).to(torch.float32)
        return torch_image

async def run_cellpose(img : np.ndarray, server_url : str = "https://ai.imjoy.io", diameter : float | None = None, model_type = 'cyto', method_timeout = 30, server = None):
    # model_type = 'cyto' or 'nuclei')
    params = {'diameter' : diameter, 'model_type' : model_type}
    img_input =img.copy()
    if server is None:
        cellpose_server = await connect_to_server({"name": "client", "server_url": server_url, "method_timeout": method_timeout})
    else:
        cellpose_server = server
    triton = await cellpose_server.get_service("triton-client")
    results = await triton.execute(inputs=[img_input,params], model_name = "cellpose-python", decode_bytes=True)
    return results

class CellposeHelp(BaseModel):
    """The help message for the Cellpose Helper to print to the user"""
    message : str = Field(description = "The help message for the Cellpose Helper")

async def print_cellpose_help(situation : str, role : Role = None) -> CellposeHelp:
    """Takes the situation description and prints a helpful message to the user. Currently this Cellpose API can take .png, .tiff, and .jpeg images up to 2MB in size and run either cytoplasmic or nuclei segmentation"""
    response = await role.aask(situation, CellposeHelp)
    return response

async def create_cellpose_help(situation : str) -> str:
    cellpose_helper = Role(name = "CellposeHelper",
                           profile = "Cellpose Helper",
                           goal = "Your goal is write helpful messages to the user telling them about how they can use this Cellpose service",
                           constraints = None,
                           actions = [print_cellpose_help])
    event_bus = cellpose_helper.get_event_bus()
    event_bus.register_default_events()
    message_input = situation
    m = Message(content = situation, role = 'User')
    responses = await cellpose_helper.handle(m)
    print(responses)
    return responses[0].data.message

class CellposeResults(BaseModel):
    """The results of running Cellpose segmentation on an image"""
    mask_shape : list[int] = Field(description = "The shape of the output mask")
    info : list = Field(description = "The output info dictionary. If the user did not specify a diameter, this was estimated by Cellpose")
    number_segmented_regions : int = Field(description = "The number of segmented regions in the output mask")
    user_diameter : bool = Field(description = "True if the user specified an input diameter, False otherwise")
    input_diameter : Optional[float] = Field(description = "The user-specified input diameter if the user specified one, None otherwise")

class ResultsSummary(BaseModel):
    """The results summary message to print to the user. Speak directly to the user"""
    message : str = Field(description = "The results summary message to print to the user from the run.")

async def print_results_summary(cellpose_results : CellposeResults, role : Role = None) -> ResultsSummary:
    """Takes the Cellpose results and creates a results summary message to print to the user"""
    response = await role.aask(cellpose_results, ResultsSummary)
    return response

async def create_results_summary(cellpose_results : CellposeResults) -> str:
    results_summarizer = Role(name = "ResultsSummarizer",
                              profile = "Results Summarizer",
                              goal = "Your goal is write helpful messages to the user summarizing the results of their image analysis",
                              actions = [print_results_summary])
    event_bus = results_summarizer.get_event_bus()
    event_bus.register_default_events()
    m = Message(content = cellpose_results.json(), data = cellpose_results, role = 'User')
    responses = await results_summarizer.handle(m)
    print(responses)
    return responses[0].data.message
    

async def cellpose_get_response(question_with_history, req : CellposeTask):
    if not question_with_history.image_data:
        situation = "User did not upload an image. User's question was: " + question_with_history.question
        return await create_cellpose_help(situation)
    try:
        decoded_image_data, image_ext = decode_base64(question_with_history.image_data)
        print('Size of data: ')
        print(len(decoded_image_data))
        mb_size_max = 2.0
        if len(decoded_image_data) / 1e6 > mb_size_max:
            situation = "User uploaded an image that is too large (greater than 2MB). User's question was: " + question_with_history.question
            return await create_cellpose_help(situation)
    except Exception as e:
        return f"I failed to decode the uploaded image, error: {e}"
    if image_ext not in ['png', 'jpeg', 'jpg', 'tiff']:
        situation = "User uploaded a file that is not a .png, .jpeg, or .tiff image. User's question was: " + question_with_history.question
        return await create_cellpose_help(situation)
    if req.task == "unknown":
        situation = "User uploaded an image but did not specify a task. Clarify that right now you can run Cellpose segmentation on either cytoplasm (`cyto`) or nuclei (`nuclei`). User's question was: " + question_with_history.question
        return await create_cellpose_help(situation)
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    image_path = os.path.join(tmp_dir, f'tmp-user-image.{image_ext}')
    with open(image_path, 'wb') as f:
        f.write(decoded_image_data)
    image_processor = ImageProcessor()
    arr = image_processor.read_image(image_path)
    axes = await guess_image_axes(image_path)
    if sorted(axes) != ['c', 'x', 'y']:
        situation = "User uploaded an image with an unexpected number of axes. Currently, only images with 3 axes (channel, x, and y) are supported"
        return await create_cellpose_help(situation)
    arr_resized = arr.astype(np.float32)
    resized_fname_npy = os.path.join(tmp_dir, 'user-image.npy')
    np.save(resized_fname_npy, arr_resized)
    fig, ax = plt.subplots()
    ax.imshow(arr_resized / 255.0)
    ax.axis('off')
    resized_fname = os.path.join(tmp_dir, 'user-image.png')
    fig.savefig(resized_fname, bbox_inches='tight')
    plt.close() 
    print("Running cellpose...")
    print(arr_resized.shape)
    diameter = req.diameter if "diameter" in req.dict().keys() else None
    results = await run_cellpose(arr_resized, server_url="https://ai.imjoy.io", model_type = req.task, diameter = diameter)
    mask = results['mask'] 
    seg_areas = len(np.unique(mask[mask != 0]))
    info = results['info'] 
    mask_shape = results['__info__']['outputs'][0]['shape']
    print(seg_areas)
    print(info)
    print(mask_shape)
    fig, ax = plt.subplots()
    ax.imshow(mask[0,:,:])
    ax.axis('off')
    output_fname = os.path.join(tmp_dir, 'output-image.png')
    fig.savefig(output_fname, bbox_inches='tight')
    plt.close()
    output_fname_npy = os.path.join(tmp_dir, 'output-image.npy')
    np.save(output_fname_npy, mask)
    result_images = [resized_fname, output_fname]
    npy_paths = [resized_fname_npy, output_fname_npy]
    result_headers = ["User image", "Cellpose segmentation results"]
    results_link = await create_results_page("https://ai.imjoy.io", result_images, result_headers, npy_paths)
    cellpose_results = CellposeResults(mask_shape = mask_shape, info = info, number_segmented_regions = seg_areas, user_diameter = diameter is not None, input_diameter = diameter)
    out_string = await create_results_summary(cellpose_results)
    out_string += f"\n\n[View and download results]({results_link})"
    return out_string

if __name__ == "__main__":
    cellpose_results = CellposeResults(mask_shape = [3,512,512], info = {'a':1}, number_segmented_regions = 2)
    loop = asyncio.get_event_loop()
    loop.create_task(create_results_summary(cellpose_results))
    loop.run_forever()