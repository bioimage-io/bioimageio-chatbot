import os
import typing as T
import pickle as pkl
import sys
import asyncio
import xml.etree.ElementTree as ET
import randomname

import scipy
import matplotlib.pyplot as plt
import cv2
import yaml
import requests
import torch
import torch.nn as nn
import numpy as np
from xarray import DataArray
from tqdm.auto import tqdm
from skimage import exposure
import matplotlib as mpl
from schema_agents.provider.openai_api import retry
from pydantic import BaseModel, Field, validator
from schema_agents.role import Role
from schema_agents.schema import Message
import shutil


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
    return(labeled_image)

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
    return(labeled_images)


async def guess_image_axes(input_files : list):
    image_processor = ImageProcessor()
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
    return(guessed_axes)


def create_svg_table_unknown(input_images, shapes, guessed_axes):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    row_height = 20
    column_widths = [200, 120, 120]  # Adjust as needed
    header_height = 30
    svg_height = header_height + row_height * len(input_images)
    svg = ET.Element(ET.QName("http://www.w3.org/2000/svg", 'svg'), width='1000', height=str(svg_height))
    ET.SubElement(svg, 'style').text = 'text { font-family: Arial; }'
    def add_text(parent, x, y, text, font_size='12'):
        element = ET.SubElement(parent, 'text', x=str(x), y=str(y), 
                                style=f'font-size:{font_size}px;', 
                                **{'text-anchor': 'start'})
        element.text = text
    # Add table headers
    headers = ["Image", "Shape", "Guessed Axes"]
    x_position = 0
    for i, header in enumerate(headers):
        add_text(svg, x_position + 5, header_height - 10, header, font_size='14')
        x_position += column_widths[i]
    # Add rows for each image
    for i, (img, shape, guessed_ax) in enumerate(zip(input_images, shapes, guessed_axes)):
        y_position = header_height + i * row_height
        row_color = "#fffbfe"
        ET.SubElement(svg, 'rect', x='0', y=str(y_position), 
                      width='1000', height=str(row_height), 
                      fill=row_color)
        x_position = 0
        for j, data in enumerate([img.split('/')[-1], str(shape), guessed_ax]):
            add_text(svg, x_position + 5, y_position + 15, data)
            x_position += column_widths[j]
    return ET.tostring(svg, encoding='unicode')

def create_svg_table_known(input_images, true_shapes, true_axes, guessed_axes):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    # Constants for table layout
    row_height = 20
    column_widths = [200, 120, 100, 120]  # Adjust as needed
    header_height = 30
    # Calculate SVG height based on the number of rows
    svg_height = header_height + row_height * len(input_images)
    # Create SVG element
    svg = ET.Element(ET.QName("http://www.w3.org/2000/svg", 'svg'), width='1000', height=str(svg_height))
    ET.SubElement(svg, 'style').text = 'text { font-family: Arial; }'
    # Function to add a text element
    def add_text(parent, x, y, text, font_size='12'):
        element = ET.SubElement(parent, 'text', x=str(x), y=str(y), 
                                style=f'font-size:{font_size}px;', 
                                **{'text-anchor': 'start'})
        element.text = text
    # Add table headers
    headers = ["Image", "True Shape", "True Axes", "Agent Guessed Axes"]
    x_position = 0
    for i, header in enumerate(headers):
        add_text(svg, x_position + 5, header_height - 10, header, font_size='14')
        x_position += column_widths[i]
    # Add rows for each image
    for i, (img, shape, true_ax, guessed_ax) in enumerate(zip(input_images, true_shapes, true_axes, guessed_axes)):
        y_position = header_height + i * row_height
        # row_color = '#90ee90' if true_ax == guessed_ax else '#ffcccb'  # Lighter shades of green and red
        if true_ax == guessed_ax:
            row_color = '#90ee90' # light green
        elif true_ax.replace('x', '[').replace('y', 'x').replace('[', 'y') == guessed_ax: # (everything is same except for x and y)
            row_color = "#ccccff" # light blue
        else:
            row_color = '#ffcccb' # light red
        # Background color for row
        ET.SubElement(svg, 'rect', x='0', y=str(y_position), 
                      width='1000', height=str(row_height), 
                      fill=row_color)
        # Each column in a row
        x_position = 0
        for j, data in enumerate([img.split('/')[-1], str(shape), true_ax, guessed_ax]):
            add_text(svg, x_position + 5, y_position + 15, data)
            x_position += column_widths[j]
    # Convert to string
    return ET.tostring(svg, encoding='unicode')


class Embedder():
    def __init__(self):
        pass

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        pass

    @property
    def vector_size(self) -> int:
        pass

    @property
    def input_format(self) -> str:
        return "bcyx"


class ResNet50Embedder(Embedder):
    def __init__(self):
        from torchvision.models import resnet50, ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.model.eval()

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embeds an image using the ResNet50 model.
        Args:
            image: A numpy array of shape (b, 3, y, x)
        """
        tensor = torch.from_numpy(image).to(torch.float32)
        vector = self.model(tensor)
        array = vector.detach().numpy()
        array = array.flatten()
        return array

    @property
    def vector_size(self) -> int:
        return 2048


class VGG16Embedder(Embedder):
    def __init__(self):
        from torchvision.models import vgg16, VGG16_Weights
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Identity()
        self.model.eval()

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embeds an image using the VGG16 model.
        Args:
            image: A numpy array of shape (b, 3, y, x)
        """
        tensor = torch.from_numpy(image).to(torch.float32)
        vector = self.model(tensor)
        array = vector.detach().numpy()
        array = array.flatten()
        return array

    @property
    def vector_size(self) -> int:
        return 4096


class ImageProcessor():
    def __init__(self, embedder: Embedder = None):
        self.vector_embedding = None
        self.image = None
        self.embedder = embedder or VGG16Embedder()

    def load_image(self, image_path):
        try:
            self.image = np.load(image_path)
        except Exception as e:
            raise Exception(f"Error loading image from {self.image_path}: {str(e)}")
        
    def standardize_image(self, input_image: np.ndarray, input_format: str, standard_format: str = 'byxcz'):
        current_format = input_format.lower()
        rearranged = input_image.copy()
        for i_c, c in enumerate(standard_format):
            if c not in current_format:
                rearranged = np.expand_dims(rearranged, axis = 0)
                current_format = c + current_format
            if c == 'b':
                rearranged = np.mean(rearranged, axis = current_format.index(c), keepdims=True)
            if c == 'z':
                rearranged = np.mean(rearranged, axis = current_format.index(c), keepdims=True)
        for c in current_format:
            if c not in standard_format:
                rearranged = np.mean(rearranged, axis = current_format.index(c), keepdims = False)
                current_format = ''.join([x for x in current_format if x != c])
        rearranged = rearranged.transpose([current_format.index(c) for c in standard_format])
        rearranged = cv2.normalize(rearranged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # gkreder
        # rearranged = rearranged.astype(np.uint8) # gkreder
        return(rearranged)

        

    def resize_image(self, input_image: np.ndarray, input_format: str, grayscale : bool, output_format="byxc", output_dims_xy=(224, 224)):
        """Resizes and converts to an RGB output image with optional grayscaling"""
        input_format = input_format.lower()
        output_format = output_format.lower()
        for c in 'xyc':
            assert c in output_format
        # Rearrange the dimensions to a standard format (e.g., "byxcz")
        standard_format = "byxcz"
        rearranged = self.standardize_image(input_image, input_format, standard_format=standard_format)
        current_format = standard_format

        # Resize image
        resized_channels = []
        num_channels = rearranged.shape[standard_format.index('c')]
        # "byxcz"
        composite_rgb = np.zeros((output_dims_xy[1], output_dims_xy[0], 3))
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
        for i in range(num_channels):
            # "byxcz"
            channel_img = rearranged[(0, slice(None), slice(None), i, 0)]
            resized_channel_img = cv2.resize(channel_img, output_dims_xy[::-1])
            resized_channels.append(resized_channel_img)
            composite_rgb += resized_channel_img[:,:,None] * np.array(colors[i]).reshape(1, 1, 3)
        composite_rgb = cv2.normalize(composite_rgb, None, 0, 255, cv2.NORM_MINMAX)
        composite_rgb = composite_rgb.astype(np.uint8)
        resized = np.stack(resized_channels, axis=-1) # yxc
        current_format = 'yxc'
        if grayscale:
            composite_rgb = cv2.cvtColor(composite_rgb, cv2.COLOR_BGR2GRAY)
            if num_channels == 1:
                composite_rgb = cv2.cvtColor(composite_rgb, cv2.COLOR_GRAY2BGR)
                composite_rgb = cv2.normalize(composite_rgb, None, 0, 255, cv2.NORM_MINMAX)
            else:
                composite_rgb = cv2.cvtColor(composite_rgb, cv2.COLOR_GRAY2BGR)
        # # Convert to grayscale and back to RGB if required
        # if grayscale:
        #     if num_channels > 1:
        #         resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #         resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        # elif num_channels == 1:
        #     resized = cv2.cvtColor(np.uint8(resized), cv2.COLOR_GRAY2BGR)
        # elif num_channels == 4:
        #     resized = cv2.cvtColor(np.uint8(resized[:,:,:3]), cv2.COLOR_GRAY2BGR)
        for c in output_format:
            if c not in current_format:
                composite_rgb = np.expand_dims(composite_rgb, axis = 0)
                current_format = c + current_format
        output_image = composite_rgb.transpose([current_format.index(c) for c in output_format])
        return output_image

    def embed_image(self, input_image, current_format: str, grayscale : bool = True):
        resized_image = self.resize_image(
            input_image,
            current_format,
            grayscale = grayscale,
            output_format=self.embedder.input_format)
        
        vector_embedding = self.embedder.embed_image(resized_image)
        self.vector_embedding = vector_embedding
        return vector_embedding

    def save_output(self):
        try:
            np.save(self.output_path, self.vector_embedding)
        except Exception as e:
            raise Exception(f"Error saving output to {self.output_path}: {str(e)}")

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


def get_model_rdf(m: dict, db_path: str) -> T.Optional[dict]:
    """Gets the model rdf_source yaml (and writes to db_path). 
    Returns None if the yaml doesn't have test_inputs or input_axes"""
    nickname = m['nickname']
    yaml_dir = os.path.join(db_path, 'rdf_sources')
    os.makedirs(yaml_dir, exist_ok=True)
    yaml_name = os.path.join(yaml_dir, f"{nickname}.yaml")
    if not os.path.exists(yaml_name):
        response = requests.get(m['rdf_source'])
        if response.status_code == 200:
            rdf_source_content = response.content.decode('utf-8')
    else:
        with open(yaml_name, 'r', encoding="utf-8") as f:
            rdf_source_content = f.read()
    rdf_dict = yaml.safe_load(rdf_source_content)
    with open(yaml_name, 'w', encoding='utf-8') as f:
        f.write(rdf_source_content)
    return rdf_dict


def get_model_torch_images(rdf_dict: dict, db_path : str, grayscale : bool = True) -> (list, list):
    image_embedder = ImageProcessor()
    mislabeled_axes = {'impartial-shark' : 'yx'} # impartial shark has mislabeled axes
    nickname = rdf_dict['config']['bioimageio']['nickname']
    input_images_dir = os.path.join(db_path, "input_images")
    os.makedirs(input_images_dir, exist_ok=True)
    test_inputs = rdf_dict['test_inputs']
    input_axes = rdf_dict['inputs'][0]['axes']
    if nickname in mislabeled_axes:
        input_axes = mislabeled_axes[nickname]
    torch_images = []
    model_metadata = []
    for i_ti, ti in enumerate(test_inputs):
        if "input_test_time_for" in ti: # nice-peacock has non-usable inputs after first?
            continue
        input_image_file = os.path.join(input_images_dir, f"{nickname}.npy")
        if not os.path.exists(input_image_file):
            response = requests.get(ti)
            if response.status_code == 200:
                with open(input_image_file, 'wb') as file:
                    file.write(response.content)
        torch_image = image_embedder.get_torch_image(input_image_file, input_axes, grayscale = grayscale)
        torch_images.append(torch_image)
        model_metadata.append(rdf_dict)
    return torch_images, model_metadata


def get_torch_db(db_path: str, processor: ImageProcessor, force_build: bool = False,
                 grayscale : bool = True) -> list[(torch.Tensor, dict)]:
    from bioimageio_chatbot.chatbot import load_model_info
    out_db_path = os.path.join(db_path, 'db.pkl')
    if (not force_build) and os.path.exists(out_db_path):
        db = pkl.load(open(out_db_path, 'rb'))
        return db
    models = load_model_info()
    all_metadata = []
    all_torch_images = []
    all_embeddings = []
    print('Creating PyTorch image database from model zoo...')
    for _, m in enumerate(tqdm(models)):
        if 'nickname' not in m:
            continue
        rdf_dict = get_model_rdf(m, db_path)
        if not rdf_dict:
            continue
        model_torch_images, model_metadata = get_model_torch_images(
            rdf_dict, db_path, grayscale=grayscale)
        all_torch_images.extend(model_torch_images)
        all_metadata.extend(model_metadata)
        for img in model_torch_images:
            vec = processor.embed_image(img.numpy(), "bcyx")
            all_embeddings.append(vec)
    # db_old = pkl.load(open(out_db_path, 'rb'))
    # all_torch_images, all_embeddings, all_metadata = zip(*db_old)
    # all_torch_images = list(all_torch_images)
    # all_embeddings = list(all_embeddings)
    # all_metadata = list(all_metadata)

    cellpose_dir = os.path.join(db_path, 'cellpose_train-set_representatives')
    cellpose_temp_dir = os.path.join(db_path, "cellpose_tmp")
    if not os.path.exists(cellpose_dir):
        os.makedirs(cellpose_temp_dir, exist_ok=True)
        cellpose_drive_id = "1kEy9DHxkGeQHQDQco8CyaL1KEEPS6NYY"
        cellpose_zip_file = os.path.join(db_path, "cellpose_train_reps.zip")
        wget_cmd = f"wget --no-check-certificate 'https://docs.google.com/uc?export=download&id={cellpose_drive_id}' -O {cellpose_zip_file}"
        os.system(wget_cmd)
        unzip_cmd = f"unzip {cellpose_zip_file} -d {cellpose_temp_dir}"
        os.system(unzip_cmd)
        shutil.move(os.path.join(cellpose_temp_dir, "cellpose_train-set_representatives"), db_path)
        shutil.rmtree(cellpose_temp_dir, ignore_errors=True)

    cellpose_input_images = [os.path.join(cellpose_dir, x) for x in os.listdir(cellpose_dir) if x.endswith('.png')]
    for input_image_path in cellpose_input_images:
        rn = randomname.get_name(noun=("cats", "dogs", "birds", "fish"))
        cp_id = os.path.splitext(os.path.basename(input_image_path))[0]
        # nickname = f"cellpose_{rn}_{cp_id}"
        nickname = f"cellpose_{cp_id}"
        rdf_dict = {"config" : {'bioimageio' : {'nickname' : nickname}},
                    'test_inputs' : os.path.abspath(input_image_path),
                    "inputs" : [{'axes' : 'yxc'}]}
        with open(os.path.join(db_path, 'rdf_sources', f"{nickname}.yaml"), 'w') as f:
            yaml.dump(rdf_dict, f, default_flow_style=False)
        torch_image = processor.get_torch_image
        input_image = processor.read_image(input_image_path)
        np.save(os.path.join(db_path, "input_images", f"{nickname}.npy"), input_image)
        resized_image = processor.resize_image(input_image, 'yxc', grayscale=grayscale, output_format = "bcyx")
        resized_image = resized_image.astype(np.float32)
        torch_image = torch.from_numpy(resized_image).to(torch.float32)
        all_torch_images.extend([torch_image])
        all_metadata.extend([rdf_dict])
        vec = processor.embed_image(torch_image.numpy(), "bcyx")
        all_embeddings.append(vec)

    db = list(zip(all_torch_images, all_embeddings, all_metadata))
    with open(out_db_path, 'wb') as f:
        pkl.dump(db, f)
    return db


def get_similarity_vec(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # cosine similarity
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_torch_db(
        image_processor: ImageProcessor,
        input_image_path: str, input_image_axes: str,
        db_path: str, top_n: int = 5, verbose : bool = False, 
        force_build : bool = False, grayscale : bool = True) -> str:
    db = get_torch_db(db_path, image_processor, force_build = force_build, grayscale=grayscale)
    user_torch_image = image_processor.get_torch_image(
        input_image_path, input_image_axes, grayscale=grayscale)
    user_embedding = image_processor.embed_image(
        user_torch_image.numpy(), "bcyx")
    if verbose:
        print('Searching DB...')
    db_iterable = tqdm(db) if verbose else db
    sims = [get_similarity_vec(user_embedding, embedding) for (_, embedding, _) in db_iterable]
    hit_indices = sorted(
        range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    out_string = []
    out_string.append("-----------------------------------\nTop Hits\n-----------------------------------")
    similarities = []
    for i_hit, hit_idx in enumerate(hit_indices):
        entry = db[hit_idx][-1]
        nickname = entry['config']['bioimageio']['nickname']
        similarity = sims[hit_idx]
        similarities.append(similarity)
        out_string.append(f"({i_hit}) - {nickname} - similarity: {similarity}\n")
    out_string.append("-----------------------------------")
    if verbose:
        print(' '.join(out_string))
    return [list(db[i]) + [similarities[i_hit]] for i_hit, i in enumerate(hit_indices)]

def get_axes(db_path : str, model_name : str, mislabeled_axes = {'impartial-shark' : 'yx'}):
    if model_name in mislabeled_axes:
        return(mislabeled_axes[model_name])
    with open(os.path.join(db_path, 'rdf_sources', f"{model_name}.yaml"), 'r') as f:
        rdf_dict = yaml.safe_load(f.read())
    input_axes = rdf_dict['inputs'][0]['axes']
    return(input_axes)


def is_channel_first(shape):
    if len(shape) == 5:  # with batch dimension
        shape = shape[1:]
    min_dim = np.argmin(list(shape))
    if min_dim == 0:  # easy case: channel first
        return True
    elif min_dim == len(shape) - 1:  # easy case: channel last
        return False
    else:  # hard case: can't figure it out, just guess channel first
        return True


def get_default_image_axes(shape, input_tensor_axes):
    ndim = len(shape)
    has_z_axis = "z" in input_tensor_axes
    if ndim == 2:
        axes = "yx"
    elif ndim == 3 and has_z_axis:
        axes = "zyx"
    elif ndim == 3:
        channel_first = is_channel_first(shape)
        axes = "cyx" if channel_first else "yxc"
    elif ndim == 4 and has_z_axis:
        channel_first = is_channel_first(shape)
        axes = "czyx" if channel_first else "zyxc"
    elif ndim == 4:
        channel_first = is_channel_first(shape)
        axes = "bcyx" if channel_first else "byxc"
    elif ndim == 5:
        channel_first = is_channel_first(shape)
        axes = "bczyx" if channel_first else "bzyxc"
    else:
        raise ValueError(f"Invalid number of image dimensions: {ndim}")
    return axes


def map_axes(
    input_array,
    input_axes,
    output_axes,
    # spatial axes: drop at middle coordnate, other axes (channel or batch): drop at 0 coordinate
    drop_function=lambda ax_name, ax_len: ax_len // 2 if ax_name in "zyx" else 0
):
    assert len(input_axes) == input_array.ndim, f"Number of axes {len(input_axes)} and dimension of input {input_array.ndim} don't match"
    shape = {ax_name: sh for ax_name, sh in zip(input_axes, input_array.shape)}
    output = DataArray(input_array, dims=tuple(input_axes))
    
    # drop axes not part of the output
    drop_axis_names = tuple(set(input_axes) - set(output_axes))
    drop_axes = {ax_name: drop_function(ax_name, shape[ax_name]) for ax_name in drop_axis_names}
    output = output[drop_axes]
    
    # expand axes missing from the input
    missing_axes = tuple(set(output_axes) - set(input_axes))
    output = output.expand_dims(dim=missing_axes)
    
    # transpose to the desired axis order
    output = output.transpose(*tuple(output_axes))
    
    # return numpy array
    return output.values


def transform_input(image: np.ndarray, image_axes: str, output_axes: str):
    """Transform the input image into an output tensor with output_axes
    
    Args:
        image: the input image
        image_axes: the axes of the input image as simple string
        output_axes: the axes of the output tensor that will be returned
    """
    return map_axes(image, image_axes, output_axes)

def guess_image_axes_deterministic(input_image_files : list[str]):
    guessed_axes = []
    image_processor = ImageProcessor()
    for shape in [image_processor.read_image(fname).shape for fname in input_image_files]:
        axes = []
        common_channel_sizes = [1, 3, 4, 5]  # Common channel sizes are 1 (grayscale), 3 (RGB), and 4 (RGBA)
        shapes_left = sorted([[x, i_x] for i_x, x in enumerate(shape)], key = lambda tup : tup[0], reverse = True, )
        dimensions_left = 'yxcbz'
        def enter(c, axes, shapes_left, dimensions_left):
            axes = [x for x in axes]
            axes.append([c] + shapes_left[0])
            shapes_left = [x for i_x, x in enumerate(shapes_left) if i_x > 0]
            dimensions_left = ''.join([x for x in dimensions_left if x != c])
            return(axes, shapes_left, dimensions_left)
        for c in ['y', 'x']:
            if c in dimensions_left:
                axes, shapes_left, dimensions_left = enter(c, axes, shapes_left, dimensions_left)
        safety_counter = 0
        while len(shapes_left) > 0:
            if shapes_left[0][0] in common_channel_sizes and 'c' in dimensions_left:
                axes, shapes_left, dimensions_left = enter('c', axes, shapes_left, dimensions_left)
            elif 'z' in dimensions_left and len(shapes_left) > 1 and max([x[1] for i_x, x in enumerate(shapes_left) if i_x > 0]) == 1:
                axes, shapes_left, dimensions_left = enter('z', axes, shapes_left, dimensions_left)
            elif 'b' in dimensions_left:
                axes, shapes_left, dimensions_left = enter('b', axes, shapes_left, dimensions_left)
            else:
                axes, shapes_left, dimensions_left = enter('z', axes, shapes_left, dimensions_left)
            if len(dimensions_left) == 0:
                break
            safety_counter += 1
            if safety_counter > 10:
                break
        out_axes = ""
        axes = sorted(axes, key = lambda tup : tup[-1])
        out_axes = ''.join([x[0] for x in axes])
        guessed_axes.append(out_axes)
    return(guessed_axes)

def get_db_inputs(db_path, model_name):
    image_dir = os.path.join(db_path, 'input_images')
    rdf_dir = os.path.join(db_path, 'rdf_sources')
    if np.any([~os.path.exists(x) for x in [image_dir, rdf_dir]]):
        ip = ImageProcessor()
        db = get_torch_db(db_path, ip)
    input_axes = get_axes(db_path, model_name)
    input_image_name = os.path.join(image_dir, f"{model_name}.npy")
    return(input_image_name, input_axes)

async def test_images(input_files : list, out_dir : str, out_suffix : str, db_path: str, 
                mislabeled_axes : dict = {'impartial-shark' : 'yx'}, 
                known_axes : bool = False,
                single_output_file : bool = False,
                num_output_cols : int = 3, llm_axis_guessing = True):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 5
    mpl.rcParams['ytick.labelsize'] = 5
    image_processor = ImageProcessor()
    if single_output_file:
        fig, axes = plt.subplots(len(input_files), 4 + num_output_cols, figsize=(15, 3.2 * len(input_files)))
        out_fig_fname = os.path.join(out_dir, f"{out_suffix}.svg")
    if known_axes:
        all_input_axes = [get_axes(db_path, os.path.basename(fname_abs).replace('.npy', '')) for fname_abs in input_files]
    elif not llm_axis_guessing:
        all_input_axes = guess_image_axes_deterministic(input_files)
    else:
        all_input_axes = await guess_image_axes(input_files)
    for i_fname, fname_abs in enumerate(tqdm(input_files)):
        fname = os.path.basename(fname_abs)
        input_axes = all_input_axes[i_fname]
        if not single_output_file:
            fig, axes = plt.subplots(1, 4 + num_output_cols, figsize = (15, 3.2))
            out_fig_fname = os.path.join(out_dir, f"{out_suffix}_{os.path.splitext(fname)[0]}.svg")
        if len(input_files) == 1 or not single_output_file:
            current_row = axes
        else:
            current_row = axes[i_fname]
        input_image = image_processor.read_image(fname_abs)
        input_image_standardized = image_processor.standardize_image(input_image, input_axes, standard_format='yxc')
        torch_image_grayscale = image_processor.get_torch_image(fname_abs, input_axes, grayscale=True)
        torch_image_color = image_processor.get_torch_image(fname_abs, input_axes, grayscale=False)
        output_image_grayscale = image_processor.standardize_image(torch_image_grayscale.numpy(), input_format = "bcyx", standard_format="yxc")
        output_image_color = image_processor.standardize_image(torch_image_color.numpy(), input_format = "bcyx", standard_format="yxc")
        res = search_torch_db(image_processor, fname_abs, input_axes, db_path, verbose = False, grayscale = True)
        best_hit = res[0][-2]['config']['bioimageio']['nickname']
        best_score = res[0][-1]

        show_images = [(input_image_standardized, f"{fname}"),
                       (output_image_grayscale, "Input (Grayscale)"),
                       (output_image_color, "Input (RGB)")]
        for i_ax in range(3):
            current_row[i_ax].imshow(show_images[i_ax][0])
            current_row[i_ax].set_title(show_images[i_ax][1])
        current_row[3].barh([len(res) - i for i in range(len(res))], [x[-1] for x in res])
        for i, hit in enumerate(res):
            value = hit[-1]
            hit_name = hit[-2]['config']['bioimageio']['nickname']
            current_row[3].set_title(f"Best sim: {best_score:.3f}")
            current_row[3].text(value / 2, len(res) - i, hit_name, ha='center', va='center', rotation='horizontal', color = 'white', size = 6)
        current_row[3].set_yticks([i+1 for i in range(len(res))])
        current_row[3].set_yticklabels([str(len(res) - i) for i in range(len(res))])
        pos_axes2 = current_row[2].get_position()
        pos_axes3 = current_row[3].get_position()
        new_width = pos_axes2.x1 - pos_axes2.x0
        new_height = pos_axes2.y1 - pos_axes2.y0
        height_diff = (pos_axes3.y1 - pos_axes3.y0) - new_height
        new_y0 = pos_axes3.y0 + ( height_diff / 2)
        current_row[3].set_position([pos_axes3.x0, new_y0, new_width, new_height])

        for i, hit in enumerate(res[0 : num_output_cols]):
            hit_name = hit[-2]['config']['bioimageio']['nickname']
            hit_image_path, hit_axes = get_db_inputs(db_path, hit_name)
            if hit_name in mislabeled_axes:
                hit_axes = mislabeled_axes[hit_name]
            hit_image_standardized = image_processor.standardize_image(np.load(hit_image_path), input_format = hit_axes, standard_format = "yxc")
            current_row[4 + i].imshow(hit_image_standardized)
            current_row[4 + i].set_title(f"{hit_name}\n({hit[-1]:.3f})")
        
        if not single_output_file:
            plt.savefig(out_fig_fname, bbox_inches = 'tight')
            plt.close()
    if single_output_file:
        plt.savefig(out_fig_fname, bbox_inches = "tight")
        plt.close()
    return(fig, axes)

def min_distance_to_representative(vector : list, representative_db : list):
    min_distance = np.inf
    for _, _, rep_vector in representative_db:
        distance = scipy.spatial.distance.cosine(vector, rep_vector)
        if distance < min_distance:
            min_distance = distance
    return min_distance    

async def get_representative_images(input_files : list, verbose : bool = False,
                              grayscale : bool = True, random_seed : int = 2010,
                              threshold_distance : float = 0.25, safety_iter : int = 100,
                              llm_axis_guessing = True):
    shapes = []
    image_processor = ImageProcessor()
    embedded_db = []
    input_files_interable = tqdm(input_files) if verbose else input_files
    if verbose:
        print(f"Embedding input images")
    for input_image_path in input_files_interable:
        fname = os.path.basename(input_image_path)
        input_image = image_processor.read_image(input_image_path)
        shapes.append(input_image.shape)
    if llm_axis_guessing:
        guessed_axes = guess_image_axes(input_files)        
    else:
        guessed_axes = guess_image_axes_deterministic(input_files)
    for i_img, input_image_path in enumerate(input_files_interable):
        input_image = image_processor.read_image(input_image_path)
        input_axes = guessed_axes[i_img]
        embedded_image = image_processor.embed_image(input_image, input_axes, grayscale = grayscale)
        embedded_db.append((fname, input_image_path, embedded_image))
    np.random.seed(random_seed)
    representative_db = []
    safety_counter = 0

    # Initialize tqdm for the while loop
    max_safety_count = safety_iter * len(input_files)
    if verbose:
        print(f"Embedded database includes {len(embedded_db)} vectors")
        safety_progress = tqdm(total=max_safety_count, desc="Safety Counter Progress", disable=not verbose)

    # Add vectors to representative_db until the condition is met
    while True:
        # random_vector = choice(embedded_db)
        random_idx = np.random.choice(len(embedded_db), replace = False)
        random_vector = embedded_db[random_idx]
        if min_distance_to_representative(random_vector[2], representative_db) > threshold_distance:
            representative_db.append(random_vector)
        # Check if all vectors in embedded_db have been compared
        if all(min_distance_to_representative(vector[2], representative_db) <= threshold_distance for vector in embedded_db):
            break
        if verbose:
            safety_progress.update(1)
        safety_counter += 1
        if safety_counter > safety_iter * len(input_files):
            if verbose:
                safety_progress.close()
            sys.exit('Error - sampling loop iterations exceeded past safety number')
    return([v[1] for v in representative_db])






class BioengineRunner():
    def __init__(
            self,
            server_url: str = "https://hypha.bioimage.io",
            method_timeout: int = 30,
            ):
        self.server_url = server_url
        self.method_timeout = method_timeout

    async def setup(self):
        from imjoy_rpc.hypha import connect_to_server
        server = await connect_to_server(
            {"name": "client", "server_url": "https://hypha.bioimage.io", "method_timeout": 30}
        )
        self.triton = await server.get_service("triton-client")

    async def bioengine_execute(self, model_id, inputs=None, return_rdf=False, weight_format=None):
        kwargs = {"model_id": model_id, "inputs": inputs, "return_rdf": return_rdf, "weight_format": weight_format}
        ret = await self.triton.execute(
            inputs=[kwargs],
            model_name="bioengine-model-runner",
            serialization="imjoy"
        )
        return ret["result"]

    async def get_model_rdf(self, model_id) -> dict:
        ret = await self.bioengine_execute(model_id, return_rdf=True)
        return ret["rdf"]

    async def shape_check(self, transformed, rdf):
        shape_spec = rdf['inputs'][0]['shape']
        expected_axes = rdf['inputs'][0]['axes']
        if isinstance(shape_spec, list):
            expected_shape = tuple(shape_spec)
            if len(expected_shape) != len(transformed.shape):
                print(
                    f"Transformed input image dimension {len(expected_shape)}"
                    f"does not match the expected dimension ({len(expected_shape)})."
                )
                return False
            for idx in range(len(expected_shape)):
                if expected_axes[idx] == "c":
                    if transformed.shape[idx] != expected_shape[idx]:
                        print(
                            f"Transformed input image channels {transformed.shape[idx]}"
                            f"does not match the expected channels ({expected_shape[idx]})."
                        )
                        return False
        return True

    async def run_model(
            self, image: np.ndarray, model_id: str, rdf: dict,
            image_axes=None, weight_format=None):
        img = image
        input_spec = rdf['inputs'][0]
        input_tensor_axes = input_spec['axes']
        print("input_tensor_axes", input_tensor_axes)
        if image_axes is None:
            shape = img.shape
            image_axes = get_default_image_axes(shape, input_tensor_axes)
            print(f"Image axes were not provided. They were automatically determined to be {image_axes}")
        else:
            print(f"Image axes were provided as {image_axes}")
        assert len(image_axes) == img.ndim
        print("Transforming input image...")
        img = transform_input(img, image_axes, input_tensor_axes)
        print(f"Input image was transformed into shape {img.shape} to fit the model")
        print("Data loaded, running model...")
        if not (await self.shape_check(img, rdf)):
            return False
        try:
            result = await self.bioengine_execute(
                model_id, inputs=[img], weight_format=weight_format)
        except Exception as exp:
            print(f"Failed to run, please check your input dimensions and axes. See the console for more details.")
            print(f"Failed to run the model ({model_id}) in the BioEngine, error: {exp}")
            return False
        if not result['success']:
            print(f"Failed to run, please check your input dimensions and axes. See the console for more details.")
            print(f"Failed to run the model ({model_id}) in the BioEngine, error: {result['error']}")
            return False
        output = result['outputs'][0]
        print(f"🎉Model execution completed! Got output tensor of shape {output.shape}")
        output_tensor_axes = rdf['outputs'][0]['axes']
        transformed_output = map_axes(output, output_tensor_axes, image_axes)
        return transformed_output



if __name__ == "__main__":
    input_img = "./tmp/user_input.jpeg"
    input_axes = "yxc"
    db_path = "./tmp/image_db"
    image_processor = ImageProcessor()
    search_torch_db(image_processor, input_img, input_axes, db_path)
