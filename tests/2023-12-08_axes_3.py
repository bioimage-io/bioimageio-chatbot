import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from xarray import DataArray
from PIL import Image
import xml.etree.ElementTree as ET
from schema_agents.provider.openai_api import retry

import asyncio
from schema_agents.role import Role
from schema_agents.schema import Message
from pydantic import BaseModel, Field, validator

from bioimageio_chatbot.image_processor import *


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

async def guess_image_axes(image : UnlabeledImage, role : Role = None) -> LabeledImage:
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

async def guess_all_axes(unlabeled_images : UnlabeledImages, role : Role = None) -> LabeledImages:
    """Labels the axes in all images in the input list of unlabeled images"""
    labeled_images = []
    for unlabeled_image in unlabeled_images.unlabeled_images:
        labeled_image = await role.aask(unlabeled_image, LabeledImage)
        labeled_images.append(labeled_image)
    
    guessing_tasks = (retry_aask(role, ui, LabeledImage) for ui in unlabeled_images.unlabeled_images)
    labeled_images = await asyncio.gather(*guessing_tasks)
    labeled_images = LabeledImages(labeled_images=labeled_images)
    return(labeled_images)

def create_svg_table(input_images, true_shapes, true_axes, guessed_axes):
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

async def main(svg_output_fname : str):
    db_path = "../tmp/image_db/"
    image_dir = "../tmp/image_db/input_images/"

    customer_service = Role(name = "Melman",
                    profile = "Customer Service",
                    goal="Your goal as Melman from Madagascar, the community knowledge base manager, is to assist users in effectively utilizing the knowledge base for bioimage analysis. You are responsible for answering user questions, providing clarifications, retrieving relevant documents, and executing scripts as needed. Your overarching objective is to make the user experience both educational and enjoyable.",
                constraints=None,
                actions=[guess_image_axes, guess_all_axes])
    event_bus = customer_service.get_event_bus()
    event_bus.register_default_events()
    input_images = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.npy')]
    image_processor = ImageProcessor()
    guessed_axes = []
    true_axes = []
    true_shapes = []
    for input_image_path in input_images:
        model_name = os.path.splitext(os.path.basename(input_image_path))[0]
        input_axes = get_axes(db_path, model_name)
        input_image = image_processor.read_image(input_image_path)
        true_axes.append(input_axes)
        true_shapes.append(input_image.shape)
    message_input = UnlabeledImages(unlabeled_images = [UnlabeledImage(shape = s) for s in true_shapes])
    m = Message(content = 'guess the image axes for each image in the list', data = message_input, role = 'User')
    responses = await customer_service.handle(m)
    guessed_axes = [''.join(x.axes.labels) for x in responses[0].data.labeled_images]
    svg_content = create_svg_table(input_images, true_shapes, true_axes, guessed_axes)
    with open(svg_output_fname, "w") as file:
        file.write(svg_content)
    print(responses)
    return(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    svg_output_fname = "model-zoo_guess-results.svg"
    loop.create_task(main(svg_output_fname))
    loop.run_forever()