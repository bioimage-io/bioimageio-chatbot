import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from xarray import DataArray
from tqdm.auto import tqdm
from PIL import Image
import xml.etree.ElementTree as ET

import asyncio
from schema_agents.role import Role
from schema_agents.schema import Message
from pydantic import BaseModel, Field
from typing import List

from bioimageio_chatbot.image_processor import *


class SearchAndRunModelByImage(BaseModel):
    """Search bioimage model zoo for the best model to analyze user's image and run the model according to user's request"""
    request: str = Field(description="The user's request")

class InputImageShape(BaseModel):
    """"The shape of the image"""
    image_shape: list[int] = Field(description = "The shape of the input image represented as a tuple. Each entry is an image dimension's size")
    # image_dtype: str = Field(description = "The dtype of the input image.")
    # image_filename: str = Field(description = "The filename of the input image.")

class ImageAxes(BaseModel):
    """The axes of the image."""
    axes: list[str] = Field(description="A tuple of characters representing the axes of the image (for example ('c','y','x'), ('b','x','y','c')). Each entry corresponds to the axis label of the image along the corresponding dimension. 't' is the least likely axis to appear in any image")

class InputImageShapes(BaseModel):
    """A list of input image shapes"""
    shapes: list[InputImageShape] = Field(description = "The list of input image shapes")

class GuessedAxes(BaseModel):
    """The estimated axes labels for each image shape provided"""
    guessed_axes: list[ImageAxes] = Field(description = "The outputted list of estimated image axes labels. Each element (guessed axes for a single image) MUST have the same length as the corresponding shape tuple. And it must always contains an assignment for batch (b), x, and y")

async def guess_image_axes(input_image_description : InputImageShapes, role: Role = None) -> GuessedAxes:
      """Guesses the axis labels given the provided image descriptions (shapes). The lengths of the axis label strings must match the length of each image shape tuple. Use your common sense as to which axis label should be assigned in which position based on the values in the tuple."""
      response = await role.aask(input_image_description, GuessedAxes)
      return(response)

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
    headers = ["Image", "True Shape", "True Axes", "Guessed Axes"]
    x_position = 0
    for i, header in enumerate(headers):
        add_text(svg, x_position + 5, header_height - 10, header, font_size='14')
        x_position += column_widths[i]
    # Add rows for each image
    for i, (img, shape, true_ax, guessed_ax) in enumerate(zip(input_images, true_shapes, true_axes, guessed_axes)):
        y_position = header_height + i * row_height
        row_color = '#90ee90' if true_ax == guessed_ax else '#ffcccb'  # Lighter shades of green and red
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
                actions=[guess_image_axes])
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
    # desc = InputImageDescription(image_shape=input_image.shape)
    print(true_shapes)
    message_input = InputImageShapes(shapes = [InputImageShape(image_shape = x) for x in true_shapes])
    m = Message(content = 'guess the image axes for each image in the list', data = message_input, role = 'User')
    responses = await customer_service.handle(m)
    guessed_axes = [''.join(x.axes) for x in responses[0].data.guessed_axes]
    svg_content = create_svg_table(input_images, true_shapes, true_axes, guessed_axes)
    with open(svg_output_fname, "w") as file:
        file.write(svg_content)
    return(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # f = await main()
    svg_output_fname = "output.svg"
    loop.create_task(main(svg_output_fname))
    loop.run_forever()
    # loop.run_until_complete(f)