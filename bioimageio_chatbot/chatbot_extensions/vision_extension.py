from bioimageio_chatbot.utils import ChatbotExtension
from openai import AsyncOpenAI
from schema_agents import schema_tool
import base64
from pydantic import Field, BaseModel
import httpx
from PIL import Image
from io import BytesIO

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

async def aask(images, messages, max_tokens=1024):
    aclient = AsyncOpenAI()
    user_message = []
    # download the images and save it into a list of PIL image objects
    img_objs = []
    for image in images:
        async with httpx.AsyncClient() as client:
            response = await client.get(image.url)
            response.raise_for_status()
        try:
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to read image: {e}")
        img_objs.append(img)
    
    # plot them in subplots with matplotlib in a row
    fig, ax = plt.subplots(1, len(img_objs), figsize=(15, 5))
    for i, img in enumerate(img_objs):
        ax[i].imshow(img)
        ax[i].set_title(images[i].title)
        ax[i].axis("off")
    # save the plot to a buffer as png format and convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    # append the image to the user message
    user_message.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
        }
    })
    
    for message in messages:
        assert isinstance(message, str), "Message must be a string."
        if message.startswith("http"):
            # # download the image and encode it to base64
            # async with httpx.AsyncClient() as client:
            #     response = await client.get(message)
            #     response.raise_for_status()
            # # read the image and resize it to make sure the maximum size is 512
            # try:
            #     img = Image.open(BytesIO(response.content))
            # except Exception as e:
            #     raise ValueError(f"Failed to read image: {e}")
            # if max(img.size) > 512:
            #     scale = 512 / max(img.size)
            #     # resize the image
            #     resized = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
            #     # save the image to a buffer
            #     buffer = BytesIO()
            #     resized.save(buffer, format="PNG")
            #     buffer.seek(0)
            #     base64_image = base64.b64encode(buffer.read()).decode("utf-8")
            #     mime = "image/png"
            # else:
            #     base64_image = base64.b64encode(response.content).decode("utf-8")
            # # base64_image = f"data:image/...;base64,{base64_image}"
            #     mime = response.headers.get("Content-Type", "image/png")
            # base64_image = f"data:{mime};base64,{base64_image}"
            user_message.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            })
        else:
            user_message.append({"type": "text", "text": message})

    response = await aclient.chat.completions.create(
        model="gpt-4-1106-vision-preview",
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

class ImageInfo(BaseModel):
    """Image information."""
    url: str=Field(..., description="The URL of the image.")
    title: str=Field(..., description="The title of the image.")

@schema_tool
async def inspect_tool(images: list[ImageInfo]=Field(..., description="A list of images to be inspected, each with a http url and title"), query: str=Field(..., description="user query about the image"),  context_description: str=Field(..., description="describe the context for the visual inspection task")) -> str:
    """Inspect an image using GPT4-Vision."""
    # assert image_url.startswith("http"), "Image URL must start with http."
    for image in images:
        assert image.url.startswith("http"), "Image URL must start with http."
    
    response = await aask(images, [context_description, query])
    return response

def get_extension():
    return ChatbotExtension(
        id="vision",
        name="Vision Inspector",
        description="Inspect an image using GPT4-Vision, used for inspect the details or ask questions about the image.",
        tools=dict(
            inspect=inspect_tool
        )
    )

if __name__ == "__main__":
    import asyncio
    async def main():
        extension = get_extension()
        # print(await extension.tools["inspect"](image_url="https://bioimage.io/static/img/bioimage-io-icon.png", query="What is this?"))
        print(await extension.tools["inspect"](image_url="https://bioimage.io/static/img/bioimage-io-logo.png", query="What is this?"))

    # Run the async function
    asyncio.run(main())