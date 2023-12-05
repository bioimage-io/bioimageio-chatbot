import os
import typing as T
import pickle as pkl

import cv2
import yaml
import requests
import torch
import torch.nn as nn
import numpy as np
from xarray import DataArray
from tqdm.auto import tqdm
from skimage import exposure


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

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embeds an image using the ResNet50 model.
        Args:
            image: A numpy array of shape (b, 3, y, x)
        """
        tensor = torch.from_numpy(image)
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

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embeds an image using the VGG16 model.
        Args:
            image: A numpy array of shape (b, 3, y, x)
        """
        tensor = torch.from_numpy(image)
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
        

    def resize_image(self, input_image, current_format, output_format = "byxc", output_dims_xy = (224,224), grayscale = True):
        # input_image = np.load(input_image_path)
        # self.load_image(input_image_path)
        # input_image = self.image
        current_axes = current_format.lower()
        output_format = output_format.lower()
        if 'b' not in current_axes:
            input_image = np.expand_dims(input_image, axis=0)
            current_axes = 'b' + current_axes
        if 'c' not in current_axes:
            input_image = np.expand_dims(input_image, axis=0)
            current_axes = 'c' + current_axes
        if 'z' in current_axes:
            input_image = np.mean(input_image, current_format.index('z'), keepdims='z' in output_format)
            current_axes = current_axes.replace('z', '')
        output_format_sorted = ''.join(sorted(output_format))
        # assert ''.join(sorted(current_axes)) == output_format_sorted
        assert ''.join(sorted(current_axes)) == 'bcxy'
        index_tup = tuple(current_axes.index(c) for c in 'bcxy') # reshape to bcyx
        input_image = np.transpose(input_image, index_tup)
        current_axes = 'bcxy'
        resized_image = np.array([[cv2.resize(img_xy, output_dims_xy) for img_xy in img_cxy] for img_cxy in input_image])
        if resized_image.shape[0] > 1:  # flatten the 'b' dimension
            resized_image = np.mean(resized_image, axis=0, keepdims='b' in output_format)
        
        channel_index = current_axes.index('c')
        num_channels = resized_image.shape[channel_index]
        channel_colors = [(0,0,1), (0,1,0), (1,0,0), (0,0.5,0.5), (0.5,0,0.5), (0,0.5,0.5)]
        temp_image_shape = [resized_image.shape[current_axes.index('x')], resized_image.shape[current_axes.index('y')]]
        if not grayscale:
            temp_image_shape = [3] + temp_image_shape
        out_image = np.zeros(temp_image_shape)
        for i_c in range(num_channels):
            channel_image = resized_image.take(indices = i_c, axis = channel_index)
            img_adapteq = exposure.equalize_adapthist(channel_image.astype('uint8'), clip_limit=0.01)
            if grayscale:
                out_image += img_adapteq[0,:,:]
            else:
                channel_rgb = np.array(channel_colors[i_c])
                # channel_colormap = create_colormap(channel_rgb, empty_val_rgb='black')
                for i_cc, cc in enumerate(channel_rgb):
                    out_image[i_cc,:,:] += cc * img_adapteq[0,:,:]
        if grayscale:
            out_image = out_image / num_channels
            out_image = np.expand_dims(out_image, axis = 0)
            out_image = np.repeat(out_image, 3, axis = 0)
        current_axes = 'cxy'                
        for c in output_format:
            if c not in current_axes:
                out_image = np.expand_dims(out_image, axis = 0)
                current_axes = c.lower() + current_axes
        for c in current_axes:
            if c not in output_format:
                out_image = np.mean(out_image, current_format.index(c))
                current_axes = current_axes.replace(c, '')
        output_tup = tuple(current_axes.index(c) for c in output_format) # reshape to output format
        resized_image = np.transpose(out_image, output_tup)
        current_axes = output_format.lower()
        return(resized_image)

    def embed_image(self, input_image, current_format: str):
        resized_image = self.resize_image(
            input_image,
            current_format,
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

    def get_torch_image(self, input_image_path, input_axes):
        input_image = self.read_image(input_image_path)
        resized_image = self.resize_image(input_image, input_axes, output_format = "bcyx")
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


def get_model_torch_images(rdf_dict: dict, db_path : str) -> (list, list):
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
        torch_image = image_embedder.get_torch_image(input_image_file, input_axes)
        torch_images.append(torch_image)
        model_metadata.append(rdf_dict)
    return torch_images, model_metadata


def get_torch_db(db_path: str, processor: ImageProcessor, force_build: bool = False) -> list[(torch.Tensor, dict)]:
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
            rdf_dict, db_path)
        all_torch_images.extend(model_torch_images)
        all_metadata.extend(model_metadata)
        for img in model_torch_images:
            vec = processor.embed_image(img.numpy(), "bcyx")
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
        db_path: str, top_n: int = 5) -> str:
    db = get_torch_db(db_path, image_processor)
    user_torch_image = image_processor.get_torch_image(
        input_image_path, input_image_axes)
    user_embedding = image_processor.embed_image(
        user_torch_image.numpy(), "bcyx")
    print('Searching DB...')
    sims = [get_similarity_vec(user_embedding, embedding) for (_, embedding, _) in tqdm(db)]
    hit_indices = sorted(
        range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    print("-----------------------------------\nTop Hits\n-----------------------------------")
    for i_hit, hit_idx in enumerate(hit_indices):
        entry = db[hit_idx][-1]
        nickname = entry['config']['bioimageio']['nickname']
        similarity = sims[hit_idx]
        print(f"({i_hit}) - {nickname} - similarity: {similarity}\n")
    print("-----------------------------------")
    return [db[i] for i in hit_indices]


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
