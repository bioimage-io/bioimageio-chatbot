import os
import typing as T
import pickle as pkl

import cv2
import yaml
import requests
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm



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

    def resize_image(self, input_image, current_format, output_format = "byxc"):
        # input_image = np.load(input_image_path)
        # self.load_image(input_image_path)
        # input_image = self.image
        current_axes = current_format.lower()
        if 'b' not in current_axes:
            input_image = np.expand_dims(input_image, axis=0)
            current_axes = 'b' + current_axes
        if 'c' not in current_axes:
            input_image = np.expand_dims(input_image, axis=0)
            current_axes = 'c' + current_axes
        if 'z' in current_axes:
            input_image = np.mean(input_image, current_format.index('z'))
            current_axes = current_axes.replace('z', '')
        assert ''.join(sorted(current_axes)) == 'bcxy'
        index_tup = tuple(current_axes.index(c) for c in 'bcyx') # reshape to bcyx
        input_image = np.transpose(input_image, index_tup)
        current_axes = 'bcyx'
        resized_image = np.array([[cv2.resize(img_xy, (224, 224)) for img_xy in img_cxy] for img_cxy in input_image])
        if resized_image.shape[0] > 1:  # flatten the 'b' dimension
            resized_image = np.mean(resized_image, axis=0, keepdims=True)
        num_channels = resized_image.shape[1]
        if num_channels == 1:
            # Grayscale to RGB: Repeat the single channel 3 times
            resized_image = np.repeat(resized_image, 3, axis=1)
        elif num_channels == 2:
            # If there are 2 channels, add a third channel by repeating one of the existing ones
            # Adding a third channel by repeating the second channel
            resized_image = np.concatenate((resized_image, resized_image[:, 1:2, :, :]), axis=1)
        elif num_channels > 3:
            # If there are more than 3 channels, select the first 3 channels
            resized_image = resized_image[:, :3, :, :]
        output_tup = tuple(current_axes.index(c) for c in output_format) # reshape to output format
        resized_image = np.transpose(resized_image, output_tup)
        return resized_image

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

    def get_torch_image(self, input_image_path, input_axes):
        if input_image_path.endswith('.npy'):
            input_image = np.load(input_image_path)
        else:
            input_image = cv2.imread(input_image_path)
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
        input_image_path: str, input_image_axes: str,
        db_path: str, top_n: int = 5) -> str:
    image_processor = ImageProcessor()
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


if __name__ == "__main__":
    input_img = "./tmp/content.png"
    input_axes = "yxc"
    db_path = "./tmp/image_db"
    search_torch_db(input_img, input_axes, db_path)
