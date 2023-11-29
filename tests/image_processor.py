import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from bioimageio_chatbot.chatbot import load_model_info
import yaml
import requests
import os
import torch
import lpips
from tqdm.auto import tqdm
import pickle as pkl

class VGG16Model:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

    def get_vector_embedding(self, image: np.ndarray, verbose = 0):
        try:
            vector_embedding = self.model.predict(image, verbose = verbose)
            return vector_embedding
        except Exception as e:
            raise Exception(f"Error getting vector embedding: {str(e)}")


class ImageProcessor():
    def __init__(self):
        self.vector_embedding = None
        self.image = None
        self.model = VGG16Model()


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
        if resized_image.shape[0] > 1: # flatten the 'b' dimension
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
        return(resized_image)

    
    def embed_image(self, input_image, current_format : str, model_format : str = "byxc"):
        resized_image = self.resize_image(input_image, current_format, output_format = model_format)
        preprocessed_image = preprocess_input(resized_image)
        vector_embedding = self.model.get_vector_embedding(preprocessed_image)
        self.vector_embedding = vector_embedding
        return(vector_embedding)

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
        resized_image = self.resize_image(input_image, input_axes, output_format = "bcyx") # https://github.com/richzhang/PerceptualSimilarity
        preprocessed_image = preprocess_input(resized_image)
        torch_image = torch.from_numpy(preprocessed_image.copy()).to(torch.float32)
        return(torch_image)
    
def get_model_rdf(m : dict, db_path : str) -> dict | None:
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
        with open(yaml_name, 'r') as f:
            rdf_source_content = f.read()
    rdf_dict = yaml.safe_load(rdf_source_content)
    try:
        test_inputs = rdf_dict['test_inputs']
        input_axes = rdf_dict['inputs'][0]['axes']
        with open(yaml_name, 'w') as f:
            f.write(rdf_source_content)
        return(rdf_dict)
    except:
        return(None)
    
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
    return(torch_images, model_metadata)


def get_torch_db(db_path : str) -> list[(torch.Tensor, dict)]:
    out_db_path = os.path.join(db_path, 'db.pkl')
    if os.path.exists(out_db_path):
        db = pkl.load(open(out_db_path, 'rb'))
        return(db)
    models = load_model_info()
    all_metadata = []
    all_torch_images = []
    print('Creating PyTorch image database from model zoo...')
    for i_m, m in enumerate(tqdm(models)):
        if 'nickname' not in m:
            continue
        rdf_dict = get_model_rdf(m, db_path)
        if not rdf_dict:
            continue
        model_torch_images, model_metadata = get_model_torch_images(rdf_dict, db_path)
        all_torch_images.extend(model_torch_images)
        all_metadata.extend(model_metadata)
    db = list(zip(all_torch_images, all_metadata))
    with open(out_db_path, 'wb') as f:
        pkl.dump(db, f)
    return(db)

def get_similarity(img1 : torch.Tensor, img2 : torch.Tensor) -> float:
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False)
    diff = loss_fn_vgg(img1, img2)
    diff = diff.detach().numpy()[0][0][0][0]
    sim = 1 - diff # Closer to 1 = more similar
    return(sim)

def search_torch_db(input_image_path : str, input_image_axes : str, db_path : str, top_n : int = 5) -> str:
    db = get_torch_db(db_path)
    image_embedder = ImageProcessor()
    user_torch_image = image_embedder.get_torch_image(input_image_path, input_image_axes)
    print('Searching DB...')
    sims = [get_similarity(user_torch_image, img) for (img,_) in tqdm(db)]
    hit_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    print("-----------------------------------\nTop Hits\n-----------------------------------")
    for i_hit, hit_idx in enumerate(hit_indices):
        entry = db[hit_idx][1]
        nickname = entry['config']['bioimageio']['nickname']
        print(f"({i_hit}) - {nickname}\n")
    print("-----------------------------------")

if __name__ == "__main__":
    # embedder = ImageProcessor()
    # embedded_vector = embedder.embed_image("/Users/gkreder/Downloads/image_db/input_images/chatty-frog.npy", "byxc", "byxc")
    # resized_vector = embedder.resize_image("/Users/gkreder/Downloads/image_db/input_images/chatty-frog.npy", "byxc", "byxc")

    # image_embedder = ImageProcessor()
    # user_img_path = "/Users/gkreder/Downloads/content.png"
    # user_img = cv2.imread(user_img_path)
    # user_img_axes = "yxc"
    # print(image_embedder.resize_image(user_img, user_img_axes))
    # db_path = create_torch_db("/Users/gkreder/Downloads/image_db")
    # input_img = "/Users/gkreder/Downloads/microscopy-fruit-fly-neurosciencenews.jpeg"
    input_img = "/Users/gkreder/Downloads/nuclear_blue_image.jpg"
    input_axes = "yxc"
    db_path = "/Users/gkreder/Downloads/image_db"
    search_torch_db(input_img, input_axes, db_path)
    
