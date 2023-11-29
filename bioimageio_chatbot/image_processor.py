import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model


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


class ImageProcessor:
    def __init__(self):
        self.vector_embedding = None
        self.image = None
        self.model = VGG16Model()


    def load_image(self, image_path):
        try:
            self.image = np.load(image_path)
        except Exception as e:
            raise Exception(f"Error loading image from {self.image_path}: {str(e)}")
        
    def resize_image(self, input_image_path, current_format, output_format = "byxc"):
        # input_image = np.load(input_image_path)
        self.load_image(input_image_path)
        input_image = self.image
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
    
    def embed_image(self, input_image_path : str, current_format : str, model_format : str = "byxc"):
        resized_image = self.resize_image(input_image_path, current_format, output_format = model_format)
        preprocessed_image = preprocess_input(resized_image)
        vector_embedding = self.model.get_vector_embedding(preprocessed_image)
        self.vector_embedding = vector_embedding
        return(vector_embedding)


    def save_output(self):
        try:
            np.save(self.output_path, self.vector_embedding)
        except Exception as e:
            raise Exception(f"Error saving output to {self.output_path}: {str(e)}")
        

if __name__ == "__main__":
    embedder = ImageProcessor()
    embedded_vector = embedder.embed_image("/Users/gkreder/Downloads/image_db/input_images/chatty-frog.npy", "byxc", "byxc")
    # resized_vector = embedder.resize_image("/Users/gkreder/Downloads/image_db/input_images/chatty-frog.npy", "byxc", "byxc")
