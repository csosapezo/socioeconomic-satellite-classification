from utils import preprocess_image

UPLOAD_DIRECTORY = 'static/'

import utils.model_utils as model_utils
from models.unet import UNet


class BuildingSegmentation(object):
    def __init__(self, model_path):
        self.model = model_utils.load_model(model_path, UNet())

    def image_loader(self, img):
        """Preprocesa una imagen para ser apta para entrar en el modelo de segmentación."""
        img = preprocess_image(img)
        return img

    def predict(self, image):
        """Procesa una imagen satelital, previamente preprocesada, mediante un modelo de segmentación y devuelve una
        máscara que señala los cuerpos de agua de la imagen original. """

        img_input = self.image_loader(image)
        trained_model = self.model
        response = model_utils.run_model(img_input, trained_model)
        del trained_model
        return response
