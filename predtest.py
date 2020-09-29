import os

import numpy as np
import rasterio

from utils.segmenter import BuildingSegmentation

MODEL_PATH = ".model/model_25_percent_UNet_500epochs.pth"
INPUT_DIR = "data/train/split/"
UPLOAD_DIRECTORY = "results/"


def main():

    model = BuildingSegmentation(MODEL_PATH)

    names = os.listdir(INPUT_DIR)

    for name in names:
        data = rasterio.open(os.path.join(INPUT_DIR, name))
        raster = data.read()
        metadata = data.profile

        output = model.predict(raster).data.cpu().numpy()[0]

        metadata['count'] = 1
        metadata['height'] = output.shape[1]
        metadata['width'] = output.shape[2]
        print(f"min: {output.min()} max: {output.max()}")
        metadata['dtype'] = output.dtype
        mask_filename = name[:-4] + "_MASK.TIF"

        if not os.path.exists(UPLOAD_DIRECTORY):
            os.mkdir(UPLOAD_DIRECTORY)

        with rasterio.open(os.path.join(UPLOAD_DIRECTORY, mask_filename), 'w', **metadata) as outds:
            outds.write(np.rint(output))


if __name__ == "__main__":
    main()
