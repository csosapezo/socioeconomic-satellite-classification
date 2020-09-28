import os

import rasterio

from utils.segmenter import BuildingSegmentation

MODEL_PATH = ".model/model_25_percent_UNet_500epochs.pth"
INPUT_DIR = "data/train/AOI_11_Rotterdam/PS-RGBNIR/"
INPUT = "SN6_Train_AOI_11_Rotterdam_PS-RGBNIR_20190823161806_20190823162129_tile_2747.tif"
UPLOAD_DIRECTORY = "results/"


def main():

    model = BuildingSegmentation(MODEL_PATH)

    data = rasterio.open(os.path.join(INPUT_DIR, INPUT))
    raster = data.read()
    metadata = data.profile

    preprocessed = model.image_loader(raster)
    output = model.predict(preprocessed)

    metadata['count'] = 1
    metadata['height'] = output.shape[1]
    metadata['width'] = output.shape[2]
    metadata['dtype'] = output.dtype
    mask_filename = INPUT[:-4] + "_MASK.TIF"

    if not os.path.exists(UPLOAD_DIRECTORY):
        os.mkdir(UPLOAD_DIRECTORY)

    with rasterio.open(os.path.join(UPLOAD_DIRECTORY, mask_filename), 'w', **metadata) as outds:
        outds.write(output)


if __name__ == "__main__":
    main()
