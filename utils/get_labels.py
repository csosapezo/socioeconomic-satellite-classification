import sqlite3

from pyproj import Transformer

import utils


def get_bounds(metadata):
    # affine transform
    transform = metadata['transform']

    # get patch boundaries: (x0, y0) , (x1, y1)
    x_ini, y_ini = transform * (0, 0)
    x_fin, y_fin = transform * (metadata['width'], metadata['height'])

    # transform from source crs to LonLat (epsg:4326)
    transformer = Transformer.from_crs(metadata['crs'], utils.constants.crs_lonlat)
    left, bottom = transformer.transform(x_ini, y_ini)
    right, top = transformer.transform(x_fin, y_fin)

    return utils.constants.polygon_format.format(left, bottom, top, right)


def search_labels(bounds, database_filename):
    connection = sqlite3.connect(database_filename)
    connection.enable_load_extension(True)

    try:
        connection.execute('SELECT load_extension("mod_spatialite")')  # Ubuntu 20.04 LTS
    except sqlite3.OperationalError:
        connection.execute('SELECT load_extension("mod_spatialite.so")')  # Ubuntu 18.04 LTS

    connection.execute('SELECT InitSpatialMetaData(1);')

    query = utils.constants.query.format(bounds)
    raw_labels = connection.execute(query).fetchall()

    return raw_labels


def get_levels(database_filename):
    connection = sqlite3.connect(database_filename)
    connection.enable_load_extension(True)

    try:
        connection.execute('SELECT load_extension("mod_spatialite")')  # Ubuntu 20.04 LTS
    except sqlite3.OperationalError:
        connection.execute('SELECT load_extension("mod_spatialite.so")')  # Ubuntu 18.04 LTS

    connection.execute('SELECT InitSpatialMetaData(1);')
    levels = connection.execute(utils.constants.distinct_query)

    level_dict = {}

    for idx, level in enumerate(levels):
        level_dict[str(level[0])] = idx

    return level_dict


def build_polygon_dict(geometry, metadata):
    data_ini_index = geometry.rfind(utils.constants.polygon_format_left) + 1
    data_end_index = geometry.find(utils.constants.polygon_format_right)
    geometry = geometry[data_ini_index:data_end_index]

    raw_coordinates_list = geometry.split(sep=utils.constants.point_splitter)
    geometry_dict = {"type": "polygon", "coordinates": []}
    coordinates_list = []

    for raw_coord in raw_coordinates_list:
        string_coord = raw_coord.split()
        coord = [float(i) for i in string_coord]

        transformer = Transformer.from_crs(utils.constants.crs_lonlat, metadata['crs'])
        x, y = transformer.transform(coord[0], coord[1])
        coord = [x, y]

        coordinates_list.append(coord)

    geometry_dict["coordinates"].append(coordinates_list)

    return geometry_dict


def extract_labels(raw_labels, metadata):
    labels = {}

    for label in raw_labels:
        processed_label = build_polygon_dict(label[utils.constants.geometry], metadata)
        income_level = str(labels[utils.constants.income_level])

        if income_level in labels:
            labels[income_level].append(processed_label)
        else:
            geometry_list = [processed_label]
            labels[income_level] = geometry_list

    return labels


def get_labels(metadata, database_filename):
    bounds = get_bounds(metadata)
    raw_labels = search_labels(bounds, database_filename)
    labels = extract_labels(raw_labels, metadata)

    return labels
