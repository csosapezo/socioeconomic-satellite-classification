# cropped patches
width = 512
height = 512
patch_suffix = "_subtile_{}-{}"
roof_suffix = "_ROOF_MASK"
income_suffix = "_INCOME_MASK"
dot_tif = ".tif"
dot_npy = ".npy"

# cropped patches zero sum percentage
max_equals0 = 0.005

# Training values
pixel_max_value = 3512
num_channel = 4

# patch bounds
crs_lonlat = 'epsg:4326'
polygon_format = 'POLYGON(({0} {1}, {0} {2}, {3} {2}, {3} {1}, {0} {1}))'

# SQL Query
database_file = 'planos.sqlite'
select = 'select ESTRATO, AsText(GeometryN(GeomFromWKB(GEOMETRY),1))'
from_ = 'from planos '
where = 'where ST_Intersects(GeomFromText("{}"), GeometryN(GeomFromWKB(GEOMETRY),1)) = 1;'
query = select + ' ' + from_ + '' + where
distinct = 'select distinct ESTRATO'
distinct_query = distinct + ' ' + from_

# Query result structure
income_level = 0
geometry = 1

# Polygon result structure
polygon_format_left = '('
polygon_format_right = ')'
point_splitter = ','

offset_values = {
    "IMG_PER1_20190217152904_ORT_P_000659.TIF": (-11.56, 38.44),
    "IMG_PER1_20200429151954_ORT_P_000054.TIF": (-25.53, 67.3),
    "IMG_PER1_20200415152730_ORT_P_000294.TIF": (-1.62, 11.03),
    "IMG_PER1_20200415152730_ORT_P_000054.TIF": (11.2, 78.4),
    "IMG_PER1_20200406154450_ORT_P_000944.TIF": (-229.8, 81.8),
    "IMG_PER1_20200406154450_ORT_P_000672.TIF": (-142.1, 58.6),
    "IMG_PER1_20190217152904_ORT_P_000041.TIF": (-15.34, 36.33),
    "IMG_PER1_20161217152156_ORT_P_000476.TIF": (25.25, -32.8)
}
