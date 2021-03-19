# cropped patches shape
width = 256
height = 256

# cropped patches zero sum percentage
max_equals0 = 0.005

# Training values
train_val_split = 0.25
pixel_max_value = 3512
num_channel = 4

# patch bounds
crs_lonlat = 'epsg:4326'
polygon_format = 'POLYGON(({0} {1}, {0} {2}, {3} {1}, {3} {2}, {0} {1}))'

# SQL Query
database_file = 'planos.sqlite'
select = 'select ESTRATO, AsText(GeometryN(GeomFromWKB(GEOMETRY),1))'
from_ = 'from planos'
where = 'where ST_Intersects(GeomFromText("{}"), GeometryN(GeomFromWKB(GEOMETRY),1)) = 1;'
query = select + ' ' + from_ + '' + where

# Query result structure
income_level = 0
geometry = 1

# Polygon result structure
polygon_format_left = '('
polygon_format_right = ')'
point_splitter = ','
