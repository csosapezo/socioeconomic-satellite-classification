import sqlite3

connR = sqlite3.connect(':memory:')

connR.enable_load_extension(True)

connR.execute('SELECT load_extension("mod_spatialite")')
connR.execute('SELECT InitSpatialMetaData(1);')

# libspatialite
connR.execute('SELECT load_extension("libspatialite")')
connR.execute('SELECT InitSpatialMetaData();')

# open database
connR.execute('.open planos.sqlite')

print("Introduza el área a buscar")
start_x = input("X inicial:")
offset_x = input("offset:")
start_y = input("Y inicial:")
offset_y = input("offset:")

end_x = start_x + offset_x
end_y = start_y + offset_y

search_polygon = 'POLYGON({0} {1}, {0} {2}, {3} {1}, {3} {2}, {0} {1})'.format(start_x, start_y, end_y, end_x)

select = 'SELECT AsText(GeomFromWKB(GEOMETRY))'
from_where = 'FROM planos WHERE ST_Intersects(GeomFromText(search_polygon)GeomFromWKB(GEOMETRY)) = 1 '

connR.execute(select + ' ' + from_where)