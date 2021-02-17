import sqlite3

connR = sqlite3.connect('planos.sqlite')

connR.enable_load_extension(True)

connR.execute('SELECT load_extension("mod_spatialite.so")')
connR.execute('SELECT InitSpatialMetaData(1);')

print("Introduza el área a buscar")
start_x = input("X inicial:")
offset_x = input("offset:")
start_y = input("Y inicial:")
offset_y = input("offset:")

end_x = start_x + offset_x
end_y = start_y + offset_y

search_polygon = 'POLYGON({0} {1}, {0} {2}, {3} {1}, {3} {2}, {0} {1})'.format(start_x, start_y, end_y, end_x)

select = 'SELECT AsText(GeomFromWKB(GEOMETRY))'
from_where = 'FROM planos WHERE ST_Intersects(GeomFromText({}), GeomFromWKB(GEOMETRY)) = 1 '.format(search_polygon)
query = select + ' ' + from_where

print("Consulta:")
print(query)
input()

connR.execute(query)