import sqlite3

connR = sqlite3.connect('planos.sqlite')

connR.enable_load_extension(True)

try:
    connR.execute('SELECT load_extension("mod_spatialite")')
except sqlite3.OperationalError:
    connR.execute('SELECT load_extension("mod_spatialite.so")')

connR.execute('SELECT InitSpatialMetaData(1);')

print("Introduza el área a buscar")
start_x = input("X inicial:")
end_x = input("X final:")
start_y = input("Y inicial:")
end_y = input("Y final:")

search_polygon = 'POLYGON(({0} {1}, {0} {2}, {3} {1}, {3} {2}, {0} {1}))'.format(start_x, start_y, end_y, end_x)

select = 'select ESTRATO, ST_Intersects(GeomFromText("{}"), GeometryN(GeomFromWKB(GEOMETRY),1)) as booll, AsText(GeometryN(' \
         'GeomFromWKB(GEOMETRY),1))'.format(search_polygon)
from_where = 'from planos where booll = 1;'
query = select + ' ' + from_where

print("Consulta:")
print(query)
input()

print(connR.execute(query).fetchall()[:5])
