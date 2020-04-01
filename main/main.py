from functions import LowQualityData
from utils.comparators import *
# -----------------------------------------------#
# -----------------------------------------------#
# -----------     DATA LOADING     --------------#
# -----------------------------------------------#
# -----------------------------------------------#

# Datos con ruido
f = open("..\\resources\\data\\50sin-noise.dat")
datos = f.readlines()
f.close()

maxdatos = len(datos)

# Test sobre datos sin ruido
f = open("..\\resources\\data\\50sin-noiseless.dat")
datostrain = f.readlines()
f.close()

# -----------------------------------------------#
# -----------------------------------------------#
# ---------     DATA PROCESSING     -------------#
# -----------------------------------------------#
# -----------------------------------------------#
# procesar la lista de lineas
for i in range(maxdatos):
	datos[i] = datos[i].split(' ')
	# convertir cada valor en float
	for j in range(len(datos[i])):
		datos[i][j] = float(datos[i][j])

datos = array(datos)

# Conjunto de datos de entrenamiento
for i in range(maxdatos):
	datostrain[i] = datostrain[i].split(' ')
	for j in range(len(datostrain[i])):
		datostrain[i][j] = float(datostrain[i][j])

datostrain = array(datostrain)


# -----------------------------------------------#
# -----------------------------------------------#
# -----------     CREATE MODEL     --------------#
# -----------------------------------------------#
# -----------------------------------------------#

modelo = LowQualityData.LowQualityData()
modelo.setLabel("comparaMSE")
# -----------------------------------------------#
# -----------------------------------------------#
# -----------      FIT MODEL       --------------#
# -----------------------------------------------#
# -----------------------------------------------#
modelo.fit(datostrain, comparaMSE)

# -----------------------------------------------#
# -----------------------------------------------#
# -----------     PREDICT MODEL    --------------#
# -----------------------------------------------#
# -----------------------------------------------#
resultado = modelo.predict(datos)

print("\n-- Resultados--\n")
print(resultado)
