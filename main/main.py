from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
modelo.setLabel("comparaStochDom")
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
ajuste = modelo.adjust(10)

regressor = RandomForestRegressor()
regressor.fit(datostrain, datostrain[:, 1])

rf_result = regressor.predict(datos)

mse_rf = mse = mean_squared_error(datostrain[:, 1], rf_result)

mse1 = modelo.mse(datostrain[:, 1], resultado)
mse2 = modelo.mse(datostrain[:, 1], ajuste)

print("Error cuadratico medio FRBS: {0}".format(mse1))
print("Error cuadratico medio FRBS (ajuste): {0}".format(mse2))
print("Error cuadratico medio RF: {0}".format(mse_rf))

modelo.save()
print("\n-- Valores reales --\n")
print(datostrain)

print("\n-- Resultados tras ajuste--\n")
print(ajuste)
