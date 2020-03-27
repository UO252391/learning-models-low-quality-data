from functions.fuzzy_rule_based import myFRBS
from functions.genetic import genetico
from functions.delta import deltacrisp
from utils.comparators import *
from utils.FRBS_plot import FRBS_plot

#-----------------------------------------------#
#-----------------------------------------------#
#-----------     DATA LOADING     --------------#
#-----------------------------------------------#
#-----------------------------------------------#
f = open("..\\resources\\data\\50sin-noise.dat")
datos = f.readlines()
f.close()

# Eliminamos muestras si es necesario para hacer pruebas rapidas
maxdatos = len(datos)

# procesar la lista de lineas
for i in range(maxdatos):
	datos[i] = datos[i].split(' ')
	# convertir cada valor en float
	for j in range(len(datos[i])):
		datos[i][j] = float(datos[i][j])

datos = array(datos)

# Test sobre datos sin ruido
f = open("..\\resources\\data\\50sin-noiseless.dat")
datostest = f.readlines()
f.close()
for i in range(maxdatos):
	datostest[i] = datostest[i].split(' ')
	for j in range(len(datostest[i])):
		datostest[i][j] = float(datostest[i][j])


#-----------------------------------------------#
#-----------------------------------------------#
#-----------  PROGRAM CONSTANTS   --------------#
#-----------------------------------------------#
#-----------------------------------------------#
datostest = array(datostest)
mygran = 10  # Granulalidad
mydimx = maxdatos  # Numero de instancias
NUMPAR = 3 * mygran
params = np.zeros([NUMPAR])
params[0:mygran] = 1  # Consecuentes de las reglas borrosas
params[mygran:2 * mygran] = [x * mydimx / (mygran - 1) for x in range(mygran)]
params[2 * mygran:3 * mygran] = [2 for x in range(mygran)]
NITER = 50
POP = 50
c = 0.25


#-----------------------------------------------#
#-----------------------------------------------#
#-----------  ESTIMATE PREDICTION --------------#
#-----------------------------------------------#
#-----------------------------------------------#
prediccion = myFRBS(datos[:, 0], params[0:mygran], params[mygran:2 * mygran], params[2 * mygran:3 * mygran], mygran,
						mydimx)
delta = prediccion - datos[:, 1]

print("\nNumero de puntos cubiertos:", np.percentile(delta, array([5, 95])))

print("\nCalculando soluciones...")


#-----------------------------------------------#
#-----------------------------------------------#
#-----------      COMPARAMSE      --------------#
#-----------------------------------------------#
#-----------------------------------------------#

solucion1 = genetico(datos, mygran, mydimx, POP, params, deltacrisp, comparaMSE, datos[:, 1], NITER, c)
FRBS_plot("comparaMSE")
print("\n\nSolucion 1\n")
print(solucion1)


#-----------------------------------------------#
#-----------------------------------------------#
#-----------    COMPARASTATPREF    -------------#
#-----------------------------------------------#
#-----------------------------------------------#
solucion2 = genetico(datos, mygran, mydimx, POP, params, deltacrisp, comparaStatPref, datos[:, 1], NITER, c)
FRBS_plot("comparaStatPref")
print("\n\nSolucion 2\n")
print(solucion2)


#-----------------------------------------------#
#-----------------------------------------------#
#-----------   COMPARASTOCHDOM    --------------#
#-----------------------------------------------#
#-----------------------------------------------#
solucion3 = genetico(datos, mygran, mydimx, POP, params, deltacrisp, comparaStochDom, datos[:, 1], NITER, c)
FRBS_plot("comparaStochDom")
print("\n\nSolucion 3\n")
print(solucion3)


