import numpy as np
from functions.delta import deltacrisp
from functions.genetic import genetico
from functions.fuzzy_rule_based import myFRBS

class LowQualityData():
	poblacion_ajuste = []
	MSE = "comparaMSE"
	StatPref = "comparaStatPref"
	StochDom = "comparaStochDom"

	def __init__(self, mygran=10, NUMPAR=3, NITER=50, POP=50, c=0.25):
		self.mygran = mygran
		self.NUMPAR = NUMPAR
		self.NITER = NITER
		self.POP = POP
		self.c = c
		self.params = np.zeros(NUMPAR)

	def is_empty(data_structure):
		if data_structure:
			return False
		else:
			return True

	#TODO: modificar el metodo genetico para que almacene y retorne la lista
	def fit(self, datos, comparador):
		# Limpiamos los arrays
		self.poblacion_ajuste.clear()
		self.params.clear()

		# Cargamos los datos y almacenamos la longitud
		self.datos = datos
		self.mydimx = len(datos)

		# Creamos el contenido de params
		self.params[0:self.mygran] = 1  # Consecuentes de las reglas borrosas
		self.params[self.mygran:2 * self.mygran] = [x * self.mydimx / (self.mygran - 1) for x in range(self.mygran)]
		self.params[2 * self.mygran:3 * self.mygran] = [2 for x in range(self.mygran)]

		# Creamos la poblecion de ajuste
		self.poblacion_ajuste = genetico(self.datos, self.mygran, self.mydimx, self.POP, self.params, deltacrisp,
										 comparador, self.datos[:, 1], self.NITER, self.c)
		return self.poblacion_ajuste

	def predict(self, datos):
		if self.is_empty(datos):
			error = '\033[93m'
			end = '\033[0m'
			print(f"{error}[ERROR]: No fitted data to use. Please, use fit() before predict(){end}")

		for gen in self.NITER:
			prediccion = myFRBS(datos[:, 0], self.poblacion_ajuste[0].genoma[0:self.mygran],
								self.poblacion_ajuste[0].genoma[self.mygran:2 * self.mygran],
								self.poblacion_ajuste[0].genoma[2 * self.mygran:3 * self.mygran], self.mygran, self.mydimx)
			f = open('..\\resources\\results\\output-crisp.csv', 'w')
			f.write('Index;Original;Prediction\n')
			for i in range(len(prediccion)):
				f.write(str(datos[i, 0]) + ';' + str(datos[i, 1]) + ';' + str(prediccion[i]) + '\n')
			f.close()



