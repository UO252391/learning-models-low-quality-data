import matplotlib.pyplot as plt
import pandas as pd

from functions.delta import deltacrisp
from functions.genetic import *
from utils.comparators import comparaMSE
from utils.comparators import comparaStatPref
from utils.comparators import comparaStochDom


class LowQualityData():
	poblacion_ajuste = []
	MSE = comparaMSE
	StatPref = comparaStatPref
	StochDom = comparaStochDom

	def __init__(self, mygran=10, NUMPAR=3, NITER=50, POP=50, c=0.25, title="Comparacion de resultados"):
		self.mygran = mygran
		self.NUMPAR = NUMPAR * mygran
		self.NITER = NITER
		self.POP = POP
		self.c = c
		self.params = np.zeros([self.NUMPAR])
		self.title = title

	# TODO: modificar el metodo genetico para que almacene y retorne la lista
	def fit(self, datos, comparador):
		# Limpiamos los arrays
		self.poblacion_ajuste.clear()

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

	# return self.poblacion_ajuste

	def predict(self, datos):
		# if self.is_empty(datos):
		# 	error = '\033[93m'
		# 	end = '\033[0m'
		# 	print(f"{error}[ERROR]: No fitted data to use. Please, use fit() before predict(){end}")

		for gen in range(self.NITER):
			self.prediccion = myFRBS(datos[:, 0], self.poblacion_ajuste[gen].genoma[0:self.mygran],
									 self.poblacion_ajuste[gen].genoma[self.mygran:2 * self.mygran],
									 self.poblacion_ajuste[gen].genoma[2 * self.mygran:3 * self.mygran], self.mygran,
									 self.mydimx)
			f = open('..\\resources\\results\\output-crisp.csv', 'w')
			f.write('Index;Original;Prediction\n')
			for i in range(len(self.prediccion)):
				f.write(str(datos[i, 0]) + ';' + str(datos[i, 1]) + ';' + str(self.prediccion[i]) + '\n')
			f.close()

		self.save()

		return self.prediccion

	def setLabel(self, label):
		self.title = label

	def save(self):
		output_crisp = pd.read_csv("..\\resources\\results\\output-crisp.csv", sep=";",
								   names=['Index', 'Original', 'Prediction'])

		output_crisp = output_crisp.apply(pd.to_numeric, errors='coerce')

		ax = plt.gca()

		output_crisp.plot(kind='line', x='Index', y='Original', color='red', ax=ax)
		output_crisp.plot(kind='line', x='Index', y='Prediction', color='blue', ax=ax)

		noiseless_data = pd.read_csv("..\\resources\\data\\50sin-noiseless.csv", sep=";", names=['Index', 'Real'])
		noiseless_data = noiseless_data.apply(pd.to_numeric, errors='coerce')
		noiseless_data.plot(kind='line', x='Index', y='Real', color='green', ax=ax)

		plt.title(self.title)
		plt.xlabel("Index")
		plt.ylabel("Value")

		plt.savefig("..\\resources\\results\\" + self.title + ".png", bbox_inches='tight')
