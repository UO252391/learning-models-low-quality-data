import matplotlib.pyplot as plt
import pandas as pd

from functions.delta import deltacrisp
from functions.fuzzy_rule_based import *
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
		self.datostrain = self.datos

	# return self.poblacion_ajuste

	def predict(self, datos):
		# if self.is_empty(datos):
		# 	error = '\033[93m'
		# 	end = '\033[0m'
		# 	print(f"{error}[ERROR]: No fitted data to use. Please, use fit() before predict(){end}")

		self.datos = np.array(datos)
		self.original = np.array(datos)

		for gen in range(self.NITER):
			self.prediccion = myFRBS(datos[:, 0], self.poblacion_ajuste[gen].genoma[0:self.mygran],
									 self.poblacion_ajuste[gen].genoma[self.mygran:2 * self.mygran],
									 self.poblacion_ajuste[gen].genoma[2 * self.mygran:3 * self.mygran], self.mygran,
									 self.mydimx)
		# f = open('..\\resources\\results\\output-crisp.csv', 'w')
		# f.write('Index;Original;Prediction\n')
		# for i in range(len(self.prediccion)):
		# 	f.write(str(datos[i, 0]) + ';' + str(datos[i, 1]) + ';' + str(self.prediccion[i]) + '\n')
		# f.close()
		# self.save()
		self.datos[:, 1] = self.prediccion
		return self.prediccion

	def adjust(self, numiter):
		for _ in range(numiter):
			self.datos = np.array(self.datos)
			predicho = [self.__suma(self.datos[0, 1], self.prediccion[0])]
			for i in range(1, len(self.datos) - 1):
				predicho.append(self.__suma(predicho[i - 1], self.datos[i + 1, 1]))
			predicho.append(self.__suma(predicho[len(predicho) - 2], self.datos[len(self.datos) - 1, 1]))
			predicho = np.array(predicho)

			# Ajuste del primero
			predicho[0] = self.__suma(predicho[0], self.datos[1, 1])

			for i in range(len(self.datos) - 1):
				self.datos[i, 1] = predicho[i]

		predicho = np.array(predicho)

		f = open('..\\resources\\results\\output-crisp.csv', 'w')
		f.write('Index;Original;Prediction;Adjust\n')
		print("PREDICHO")
		print(predicho)
		for i in range(len(self.prediccion) - 1):
			f.write(str(self.datos[i, 0]) + ';' + str(self.original[i, 1]) + ';' + str(self.prediccion[i]) + ';' + str(
				predicho[i]) + '\n')
		f.close()
		self.predicho = predicho
		return self.predicho

	def mse(self, Y, predicted):
		array1 = np.array(Y)
		array2 = np.array(predicted)

		diference_array = np.subtract(array1, array2)

		squared_array = np.square(diference_array)

		return squared_array.mean()

	def __suma(self, a, b):
		if a < 0 <= b:
			return a + ((abs(a) + abs(b)) / 2)
		elif a < 0 and b < 0:
			return -(abs(a) + abs(b)) / 2
		else:
			return (abs(a) + abs(b)) / 2

	def setLabel(self, label):
		self.title = label

	def save(self):
		output_crisp = pd.read_csv("..\\resources\\results\\output-crisp.csv", sep=";",
								   names=['Index', 'Original', 'Prediction', 'Adjust'])

		output_crisp = output_crisp.apply(pd.to_numeric, errors='coerce')

		ax = plt.gca()

		output_crisp.plot(marker='o', markersize=1, x='Index', y='Original', color='red', ax=ax)
		output_crisp.plot(marker='o', markersize=1, x='Index', y='Prediction', color='black', ax=ax)
		output_crisp.plot(kind='line', x='Index', y='Adjust', color='blue', ax=ax)

		noiseless_data = pd.read_csv("..\\resources\\data\\50sin-noiseless.csv", sep=";", names=['Index', 'Real'])
		noiseless_data = noiseless_data.apply(pd.to_numeric, errors='coerce')
		noiseless_data.plot(kind='line', x='Index', y='Real', color='green', ax=ax)

		plt.title(self.title)
		plt.xlabel("Index")
		plt.ylabel("Value")

		plt.savefig("..\\resources\\results\\" + self.title + ".png", bbox_inches='tight')
