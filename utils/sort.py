from operator import attrgetter
from statistics import mean
import numpy as np


def ordenapreferencias(POPULATION, poblacion, comparaciones, distancias):
	for individual in range(POPULATION):
		poblacion[individual].rank = sum(comparaciones[individual, :])
		poblacion[individual].distancia = 1.0 / sum(distancias[individual, :])
	# print(individual, " antes sort=", comparaciones[individual, :], "rango=", poblacion[individual].rank)
	poblacion.sort(key=attrgetter('rank', 'distancia'))


def ordenaSimplex(POPULATION, poblacion, comparaciones, distancias):
	for individual in range(POPULATION):
		poblacion[individual].rank = sum(comparaciones[individual, :])
		# Como segundo criterio ponemos el MSE de los centros
		poblacion[individual].distancia = mean(np.square(poblacion[individual].delta))
	# print(individual, " antes sort=", comparaciones[individual, :], "rango=", poblacion[individual].rank)
	poblacion.sort(key=attrgetter('rank', 'distancia'))
