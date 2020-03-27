from numpy.core._multiarray_umath import array
import numpy as np


def comparaStatPref(delta1, delta2, c):
	mejor1 = sum(abs(delta1) < abs(delta2))
	mejor2 = sum(abs(delta2) < abs(delta1))
	return mejor1 > mejor2


def K(x, c):
	t = array(np.square(x) < np.square(c))
	return sum(t)


def Kg(x, c):
	t = np.exp(-np.square(x / c))
	return sum(t)


def comparaStochDom(delta1, delta2, c):
	m1 = Kg(delta1, c)
	m2 = Kg(delta2, c)
	return m1 > m2


def comparaMSE(delta1, delta2, c):
	mse1 = sum(np.square(delta1))
	mse2 = sum(np.square(delta2))
	return mse1 < mse2


def comparaciones2a2(POPULATION, poblacion, comparator, c):
	comparaciones = np.zeros([POPULATION, POPULATION])
	distancias = np.zeros([POPULATION, POPULATION])
	for individual in range(POPULATION):
		# print("ECM(", individual, ")=", np.mean(np.square(poblacion[individual].delta)))
		for ind2 in range(individual, POPULATION):
			comparaciones[ind2, individual] = comparator(poblacion[individual].delta, poblacion[ind2].delta, c)
			comparaciones[individual, ind2] = comparator(poblacion[ind2].delta, poblacion[individual].delta, c)
			distancias[ind2, individual] = np.sqrt(
				sum(np.square(poblacion[ind2].genoma - poblacion[individual].genoma)))
			distancias[individual, ind2] = distancias[ind2, individual]
	return comparaciones, distancias
