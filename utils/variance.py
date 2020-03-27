import numpy as np


def varianza(simplex):
	xo = simplex[0].genoma
	for i in range(1, len(simplex)):
		xo = xo + simplex[i].genoma
	xo = 1.0 / len(simplex) * xo
	v = np.zeros(len(simplex[0].genoma))
	for i in range(1, len(simplex)):
		v = v + np.square(simplex[i].genoma - xo)
	v = 1.0 / len(simplex) * v
	# print("v=", np.sqrt(np.dot(v, v)))
	return np.sqrt(np.dot(v, v))

