import numpy as np
import math


def purgeNaNs(prediccion):
	np_prediccion = np.fromstring(prediccion)
	for i in range(np_prediccion.ndim):
		if math.isnan(np_prediccion[i]):
			np_prediccion[i] = 10
