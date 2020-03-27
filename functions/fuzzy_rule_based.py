import numpy as np
from utils.matrix_transpose import trans


def myFRBS(vx, cons, baseX, sigmaX, mygran, mydimx):
	tile1 = np.tile(vx, mygran)
	shape1 = np.reshape(tile1, [mygran, mydimx])
	transpose = trans(shape1)
	tile2 = np.tile(baseX, mydimx)
	shape2 = np.reshape(tile2, (mydimx, mygran))
	res = transpose - shape2
	activacion = np.exp(-np.square(res) / abs(sigmaX))
	tile3 = np.tile(cons, mydimx)
	shape3 = np.reshape(tile3, [mydimx, mygran])
	activcons = activacion * shape3
	sum1 = sum(activcons, 1)
	sum11 = np.matrix(activcons)
	sum2 = sum(activacion, 1) + 1e-6
	sum22 = np.matrix(activacion)
	result1 = sum11.sum(axis=1) + 1
	result2 = sum22.sum(axis=1) + 1 + 1e-6
	result = result1 / result2
	result = np.fromstring(result)

	return result
