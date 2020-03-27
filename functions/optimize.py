from statistics import mean
from numpy.core._multiarray_umath import array
from functions.chromosome import cromosoma
from utils.variance import varianza
from utils.comparators import comparaciones2a2
from utils.sort import ordenaSimplex
import numpy as np


def optimLocal(datos, mygran, mydimx, start, delta, comparator, observed, NITER, c):
	# Optimizacion local mediante una variante de Nelder-Mead
	# NECESITA UN ORDEN TOTAL.
	alpha = 1
	gamma = 2
	rho = 0.5
	sigma = 0.5
	N = len(start)
	N1 = N + 1
	volsimplex = array([2 for x in range(mygran)] + [10 for x in range(mygran)] + [1 for x in range(mygran)])
	simplex = []
	ge = start.copy()
	vertex = cromosoma(ge, 0, delta(datos, mygran, mydimx, ge, observed), 0)
	simplex.append(vertex)
	for i in range(N):
		ge = start.copy()
		ge[i] = ge[i] + volsimplex[i]
		vertex = cromosoma(ge, 0, delta(datos, mygran, mydimx, ge, observed), 0)
		simplex.append(vertex)
	MAXITER = NITER
	iter = 0
	lastvar = 0
	while iter <= MAXITER:
		if iter == 0:
			savefit = mean(np.square(simplex[0].delta))
		if iter % 10 == 0:
			var = varianza(simplex)

		iter = iter + 1

	if var == lastvar:
		MAXITER = iter - 1
		lastvar = var
		if iter == MAXITER and var > 0.05 and MAXITER < 400:
			MAXITER = MAXITER + 10
		else:
			if var < 1e-4:
				MAXITER = iter - 1
			else:
				iter = iter + 1

		comparaciones, distancias = comparaciones2a2(N1, simplex, comparator, c)
		ordenaSimplex(N1, simplex, comparaciones, distancias)

		# Centroide de todos los puntos menos el peor (2)
		xn1 = simplex[N].genoma
		xo = simplex[0].genoma
		for i in range(1, N):
			xo = xo + simplex[i].genoma
		xo = 1.0 / N * xo
		# Reflexion (3)
		xr = xo + alpha * (xo - xn1)
		deltaxr = delta(xr, observed)
		# Reflejado mejor que el segundo peor, no el mejor
		better2w = comparator(deltaxr, simplex[N - 1].delta, c)
		betterbst = comparator(deltaxr, simplex[0].delta, c)
		if better2w and not betterbst:
			simplex[N] = cromosoma(xr, 0, deltaxr, 0)
		else:
			if betterbst:  # Expansion (4)
				xe = xr + gamma * (xr - xo)
				deltaxe = delta(xe, observed)
				bettere = comparator(deltaxe, deltaxr, c)
				if bettere:
					simplex[N] = cromosoma(xe, 0, deltaxe, 0)
				else:
					simplex[N] = cromosoma(xr, 0, deltaxr, 0)
			else:  # Contraccion (5)
				xc = xo + rho * (xn1 - xo)
				deltaxc = delta(xc, observed)
				betterc = comparator(deltaxc, simplex[N].delta, c)
				if betterc:
					simplex[N] = cromosoma(xc, 0, deltaxc, 0)
				else:  # Reduccion (6)
					for i in range(1, N1):
						xi = simplex[0].genoma + sigma * (simplex[i].genoma - simplex[0].genoma)
						deltaxi = delta(xi, observed)
						simplex[i] = cromosoma(xi, 0, deltaxi, 0)

	comparaciones, distancias = comparaciones2a2(N1, simplex, comparator, c)
	ordenaSimplex(N1, simplex, comparaciones, distancias)
	return simplex[0].genoma.copy()
