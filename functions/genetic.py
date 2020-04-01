import random

import numpy as np

from functions.chromosome import cromosoma
from functions.cross import cruce
from functions.mutation import mutacion
from functions.optimize import optimLocal
from utils.comparators import comparaciones2a2
from utils.randomcr import randomcr
from utils.sort import ordenapreferencias


def genetico(datos, mygran, mydimx, POPULATION, start, delta, comparator, observed, NITER, c):
	# individual: punto de partida
	# fitness: calcula la respuesta para todos los puntos
	# comparator: compara dos respuestas entre si
	# observed: respuesta observada
	SIZE = len(start)
	SIZEOUTPUT = len(observed)
	NLOCAL = 100
	MEMETICO = 2
	results = []
	#
	# Inicializacion de la poblacion inicial
	#
	poblacion = []
	ge = start.copy()
	cr = cromosoma(ge, 0, delta(datos, mygran, mydimx, ge, observed), 0)
	poblacion.append(cr)
	for individual in range(1, POPULATION):
		ge = randomcr(mygran, mydimx)
		cr = cromosoma(ge, 0, delta(datos, mygran, mydimx, ge, observed), 0)
		poblacion.append(cr)
	#
	# Ordenacion por preferencias
	#
	for gen in range(NITER):
		comparaciones, distancias = comparaciones2a2(POPULATION, poblacion, comparator, c)
		ordenapreferencias(POPULATION, poblacion, comparaciones, distancias)
		#
		# Aplicacion de operadores geneticos para formar la poblacion intermedia
		#
		savepoblacion = []
		for iter in range(POPULATION):
			savepoblacion.append(poblacion[iter])
		poblacion = []
		# Todos los de rango cero y uno (podriamos meter toda la poblacion y entonces es NSGA-II)
		MINRANK = 0
		poblacion.append(savepoblacion[0])
		if gen > 0:
			iter = 1
			while savepoblacion[iter].rank <= MINRANK:
				poblacion.append(savepoblacion[iter])
				iter = iter + 1
				if iter >= len(savepoblacion):
					break
		# Aplicacion de operadores
		PRMUTA = 0.05
		while len(poblacion) < POPULATION:
			ind1 = int(np.floor(min(random.uniform(0, POPULATION), random.uniform(0, POPULATION))))
			ind2 = int(np.floor(min(random.uniform(0, POPULATION), random.uniform(0, POPULATION))))
			chr1, chr2 = cruce(datos, mygran, mydimx, savepoblacion[ind1], savepoblacion[ind2], delta, comparator, observed, c)
			if random.uniform(0, 1) < PRMUTA:
				chr1 = mutacion(datos, mygran, mydimx, chr1, delta, comparator, observed, c)
			if random.uniform(0, 1) < PRMUTA:
				chr2 = mutacion(datos, mygran, mydimx, chr2, delta, comparator, observed, c)
			if len(poblacion) < POPULATION:
				poblacion.append(chr1)
			else:
				break
			if len(poblacion) < POPULATION:
				poblacion.append(chr2)
			else:
				break
		comparaciones, distancias = comparaciones2a2(len(poblacion), poblacion, comparator, c)
		ordenapreferencias(len(poblacion), poblacion, comparaciones, distancias)
		# quitamos duplicados
		posicion = 0
		eliminados = 0
		while posicion < len(poblacion) - 2:
			while sum(np.square(poblacion[posicion + 1].genoma - poblacion[posicion].genoma)) < 1e-6:
				if posicion == len(poblacion) - 2:
					break
				del poblacion[posicion + 1]
				eliminados = eliminados + 1
			posicion = posicion + 1
		# Rellenamos con aleatorios si hiciese falta
		while len(poblacion) < POPULATION:
			ge = randomcr(mygran, mydimx)
			cr = cromosoma(ge, 0, delta(datos, mygran, mydimx, ge, observed), 0)
			poblacion.append(cr)
		#
		# Hibridacion local mejor 5%
		#
		for iter in range(MEMETICO):
			opt = optimLocal(datos, mygran, mydimx, poblacion[iter].genoma, delta, comparator, observed, NLOCAL, c)
			deltaopt = delta(datos, mygran, mydimx, opt, observed)
			poblacion[iter] = cromosoma(opt.copy(), 99.0, deltaopt, 99.0)
		# limitamos tamanho
		poblacion = poblacion[0:POPULATION]
		# Ordenacion por rangos
		comparaciones, distancias = comparaciones2a2(len(poblacion), poblacion, comparator, c)
		ordenapreferencias(len(poblacion), poblacion, comparaciones, distancias)

		f = open('..\\resources\\results\\params-crisp.csv', 'w')
		for i in range(len(poblacion[0].genoma)):
			f.write(str(poblacion[0].genoma[i]) + '\n')
		f.close()
		# prediccion = myFRBS(datos[:, 0], poblacion[0].genoma[0:mygran],
		# 					poblacion[0].genoma[mygran:2 * mygran],
		# 					poblacion[0].genoma[2 * mygran:3 * mygran], mygran, mydimx)
		# f = open('..\\resources\\results\\output-crisp.csv', 'w')
		# f.write('Index;Original;Prediction\n')
		# for i in range(len(prediccion)):
		# 	f.write(str(datos[i, 0]) + ';' + str(datos[i, 1]) + ';' + str(prediccion[i]) + '\n')
		# f.close()
		results.append(poblacion[0])
	# return poblacion[0].genoma
	return results
