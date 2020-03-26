import math
import random
import numpy as np
from operator import attrgetter
from statistics import mean
from numpy.core._multiarray_umath import array


f = open("..\\resources\\50sin-noise.dat")
datos = f.readlines()
f.close()

# Eliminamos muestras si es necesario para hacer pruebas rapidas
maxdatos = len(datos)

# procesar la lista de lineas
for i in range(maxdatos):
	datos[i] = datos[i].split(' ')
	# convertir cada valor en float
	for j in range(len(datos[i])):
		datos[i][j] = float(datos[i][j])

datos = array(datos)

# Test sobre datos sin ruido
f = open("..\\resources\\50sin-noiseless.dat")
datostest = f.readlines()
f.close()
for i in range(maxdatos):
	datostest[i] = datostest[i].split(' ')
	for j in range(len(datostest[i])):
		datostest[i][j] = float(datostest[i][j])

datostest = array(datostest)


def purgeNaNs(prediccion):
	np_prediccion = np.fromstring(prediccion)
	for i in range(np_prediccion.ndim):
		if math.isnan(np_prediccion[i]):
			np_prediccion[i] = 10


def trans(M):
	return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def myFRBS(vx, cons, baseX, sigmaX, mygran, mydimx):
	tile1 = np.tile(vx, mygran) # 50 - 10
	shape1 = np.reshape(tile1, [mygran, mydimx]) # 10 - 50
	transpose = trans(shape1) # 50 -10
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


def deltacrisp(params, observed):
	prediccion = myFRBS(datos[:, 0], params[0:mygran], params[mygran:2 * mygran], params[2 * mygran:3 * mygran], mygran,
						mydimx)
	purgeNaNs(prediccion)
	return prediccion - observed


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


class cromosoma:
	def __init__(self, genoma, rank, delta, distancia):
		self.genoma = genoma
		self.rank = rank
		self.delta = delta
		self.distancia = distancia

	def __repr__(self):
		return repr((self.genoma, self.rank, self.delta, self.distancia))


def comparaciones2a2(POPULATION, poblacion, comparator, c):
	comparaciones = np.zeros([POPULATION, POPULATION])
	distancias = np.zeros([POPULATION, POPULATION])
	for individual in range(POPULATION):
		print("ECM(", individual, ")=", np.mean(np.square(poblacion[individual].delta)))
		for ind2 in range(individual, POPULATION):
			comparaciones[ind2, individual] = comparator(poblacion[individual].delta, poblacion[ind2].delta, c)
			comparaciones[individual, ind2] = comparator(poblacion[ind2].delta, poblacion[individual].delta, c)
			distancias[ind2, individual] = np.sqrt(
				sum(np.square(poblacion[ind2].genoma - poblacion[individual].genoma)))
			distancias[individual, ind2] = distancias[ind2, individual]
	return comparaciones, distancias


def ordenapreferencias(POPULATION, poblacion, comparaciones, distancias):
	for individual in range(POPULATION):
		poblacion[individual].rank = sum(comparaciones[individual, :])
		poblacion[individual].distancia = 1.0 / sum(distancias[individual, :])
	print(individual, " antes sort=", comparaciones[individual, :], "rango=", poblacion[individual].rank)
	poblacion.sort(key=attrgetter('rank', 'distancia'))


def ordenaSimplex(POPULATION, poblacion, comparaciones, distancias):
	for individual in range(POPULATION):
		poblacion[individual].rank = sum(comparaciones[individual, :])
		# Como segundo criterio ponemos el MSE de los centros
		poblacion[individual].distancia = mean(np.square(poblacion[individual].delta))
	print(individual, " antes sort=", comparaciones[individual, :], "rango=", poblacion[individual].rank)
	poblacion.sort(key=attrgetter('rank', 'distancia'))


def varianza(simplex):
	xo = simplex[0].genoma
	for i in range(1, len(simplex)):
		xo = xo + simplex[i].genoma
	xo = 1.0 / len(simplex) * xo
	v = np.zeros(len(simplex[0].genoma))
	for i in range(1, len(simplex)):
		v = v + np.square(simplex[i].genoma - xo)
	v = 1.0 / len(simplex) * v
	print("v=", np.sqrt(np.dot(v, v)))
	return np.sqrt(np.dot(v, v))


def optimLocal(start, delta, comparator, observed, NITER, c):
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
	vertex = cromosoma(ge, 0, delta(ge, observed), 0)
	simplex.append(vertex)
	for i in range(N):
		ge = start.copy()
		ge[i] = ge[i] + volsimplex[i]
		vertex = cromosoma(ge, 0, delta(ge, observed), 0)
		simplex.append(vertex)
	MAXITER = NITER
	iter = 0
	lastvar = 0
	while iter <= MAXITER:
		if iter == 0:
			savefit = mean(np.square(simplex[0].delta))
		if iter % 10 == 0:
			var = varianza(simplex)
			print("     ** it=", iter, "bst=", mean(np.square(simplex[0].delta)), "Descub=",
				  np.percentile(simplex[0].delta, array([5, 95])),
				  "K=", Kg(simplex[0].delta, c), "(var", var, ")")
		iter = iter + 1

	if var == lastvar:
		print("Saliendo por varianza constante")
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
			print("Elegido el reflejado: xr=", xr, " MSE=", np.mean(np.square(deltaxr)))
		else:
			if betterbst:  # Expansion (4)
				xe = xr + gamma * (xr - xo)
				deltaxe = delta(xe, observed)
				bettere = comparator(deltaxe, deltaxr, c)
				if bettere:
					simplex[N] = cromosoma(xe, 0, deltaxe, 0)
					print("Elegido el expandido: xe=", xe, " MSE=", np.mean(np.square(deltaxe)))
				else:
					simplex[N] = cromosoma(xr, 0, deltaxr, 0)
				print("Elegido el reflejado (II): xr=", xr, " MSE=", np.mean(np.square(deltaxr)))
			else:  # Contraccion (5)
				xc = xo + rho * (xn1 - xo)
				deltaxc = delta(xc, observed)
				betterc = comparator(deltaxc, simplex[N].delta, c)
				if betterc:
					simplex[N] = cromosoma(xc, 0, deltaxc, 0)
					print("Elegido el contraido: xc=", xc, " MSE=", np.mean(np.square(deltaxc)))
				else:  # Reduccion (6)
					for i in range(1, N1):
						xi = simplex[0].genoma + sigma * (simplex[i].genoma - simplex[0].genoma)
						deltaxi = delta(xi, observed)
						simplex[i] = cromosoma(xi, 0, deltaxi, 0)

	comparaciones, distancias = comparaciones2a2(N1, simplex, comparator, c)
	ordenaSimplex(N1, simplex, comparaciones, distancias)
	return simplex[0].genoma.copy()


def cambia(a, b, alpha):
	return alpha * a + (1 - alpha) * b, (1 - alpha) * a + alpha * b


def randomcr():
	return array([random.uniform(-5, 5) for x in range(mygran)]
				 + [random.uniform(0.75, 1.25) * (x * mydimx / (mygran - 1)) for x in range(mygran)]
				 + [random.uniform(1, 5) for x in range(mygran)])


def cruce(chr1, chr2, delta, comparator, observed, c):
	off1 = chr1.genoma.copy()
	off2 = chr2.genoma.copy()
	componente = int(random.uniform(0, len(off1)))
	for i in range(mygran):
		if random.uniform(0, 1) < 0.25:
			alpha = random.uniform(-0.1, 0.5)
			off1[i], off2[i] = cambia(off1[i], off2[i], alpha)
			off1[i + mygran], off2[i + mygran] = cambia(off1[i + mygran], off2[i + mygran], alpha)
			off1[i + 2 * mygran], off2[i + 2 * mygran] = cambia(off1[i + 2 * mygran], off2[i + 2 * mygran], alpha)
	# Calculo del fitness
	delta1 = delta(off1, observed)
	delta2 = delta(off2, observed)
	print("Mediana del fitness (cruce)")
	m1 = np.median(delta1)
	m2 = np.median(delta2)
	for i in range(mygran):
		off1[i] = off1[i] - m1
		off2[i] = off2[i] - m2
	delta1 = delta(off1,observed)
	delta2 = delta(off2,observed)
	print(np.median(delta1), np.median(delta2))
	return cromosoma(off1.copy(), 99.0, delta1, 99.0), cromosoma(off2.copy(), 99.0, delta2, 99.0)


def mutacion(chr1, delta, comparator, observed, c):
	off1 = chr1.genoma.copy()
	off2 = randomcr()
	componente = int(random.uniform(0, len(off1)))
	for i in range(mygran):
		if random.uniform(0, 1) < 0.25:
			alpha = random.uniform(-0.1, 0.5)
			off1[i], off2[i] = cambia(off1[i], off2[i], alpha)
			off1[i + mygran], off2[i + mygran] = cambia(off1[i + mygran], off2[i + mygran], alpha)
			off1[i + 2 * mygran], off2[i + 2 * mygran] = cambia(off1[i + 2 * mygran], off2[i + 2 * mygran], alpha)
	# Calculo del fitness
	delta1 = delta(off1, observed)
	'''
		To test, I don't know with it does and if its good or wrong
	'''
	m1 = np.median(delta1)
	for i in range(mygran):
		off1[i] = off1[i] - m1
	delta1 = delta(off1,observed)
	return cromosoma(off1.copy(), 99.0, delta1, 99.0)


def genetico(POPULATION, start, delta, comparator, observed, NITER, c):
	# individual: punto de partida
	# fitness: calcula la respuesta para todos los puntos
	# comparator: compara dos respuestas entre si
	# observed: respuesta observada
	SIZE = len(start)
	SIZEOUTPUT = len(observed)
	NLOCAL = 100
	MEMETICO = 2
	#
	# Inicializacion de la poblacion inicial
	#
	poblacion = []
	ge = start.copy()
	cr = cromosoma(ge, 0, delta(ge, observed), 0)
	poblacion.append(cr)
	for individual in range(1, POPULATION):
		ge = randomcr()
		cr = cromosoma(ge, 0, delta(ge, observed), 0)
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
		print("Conservamos ", len(poblacion))
		# Aplicacion de operadores
		PRMUTA = 0.05
		while len(poblacion) < POPULATION:
			ind1 = int(np.floor(min(random.uniform(0, POPULATION), random.uniform(0, POPULATION))))
			ind2 = int(np.floor(min(random.uniform(0, POPULATION), random.uniform(0, POPULATION))))
			chr1, chr2 = cruce(savepoblacion[ind1], savepoblacion[ind2], delta, comparator, observed, c)
			if random.uniform(0, 1) < PRMUTA:
				chr1 = mutacion(chr1, delta, comparator, observed, c)
			if random.uniform(0, 1) < PRMUTA:
				chr2 = mutacion(chr2, delta, comparator, observed, c)
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
		print("Eliminados", eliminados)
		# Rellenamos con aleatorios si hiciese falta
		while len(poblacion) < POPULATION:
			ge = randomcr()
			cr = cromosoma(ge, 0, delta(ge, observed), 0)
			poblacion.append(cr)
		#
		# Hibridacion local mejor 5%
		#
		for iter in range(MEMETICO):
			opt = optimLocal(poblacion[iter].genoma, delta, comparator, observed, NLOCAL, c)
			deltaopt = delta(opt, observed)
			poblacion[iter] = cromosoma(opt.copy(), 99.0, deltaopt, 99.0)
		# limitamos tamanho
		poblacion = poblacion[0:POPULATION]
		# Ordenacion por rangos
		comparaciones, distancias = comparaciones2a2(len(poblacion), poblacion, comparator, c)
		ordenapreferencias(len(poblacion), poblacion, comparaciones, distancias)

		f = open('params-crisp.dat', 'w')
		for i in range(len(poblacion[0].genoma)):
			f.write(str(poblacion[0].genoma[i]) + '\n')
		f.close()
		prediccion = myFRBS(datos[:, 0], poblacion[0].genoma[0:mygran],
							poblacion[0].genoma[mygran:2 * mygran],
							poblacion[0].genoma[2 * mygran:3 * mygran], mygran, mydimx)
		f = open('output-crisp.dat', 'w')
		for i in range(len(prediccion)):
			f.write(str(datos[i, 0]) + ' ' + str(datos[i, 1]) + ' ' + str(prediccion[i]) + '\n')
		f.close()
	# Un global por si interrumpimos el aprendizaje
	# for individual in range(POPULATION):
	#	print(individual, poblacion[individual].rank, mean(square(poblacion[individual].delta)), end=' ')
	#	mean(square(delta(poblacion[individual].genoma,observed))),
	#	poblacion[individual].distancia
	# Se guardan a disco parametros y prediccion
	return poblacion[0].genoma


mygran = 10  # Granulalidad
mydimx = maxdatos  # Numero de instancias

NUMPAR = 3 * mygran
params = np.zeros([NUMPAR])
params[0:mygran] = 1  # Consecuentes de las reglas borrosas
params[mygran:2 * mygran] = [x * mydimx / (mygran - 1) for x in range(mygran)]
params[2 * mygran:3 * mygran] = [2 for x in range(mygran)]

# f = open("params-crisp.dat")
# params = f.readlines()
# for i in range(len(params)):
# 	params[i]=float(params[i])
# params=array(params)

NITER = 50
POP = 50
c = 0.25
for rep in range(1):
	prediccion = myFRBS(datos[:, 0], params[0:mygran], params[mygran:2 * mygran], params[2 * mygran:3 * mygran], mygran,
						mydimx)
	delta = prediccion - datos[:, 1]
	print("Numero de puntos cubiertos:", np.percentile(delta, array([5, 95])))
	solucion1 = genetico(POP, params, deltacrisp, comparaMSE, datos[:, 1], NITER, c)
	solucion2 = genetico(POP,params,deltacrisp,comparaStatPref,datos[:,1],NITER,c)
	solucion3 = genetico(POP, params, deltacrisp, comparaStochDom, datos[:, 1], NITER, c)
	params = solucion1
	print("Solucion 1")
	print(solucion1)
	print("Solucion 2")
	print(solucion2)
	print("Solucion 3")
	print(solucion3)

'''
	Fix and test
'''
# x <- read.table("output-crisp.dat")
# plot(x[:,1],x[:,2],ylim=c(min(x[,2]),max(x[,2])),xlab="independent variable",ylab="observed variable")
# lines(x[,1],x[,3],col="red",lw=2)
# z <- read.table("50sin-noiseless.dat")
# mean((z[,2]-x[,3])**2)
