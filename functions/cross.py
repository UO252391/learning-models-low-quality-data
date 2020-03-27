import random
import numpy as np
from utils.change import cambia
from functions.chromosome import cromosoma


def cruce(datos, mygran, mydimx, chr1, chr2, delta, comparator, observed, c):
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
	delta1 = delta(datos, mygran, mydimx, off1, observed)
	delta2 = delta(datos, mygran, mydimx, off2, observed)
	m1 = np.median(delta1)
	m2 = np.median(delta2)
	for i in range(mygran):
		off1[i] = off1[i] - m1
		off2[i] = off2[i] - m2
	delta1 = delta(datos, mygran, mydimx, off1, observed)
	delta2 = delta(datos, mygran, mydimx, off2, observed)
	return cromosoma(off1.copy(), 99.0, delta1, 99.0), cromosoma(off2.copy(), 99.0, delta2, 99.0)
