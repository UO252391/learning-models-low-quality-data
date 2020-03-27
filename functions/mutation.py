import random

from functions.chromosome import cromosoma
from utils.randomcr import randomcr
from utils.change import cambia
import numpy as np


def mutacion(datos, mygran, mydimx, chr1, delta, comparator, observed, c):
	off1 = chr1.genoma.copy()
	off2 = randomcr(mygran, mydimx)
	componente = int(random.uniform(0, len(off1)))
	for i in range(mygran):
		if random.uniform(0, 1) < 0.25:
			alpha = random.uniform(-0.1, 0.5)
			off1[i], off2[i] = cambia(off1[i], off2[i], alpha)
			off1[i + mygran], off2[i + mygran] = cambia(off1[i + mygran], off2[i + mygran], alpha)
			off1[i + 2 * mygran], off2[i + 2 * mygran] = cambia(off1[i + 2 * mygran], off2[i + 2 * mygran], alpha)
	# Calculo del fitness
	delta1 = delta(datos, mygran, mydimx, off1, observed)
	'''
		To test, I don't know with it does and if its good or wrong
	'''
	m1 = np.median(delta1)
	for i in range(mygran):
		off1[i] = off1[i] - m1
	delta1 = delta(datos, mygran, mydimx, off1, observed)
	return cromosoma(off1.copy(), 99.0, delta1, 99.0)

