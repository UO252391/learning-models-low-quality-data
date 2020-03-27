import random
from numpy.core._multiarray_umath import array


def randomcr(mygran, mydimx):
	return array([random.uniform(-5, 5) for x in range(mygran)]
				 + [random.uniform(0.75, 1.25) * (x * mydimx / (mygran - 1)) for x in range(mygran)]
				 + [random.uniform(1, 5) for x in range(mygran)])
