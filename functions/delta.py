from functions.fuzzy_rule_based import myFRBS
from functions.purgeNaNs import purgeNaNs


def deltacrisp(datos, mygran, mydimx, params, observed):
	prediccion = myFRBS(datos[:, 0], params[0:mygran], params[mygran:2 * mygran], params[2 * mygran:3 * mygran], mygran,
						mydimx)
	purgeNaNs(prediccion)
	return prediccion - observed