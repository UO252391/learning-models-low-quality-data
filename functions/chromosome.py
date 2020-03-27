
class cromosoma:
	def __init__(self, genoma, rank, delta, distancia):
		self.genoma = genoma
		self.rank = rank
		self.delta = delta
		self.distancia = distancia

	def __repr__(self):
		return repr((self.genoma, self.rank, self.delta, self.distancia))
