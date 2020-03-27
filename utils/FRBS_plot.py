import matplotlib.pyplot as plt
import pandas as pd


def FRBS_plot(solucion):
	output_crisp = pd.read_csv("..\\resources\\results\\output-crisp.csv", sep=";",
							   names=['Index', 'Original', 'Prediction'])

	output_crisp = output_crisp.apply(pd.to_numeric, errors='coerce')

	ax = plt.gca()

	output_crisp.plot(kind='line', x='Index', y='Original', color='red', ax=ax)
	output_crisp.plot(kind='line', x='Index', y='Prediction', color='blue', ax=ax)

	# pd.plot(x[:, 1],
	# 		 x[:, 2],
	# 		 ylim=c(x.max()),
	# 		 xlabel="independent variable",
	# 		 ylabel="observed variable")
	# lines(x[:, 1], x[:, 3], col="red", lw=2)
	# z = np.loadtxt("50sin-noiseless.dat")
	# noiseless_data = pd.read_table("..\\resources\\50sin-noiseless2.dat", sep=' ', names=['Index', 'Value'])
	noiseless_data = pd.read_csv("..\\resources\\data\\50sin-noiseless.csv", sep=";", names=['Index', 'Real'])
	# TODO arreglar esto
	noiseless_data = noiseless_data.apply(pd.to_numeric, errors='coerce')
	noiseless_data.plot(kind='line', x='Index', y='Real', color='green', ax=ax)

	plt.title(solucion)
	plt.xlabel("Index")
	plt.ylabel("Value")
	plt.show()
	plt.savefig("..\\resources\\results\\" + solucion + ".png")