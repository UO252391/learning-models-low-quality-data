import matplotlib.pyplot as plt
import pandas as pd


def FRBS_plot(solucion):
	output_crisp = pd.read_csv("..\\resources\\results\\output-crisp.csv", sep=";",
							   names=['Index', 'Original', 'Prediction'])

	output_crisp = output_crisp.apply(pd.to_numeric, errors='coerce')

	ax = plt.gca()

	output_crisp.plot(kind='line', x='Index', y='Original', color='red', ax=ax)
	output_crisp.plot(kind='line', x='Index', y='Prediction', color='blue', ax=ax)

	noiseless_data = pd.read_csv("..\\resources\\data\\50sin-noiseless.csv", sep=";", names=['Index', 'Real'])
	noiseless_data = noiseless_data.apply(pd.to_numeric, errors='coerce')
	noiseless_data.plot(kind='line', x='Index', y='Real', color='green', ax=ax)

	plt.title(solucion)
	plt.xlabel("Index")
	plt.ylabel("Value")
	plt.show()
	plt.savefig("..\\resources\\results\\" + solucion + ".png")