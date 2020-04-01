# learning-models-low-quality-data
####Un paquete python para aprendizaje de modelos con datos de baja calidad 

Inducir el conocimiento a partir de conjuntos de datos limitados plantea un conjunto completamente diferente de desafíos
que el análisis de big data porque, aparte de las consideraciones de eficiencia computacional, pequeños conjuntos de 
datos obligan al investigador a usar la misma información tanto para el análisis de datos exploratorios (revelación de 
relaciones de causa / efecto) como para la prueba de si estas relaciones son compatibles con los datos. Descartar datos
"menos que perfectos" no es una opción para conjuntos de datos pequeños. En este proyecto se estudian técnicas para
explotar datos "menos que perfectos" mediante la optimización de funciones de pérdida generalizadas, y se desarrolla 
un paquete python con un algoritmo memético capaz de encontrar los elementos minimales de una ordenación estocástica.

####1. Control de versiones

Con el fin de facilitar la tarea de documentacion del proyecto se ha creado un repositrio remoto en Github en el que se
albergara el proyecto, y con el, todo el codigo desarrollado a partir de este punto. 

####2. Toma de contacto con el proyecto
Implementacion del apendice A del paper adjuntado (\resources). Mirar el paper por encima. Se adjunta codigo  viejo en python2

#####  2.1 Trabajo realizado
Para la imlementacion del apendice A se ha reestructurado todo el proyecto. Esta reestructuracion
afecta tanto al codigo como a las clases. Primero se ha reformulado todo el codigo con el fin de hacerlo
mas legible a simple vista. Lo siguiente fue la eliminacion de prints de depuracion innecesarios, manteniendo unicamente
aquellos que muestran resultados de interes. Por ultimo se opto por la division de la unica clase FRBS-Ycrisp.py en varias nuevas,
dividiendo estas en diferentes packages segun su funcionalidad. 

#####  2.2 Contenido añadido
Se añadio un metodo capaz de representar los resultados, comparando el valor real, el proporcionado con ruido y el predicho 
por el algoritmo

####3. Implementacion similar a otros algoritmos de clasificacion/regresion
En terminos generales consiste en la creacion de metodos fit y predict. 

#####3.1 Trabajo realizado
Para poder llevar a cabo esto se creo la clse LowQualityData, 
que alberga estos 2 metodos, ademas de un metodo para la representaacion de resultados, un metodo para la modificacion del titulo 
del grafico y su nombre, y por ultimo un constructor que por defecto instaura los valores utilizados en el codigo proporcionado 
al comienzo del projecto pero que permite modificar alguno si se desea. Como resultado de estos cambios, se ha modificado 
la clase main, leyendo ahora los datos de train(originales sin ruido) y de test(originales con ruido), llamando a fit con los 
datos de train y a predict con los datos de test.