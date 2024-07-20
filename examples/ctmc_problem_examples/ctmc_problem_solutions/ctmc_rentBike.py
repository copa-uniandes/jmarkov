import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.ctmc import ctmc

tasaLlegada = 5
tasaSalida = 9
estados = [i for i in range(0,7)]

matriz = []
for i in estados:
    listaAux = []
    for j in estados:
        if j==i+1 and i<6:
            listaAux.append(tasaLlegada)
        elif j==i-1 and i>0:
            listaAux.append(tasaSalida*i)
        else: 
            listaAux.append(0)
    matriz.append(listaAux)
for i in estados:
    for j in estados:
        if i==j:
            matriz[i][j]=-sum(matriz[i])

matriz = np.array(matriz)
rentBike = ctmc(matriz)
  