import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.ctmc import ctmc

lam = 3
mu = 2
alpha = 5


# Espacio de estados
Sx = [0, 1] # Vacío (0), Ocupado en la primera fase del servicio (1)
Sy = [0, 1, 2] # Vacío (0), Ocupado en la primera fase del servicio (1), Ocupado en la segunda fase del servicio (2)

estados = []
for i in Sx:
    for j in Sy:
        estado = ",".join((str(i),str(j)))
        estados.append(estado)
estados = np.array(estados)

matriz = []
for estado_actual in estados:
    i = int(estado_actual[0])
    j = int(estado_actual[-1])
    listaAux = []
    for estado_futuro in estados:
        i_ = int(estado_futuro[0])
        j_ = int(estado_futuro[-1])
        
        if i_==1 and j_==j and i==0:
            listaAux.append(0.7*lam)
        elif i_==i and j_==1 and j==0:
            listaAux.append(0.3*lam)
        elif i_==0 and j_==j and i==1:
            listaAux.append(mu)
        elif i_==i and j_==2 and j==1:
            listaAux.append(alpha)
        elif i_==i and j_==0 and j==2:
            listaAux.append(alpha)
        else:
            listaAux.append(0)
    matriz.append(listaAux)
    
for estado_actual in range(0, len(estados)):
    for estado_futuro in range(0, len(estados)):
        if estado_actual == estado_futuro:
            matriz[estado_actual][estado_futuro]=-sum(matriz[estado_actual])

matriz = np.array(matriz)

autosElectricos = ctmc(matriz)