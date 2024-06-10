import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp


# Parametros:
p = 0.4
q = 0.2
retLleno = 5
retAlto = 4
retMedio = 3
retBajo = 2
W = 10
Z = 4
C = 12

# Variable de estado:
# X_{t}: Nivel del embalse en la época t

# Espacio de estados
#estadosRepresa = [i for i in range(0,4)] # [0, 1, 2, 3] --> 0: Lleno, 1: Alto, 2: Medio, 3: Bajo
estadosRepresa = np.array(['Lleno', 'Alto', 'Medio', 'Bajo'])

# Acciones
#accionesRepresa = np.array([a for a in range(0,2)]) # array([0, 1])--> 0: Abrir, 1:Cerrar
accionesRepresa = np.array(['Abrir','Cerrar'])

# Retornos Inmediatos
retornosRepresa = np.array([[W*retLleno - Z,  -C*(p*q)],
                            [W*retAlto - Z, 0],
                            [W*retMedio - Z, 0],
                            [W*retBajo - Z, 0]])

# Matrices de transición
matricesRepresa = {}
# Para a == 0
matrizRepresa = []
for estado_actual in range(0,len(estadosRepresa)):
    listaAux = []
    for estado_futuro in range(0,len(estadosRepresa)):
        if estado_futuro == 3:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizRepresa.append(listaAux)
    matricesRepresa[accionesRepresa[0]] = np.array(matrizRepresa)
    
# Para a == 1
matrizRepresa = []
for estado_actual in range(0,len(estadosRepresa)):
    listaAux = []
    for estado_futuro in range(0,len(estadosRepresa)):
        # Si decido cerrar y estaba lleno
        if estado_futuro == estado_actual and estado_actual>0:
            listaAux.append(1-(p*q))
        # Si decido cerrar y llueve y se aumenta el nivel
        elif estado_futuro == estado_actual-1 and estado_actual>0:
            listaAux.append(p*q)
        # Si decido cerrar y no aumenta el nivel
        elif estado_actual == 0 and estado_futuro == 0:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizRepresa.append(listaAux)
    matricesRepresa[accionesRepresa[1]] = np.array(matrizRepresa)

# Creo el problema como un mdp
mdpRepresa = dtmdp(estadosRepresa, accionesRepresa, matricesRepresa, retornosRepresa, 0.8)
print(mdpRepresa.solve(0, method = "value_iteration"))
print(mdpRepresa.solve(0, method="policy_iteration"))