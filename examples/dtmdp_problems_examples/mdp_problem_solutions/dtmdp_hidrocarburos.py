import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp

# Variable de estado:
# X_{t}: Grado de deterioro de la máquina en la época t

# Espacio de estados
estadosMaquina = np.array(['Perfectas Condiciones','Defecto','Avería Total'])
# Acciones
accionesMaquina = np.array(['Reparar','Sustituir','Nada'])
# Retornos Inmediatos
retornosMaquina = []
for estado_actual in range(0,len(estadosMaquina)):
    listaAux = []
    for a in range(0,len(accionesMaquina)):
        # Si decide reparar
        if a == 0:
            listaAux.append(50)
        # Si decide sustituir
        elif a == 1:
            listaAux.append(80)
        # Si decide no hacer nada y está en defecto
        elif a == 2 and estado_actual == 1:
            listaAux.append(10)
        # Si decide no hacer nada y está en avería total
        elif a == 2 and estado_actual == 2:
            listaAux.append(20)
        else:
            listaAux.append(0)
    retornosMaquina.append(listaAux)
retornosMaquina = np.array(retornosMaquina)

# Matrices de transición
matricesMaquina = {}
# Para a == 0
matrizMaquina = []
for estado_actual in range(0,len(estadosMaquina)):
    listaAux = []
    for estado_futuro in range(0,len(estadosMaquina)):
        # Si está en perfectas condiciones
        if estado_actual == 0 and estado_futuro == 0:
            listaAux.append(1)
        # Si está en defecto y queda en perfectas condiciones
        elif estado_actual == 1 and estado_futuro == 0:
            listaAux.append(0.8)
        # Si está en defecto y permanece en defecto
        elif estado_actual == 1 and estado_futuro == 1:
            listaAux.append(0.2)
        # Si está en avería total y queda en perfectas condiciones
        elif estado_actual == 2 and estado_futuro == 0:
            listaAux.append(0.3)
        # Si está en avería total y permanece en avería total
        elif estado_actual == 2 and estado_futuro == 2:
            listaAux.append(0.7)
        else:
            listaAux.append(0)
    matrizMaquina.append(listaAux)
    matricesMaquina[accionesMaquina[0]] = np.array(matrizMaquina)
    
# Para a == 1
matrizMaquina = []
for estado_actual in range(0,len(estadosMaquina)):
    listaAux = []
    for estado_futuro in range(0,len(estadosMaquina)):
        # Si está en perfectas condiciones
        if estado_futuro == 0:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizMaquina.append(listaAux)
    matricesMaquina[accionesMaquina[1]] = np.array(matrizMaquina)    

# Para a == 2
matrizMaquina = []
for estado_actual in range(0,len(estadosMaquina)):
    listaAux = []
    for estado_futuro in range(0,len(estadosMaquina)):
        # Si está en perfectas condiciones y permanece en perfectas condiciones
        if estado_actual == 0 and estado_futuro == 0:
            listaAux.append(0.9)
        # Si está en perfectas condiciones y presenta defecto
        elif estado_actual == 0 and estado_futuro == 1:
            listaAux.append(0.09)
        # Si está en perfectas condiciones y presenta avería total
        elif estado_actual == 0 and estado_futuro == 2:
            listaAux.append(0.01)
        # Si está en defecto y permanece en defecto
        elif estado_actual == 1 and estado_futuro == 1:
            listaAux.append(0.55)
        # Si está en defecto y presenta avería total
        elif estado_actual == 1 and estado_futuro == 2:
            listaAux.append(0.45)
        # Si está en avería total y permanece en avería total
        elif estado_actual == 2 and estado_futuro == 2:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizMaquina.append(listaAux)
    matricesMaquina[accionesMaquina[2]] = np.array(matrizMaquina)
    
# Creo el problema como un mdp
mdpMaquina = dtmdp(estadosMaquina, accionesMaquina, matricesMaquina, retornosMaquina, 0.8)
print(mdpMaquina.solve(0, minimize = True, method = "value_iteration"))
print(mdpMaquina.solve(0, minimize = True, method="policy_iteration"))
