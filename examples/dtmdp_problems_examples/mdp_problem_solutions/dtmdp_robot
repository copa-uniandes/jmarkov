import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp

# Parametros:
beta = 0.3
alpha = 0.6

# Variables de estado:
# X_{t}: Nivel de carga del robot al inicio de los t-ésimos 15 minutos

# Espacio de estados
estadosRobot = [i for i in range(0,3)] # [0, 1, 2] --> 0: Alto 1: Bajo 2: Descargado

# Acciones
accionesRobot = np.array([a for a in range(0,3)]) # array([0, 1, 2])--> 0: buscar activamente 1: esperar un humano 2: recargar

# Retornos Inmediatos
retornosRobot = []
for i in range(0,len(estadosRobot)):
    listaAux = []
    for a in accionesRobot:
        # Si decide buscar y el nivel de batería es alto
        if a == 0 and i == 0:
            listaAux.append(1)
        # Si decide esperar y el nivel de batería es alto
        elif a == 1 and i == 0:
            listaAux.append(0.5)
        # Si decide recargar y el nivel de batería es alto
        elif a == 2 and i == 0:
            listaAux.append(0)
        # Si decide buscar y el nivel de batería es bajo
        elif a == 0 and i == 1:
            listaAux.append(1)
        # Si decide esperar y el nivel de batería es bajo
        elif a == 1 and i == 1:
            listaAux.append(0.5)
        # Si decide recargar y el nivel de batería es bajo
        elif a == 2 and i == 1:
            listaAux.append(0)
        # Si decide buscar y el nivel de batería es descargado (infactible)
        elif a == 0 and i == 2:
            listaAux.append(-100)
        # Si decide esperar y el nivel de batería es descargado (infactible)
        elif a == 1 and i == 2:
            listaAux.append(-100)
        # Si decide recargar y el nivel de batería es bajo
        elif a == 2 and i == 2:
            listaAux.append(-3)
        else:
            listaAux.append(0)
    retornosRobot.append(listaAux)
retornosRobot = np.array(retornosRobot)

# Matrices de transición
matricesRobot = {}
# Para a == 0
matrizRobot = []
for estado_actual in range(0,len(estadosRobot)):
    listaAux = []
    for estado_futuro in range(0,len(estadosRobot)):
        if estado_actual == 0 and estado_futuro == 0:
            listaAux.append(alpha)
        elif estado_actual == 0 and estado_futuro == 1:
            listaAux.append(1-alpha)
        elif estado_actual == 1 and estado_futuro == 1:
            listaAux.append(1-beta)
        elif estado_actual == 1 and estado_futuro == 2:
            listaAux.append(beta)
        elif estado_actual == 2 and estado_futuro == 2:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizRobot.append(listaAux)
    matricesRobot["0"] = np.array(matrizRobot)
# Para a == 1
matrizRobot = []
for estado_actual in range(0,len(estadosRobot)):
    listaAux = []
    for estado_futuro in range(0,len(estadosRobot)):
        if estado_actual == estado_futuro:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizRobot.append(listaAux)
    matricesRobot["1"] = np.array(matrizRobot)
# Para a == 2
matrizRobot = []
for estado_actual in range(0,len(estadosRobot)):
    listaAux = []
    for estado_futuro in range(0,len(estadosRobot)):
        if estado_futuro == 2:
            listaAux.append(1)
        else:
            listaAux.append(0)
    matrizRobot.append(listaAux)
    matricesRobot["2"] = np.array(matrizRobot)
    
# Creo el problema como un mdp
mdpRobot = dtmdp(estadosRobot, accionesRobot, matricesRobot, retornosRobot, 0.8)
print(mdpRobot.solve(0, method = "value_iteration"))
print(mdpRobot.solve(0, method="policy_iteration"))