import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp


# Parámetros:
q = 0.6
r = 0.4
discount_factor = 0.8
# Variables de estado:
# X_{t}: Estado de la batería en la época t
# Y_{t}: Horas acumuladas de la batería en la época t
# Z_{t}: (X_{t},Y_{t})
# Espacio de estados
Sx = [i for i in range(1,6)] # [1, 2, 3, 4, 5]
Sy = [j for j in range(0,100500,500)] # [0, 500,1000, ..., 99500, 100000]

estados = []
for i in Sx:
    for j in Sy:
        estado = ",".join((str(i),str(j)))
        estados.append(estado)
estados = np.array(estados)

# Acciones
acciones = np.array([a for a in range(0,3)]) # array([0, 1, 2])--> 0: no hacer nada. 1: reparar. 2: reemplazar.

# Retornos Inmediatos
retornos = []
for estado_actual in range(0,len(estados)):
    i = int(estados[estado_actual].split(",")[0])
    j = int(estados[estado_actual].split(",")[1])
    listaAux = []
    for a in acciones:
        if a == 0 and i<5 and j<100000:
            listaAux.append(0)
        elif a == 1 and i<5 and j<100000:
            listaAux.append(1)
        elif a == 2 and i==5:
            listaAux.append(100)
        elif a == 2 and j==100000:
            listaAux.append(100)
        elif a==0 and i==5:
            listaAux.append(1000)
        elif a==0 and j==100000:
            listaAux.append(1000)
        elif a==1 and i==5:
            listaAux.append(1000)
        elif a==1 and j==100000:
            listaAux.append(1000)
        elif a==2 and i<5:
            listaAux.append(1000)
        elif a==2 and j<100000:
            listaAux.append(1000)
    retornos.append(listaAux)
retornos = np.array(retornos)

# Matrices de transición
matrices = {}
for a in acciones:
    # Para a == 0 ~ no hacer nada
    if a == 0:
        matriz = []
        for estado_actual in range(0,len(estados)):
            i = int(estados[estado_actual].split(",")[0])
            j = int(estados[estado_actual].split(",")[1])
            listaAux = []
            for estado_futuro in range(0,len(estados)):
                i_ = int(estados[estado_futuro].split(",")[0])
                j_ = int(estados[estado_futuro].split(",")[1])
                # Que se mantenga el nivel y aumenten las horas en 500
                if i_ == i and j_==j+500 and i<5 and j<99500:
                    listaAux.append((1-q)*(1-r))
                # Que aumente el nivel y aumenten las horas en 500
                elif i_==i+1 and j_==j+500 and i<5 and j<99500:
                    listaAux.append(q*r)
                # Que se mantenga el nivel y aumenten las horas en 1000
                elif i_==i and j_==j+1000 and i<5 and j<99500:
                    listaAux.append((1-q)*r)
                # Que aumente el nivel y aumenten las horas en 1000
                elif i_==i+1 and j_==j+1000 and i<5 and j<99500:
                    listaAux.append(q*(1-r))
                # Que se mantenga el nivel y el aumento de horas me haga llegar a 100000 (casos extremos)
                elif i_==i and j_==100000 and i<5 and j>99000:
                    listaAux.append(((1-q)*r)+((1-q)*(1-r)))
                # Que aumente el nivel y el aumento de horas me haga llegar a 100000 (casos extremos)
                elif i_==i+1 and j_==100000 and i<5 and j>99000:
                    listaAux.append((q*(1-r))+(q*r))
                # Que aumente o se mantenga el nivel y aumenten las horas en 500 pero estoy en 5 así que el nivel no puede empeorar más
                elif i_==i and j_==j+500 and i==5 and j<99500:
                    listaAux.append((q*r)+((1-q)*(1-r)))
                # Que aumente o se mantenga el nivel y aumenten las horas en 1000 pero estoy en 5 así que el nivel no puede empeorar más
                elif i_==i and j_==j+1000 and i==5 and j<99500:
                    listaAux.append((q*(1-r))+((1-q)*r))
                # Que aumente o se mantenga el nivel y aumenten las horas en 500 o 1000 y llegue a 100000
                elif i_==i and j_==100000 and i==5 and j>99000:
                    listaAux.append(1)
                else:
                    listaAux.append(0)
            matriz.append(listaAux)
        matrices[acciones[0]] = np.array(matriz)
    # Para a == 1 ~ hacer mantenimiento
    elif a == 1:
        matriz = []
        for estado_actual in range(0,len(estados)):
            i = int(estados[estado_actual].split(",")[0])
            j = int(estados[estado_actual].split(",")[1])
            listaAux = []
            for estado_futuro in range(0,len(estados)):
                i_ = int(estados[estado_futuro].split(",")[0])
                j_ = int(estados[estado_futuro].split(",")[1])
                # Que se acumulen 500 horas 
                if i_==i and j_==j+500 and j<99500:
                    listaAux.append(1-r)
                # Que se acumulen 1000 horas
                elif i_==i and j_==j+1000 and j<99500:
                    listaAux.append(r)
                # Que se acumulen o 500 o 1000 horas y llegue a 100000
                elif i_==i and j_==100000 and j>99000:
                    listaAux.append(1)
                else:
                    listaAux.append(0)
            matriz.append(listaAux)
        matrices[acciones[1]] = np.array(matriz)
    # Para a == 2 ~ reemplazar
    elif a == 2:
        matriz=[]
        for estado_actual in range(0,len(estados)):
            i = int(estados[estado_actual].split(",")[0])
            j = int(estados[estado_actual].split(",")[1])
            listaAux = []
            for estado_futuro in range(0,len(estados)):
                i_ = int(estados[estado_futuro].split(",")[0])
                j_ = int(estados[estado_futuro].split(",")[1])
                # Siempre que reemplazo llego a perfecto y 0 horas
                if i_==1 and j_==0:
                    listaAux.append(1)
                else:
                    listaAux.append(0)
            matriz.append(listaAux)
        matrices[acciones[2]] = np.array(matriz)
        
# Creo el problema como un mdp
mdpBateria = dtmdp(estados, acciones, matrices, retornos, discount_factor)
print(mdpBateria.solve(0, minimize=True, method = "value_iteration"))
print(mdpBateria.solve(0, minimize=True, method="policy_iteration"))