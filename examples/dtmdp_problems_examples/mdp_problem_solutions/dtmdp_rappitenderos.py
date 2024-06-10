import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp

# Parámetros:
lambdaP = 10
lambdaB = 20
lambdaE = 30
lambdaF = 40
lambdaTotal = lambdaP + lambdaB + lambdaE + lambdaF
lambdaPB = lambdaP + lambdaB

w = [20, 15, 5, 2.5]

# Variable de estado:
# X_{t}: tipo de encargo que recibe un rappitendero en la época t

# Espacio de Estados
estadosRappi = np.array(['Premium', 'Bundle', 'Estándar','RappiFavor'])

# Espacio de Acciones
accionesRappi = np.array(['Aceptar', 'Rechazar'])

# Retornos Inmediatos
retornosRappi = []
for i in range(0,len(estadosRappi)):
    listaAux = []
    for a in range(0,len(accionesRappi)):
        if accionesRappi[a] == 'Rechazar':
            listaAux.append(0)
        else:
            listaAux.append(w[i])
    retornosRappi.append(listaAux)
retornosRappi = np.array(retornosRappi)

# Probabilidades de Transición

# Creo el diccionario con matrices vacías
matricesRappi = {}
for a in range(0,len(accionesRappi)):
    matricesRappi[accionesRappi[a]] = np.zeros((len(estadosRappi),len(estadosRappi)))
# Recorro sobre las acciones
for a in range(0,len(accionesRappi)):
    # Recorro sobre el estado actual
    for estadoActual in range(0,len(estadosRappi)):
        # Recorro sobre el estado futuro 
        for estadoFuturo in range(0, len(estadosRappi)):
            if accionesRappi[a] == 'Aceptar':
                if estadosRappi[estadoFuturo] == 'Premium':
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = lambdaP/lambdaTotal
                elif estadosRappi[estadoFuturo] == 'Bundle':
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = lambdaB/lambdaTotal
                elif estadosRappi[estadoFuturo] == 'Estándar':
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = lambdaE/lambdaTotal
                elif estadosRappi[estadoFuturo] == 'RappiFavor':
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = lambdaF/lambdaTotal
                else:
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = 0
            elif accionesRappi[a] == 'Rechazar':
                if estadosRappi[estadoFuturo] == 'Premium':
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = lambdaP/lambdaPB
                elif estadosRappi[estadoFuturo] == 'Bundle':
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = lambdaB/lambdaPB
                else:
                    matricesRappi[accionesRappi[a]][estadoActual][estadoFuturo] = 0

# Creo el problema como un mdp
mdpRappi = dtmdp(estadosRappi, accionesRappi, matricesRappi, retornosRappi, 0.8)
print(mdpRappi.solve(0, minimize = False, method = "value_iteration"))
print(mdpRappi.solve(0, minimize = False, method="policy_iteration"))