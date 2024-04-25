import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp

# Variable de estado:
# X_{t}: Clasificación de Fontibón en la época t
# Y_{t}: Clasificación de Usme en la época t
# Z_{t}: (X_{t},Y_{t})

estadosFontibon = ['Normal', 'Alerta'] 
estadosUsme = ['Normal', 'Alerta'] 

estadosAcueducto = []
for i in estadosFontibon:
    for j in estadosUsme:
        estado = ",".join((str(i),str(j)))
        estadosAcueducto.append(estado)
estadosAcueducto = np.array(estadosAcueducto)

# Acciones
accionesAcueducto = np.array(['Nada','Fontibón','Usme'])
# Retornos Inmediatos
retornosAcueducto = np.array([[0, 90, 55],
                              [149, 239, 175],
                              [183, 180, 238],
                              [332, 329, 358]])
# Matrices de transición
matricesAcueducto = {}
# Para a == Nada
matricesAcueducto["0"] = np.array([[0.09, 0.21, 0.21, 0.49],
                                   [0, 0.3, 0, 0.7],
                                   [0, 0, 0.3, 0.7],
                                   [0, 0, 0, 1]])  
# Para a == Fontibón
matricesAcueducto["1"] = np.array([[0.21, 0.49, 0.09, 0.21],
                                   [0, 0.7, 0, 0.3],
                                   [0.24, 0.56, 0.06, 0.14],
                                   [0, 0.8, 0, 0.2]])
# Para a == Usme
matricesAcueducto["2"] = np.array([[0.21, 0.09, 0.49, 0.21],
                                   [0.24, 0.06, 0.56, 0.14],
                                   [0, 0, 0.7, 0.3],
                                   [0, 0, 0.8, 0.2]])

# Creo el problema como un mdp
mdpAcueducto = dtmdp(estadosAcueducto, accionesAcueducto, matricesAcueducto, retornosAcueducto, 0.8)
print(mdpAcueducto.solve(0, minimize = True, method = "value_iteration"))
print(mdpAcueducto.solve(0, minimize = True, method="policy_iteration"))