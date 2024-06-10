import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.mdp.dtmdp import dtmdp

# Variable de estado:
# X_{t}: Incidencia de la moniliasis en los cultivos de cacao en la época t

estadosAcacias = np.array(['Baja', 'Media','Alta'])

# Acciones
accionesAcacias = np.array(['Erradicar (E)','Fumigar (F)','No hacer nada (N)'])
# Retornos Inmediatos
retornosAcacias = np.array([[-450, -125, -50],
                              [-800, -1000,-400],
                              [-1000, -1500, -600]])
# Matrices de transición
matricesAcacias = {}
# Para a == Nada
matricesAcacias['Erradicar (E)'] = np.array([[0.95, 0.05, 0],
                                 [0.8, 0.15, 0.05],
                                 [0, 0.8, 0.2]])  
# Para a == Fontibón
matricesAcacias['Fumigar (F)'] = np.array([[0.8, 0.2, 0],
                                 [0.5, 0.3, 0.2],
                                 [0, 0.5, 0.5]])
# Para a == Usme
matricesAcacias['No hacer nada (N)'] = np.array([[0.6, 0.4, 0],
                                 [0.1, 0.5, 0.4],
                                 [0, 0.1, 0.9]])

# Creo el problema como un mdp
mdpAcacias = dtmdp(estadosAcacias, accionesAcacias, matricesAcacias, retornosAcacias, 0.8)
print(mdpAcacias.solve(0, method = "value_iteration"))
print(mdpAcacias.solve(0, method="policy_iteration"))