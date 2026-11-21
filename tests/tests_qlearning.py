import os.path
import sys
import unittest
import numpy as np

# Permitir importar la copia de jMarkov de este repositorio.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.mdp.qlearning import qlearning
from jmarkov.mdp.dtmdp import dtmdp


class TestQLearning(unittest.TestCase):
    def test_mantenimiento(self):
        """Una máquina: pagar mantenimiento evita una reparacion futura.

        Estados: normal y averiado. Acciones: mantener e ignorar.
        Normal + mantener: costo 2, permanece normal.
        Normal + ignorar: costo 0, pasa a averiado.
        Averiado + cualquier accion: reparacion obligatoria de 10, vuelve normal.
        Objetivo: minimizar el costo descontado, con beta = 0.9.
        Referencia esperada: valores 20 y 28; mantener en normal.
        """
        # 1. DESCRIBIR EL PROBLEMA PARA Q-LEARNING.
        def initial_state(rng):
            # Comenzamos con la maquina funcionando.
            return "normal"

        def feasible_actions(state):
            # En este ejemplo ambas acciones estan disponibles en cada estado.
            return ("mantener", "ignorar")

        def step(state, action, rng):
            # Devolver (estado siguiente, costo inmediato).
            if state == "normal":
                if action == "mantener":
                    return "normal", 2.0
                return "averiado", 0.0
            return "normal", 10.0

        # 2. ENTRENAR CON NUESTRA CLASE QLEARNING.
        model = qlearning(initial_state, feasible_actions, step, discount_factor=0.9)
        values, policy = model.solve(
            n_steps=50_000, alpha=0.1, epsilon=0.2, minimize=True, seed=42)

        # 3. REPRESENTAR EL MISMO MDP MEDIANTE MATRICES PARA DTMDP.
        # Filas = estado actual; columnas de transicionn = estado siguiente.
        states = np.array(["normal", "averiado"])
        actions = np.array(["mantener", "ignorar"])
        matrices = {
            "mantener": np.array([[1.0, 0.0], [1.0, 0.0]]),
            "ignorar": np.array([[0.0, 1.0], [1.0, 0.0]])
        }
        # Filas = estados; columnas de costos = acciones, en el orden anterior.
        costs = np.array([[2.0, 0.0], [10.0, 10.0]])
        reference = dtmdp(states, actions, matrices, costs, 0.9)
        reference_values, reference_policy = reference.solve(
            tolerance=1e-10, minimize=True, method="value_iteration")

        # 4. COMPARAR LOS RESULTADOS DE LOS DOS ALGORITMOS.
        print("\nMantenimiento")
        print("Q-learning:", values, policy)
        print("Value Iteration:", reference_values, reference_policy)
        for position, state in enumerate(states):
            # Q-learning devuelve un diccionario; dtmdp devuelve un arreglo.
            learned_value = values[state]
            reference_value = reference_values[position]
            self.assertAlmostEqual(learned_value, reference_value, delta=1e-6)
        self.assertEqual(policy["normal"], reference_policy["normal"])
        self.assertEqual(policy["normal"], "mantener")
        # En averiado ambas acciones sonptimas: aceptamos cualquiera.
        self.assertIn(policy["averiado"], ("mantener", "ignorar"))

    def test_maximizacion(self):
        """Una maquina alterna entre dos turnos y permite elegir una tarea.

        Estados 0 y 1 representan los turnos; acciones 0 y 1 son tareas.
        El siguiente turno siempre es el otro, cualquiera que sea la tarea.
        Recompensas: turno 0 -> tareas 0 y 1 dan 3 y 1;
                    turno 1 -> tareas 0 y 1 dan 2 y 4.
        Objetivo: maximizar, beta = 0.8. Poli­tica esperada: {0: 0, 1: 1}.
        """
        # 1. DESCRIBIR EL PROBLEMA PARA Q-LEARNING.
        def initial_state(rng):
            return 0

        def feasible_actions(state):
            # Las acciones son nÃºmeros; la clase tambiÃ©n admite estas etiquetas.
            return (0, 1)

        def step(state, action, rng):
            next_state = 1 - state
            if state == 0:
                reward = 3.0 if action == 0 else 1.0
            else:
                reward = 2.0 if action == 0 else 4.0
            return next_state, reward

        # 2. RESOLVER CON Q-LEARNING, AHORA MAXIMIZANDO RECOMPENSAS.
        model = qlearning(initial_state, feasible_actions, step, discount_factor=0.8)
        values, policy = model.solve(
            n_steps=10_000, alpha=0.1, epsilon=0.2, minimize=False, seed=42)

        # 3. RESOLVER EL MISMO MDP CON VALUE ITERATION.
        states = np.array([0, 1])
        actions = np.array([0, 1])
        matrices = {
            0: np.array([[0.0, 1.0], [1.0, 0.0]]),
            1: np.array([[0.0, 1.0], [1.0, 0.0]])
        }
        rewards = np.array([[3.0, 1.0], [2.0, 4.0]])
        reference = dtmdp(states, actions, matrices, rewards, 0.8)
        reference_values, reference_policy = reference.solve(
            tolerance=1e-10, minimize=False, method="value_iteration")

        # 4. COMPARAR VALORES Y ACCIONES Ã“PTIMAS EN CADA TURNO.
        print("\nMaximizacion")
        print("Q-learning:", values, policy)
        print("Value Iteration:", reference_values, reference_policy)
        for position, state in enumerate(states):
            self.assertAlmostEqual(values[state], reference_values[position], delta=1e-6)
            self.assertEqual(policy[state], reference_policy[state])
        self.assertEqual(policy, {0: 0, 1: 1})

    def test_transiciones_aleatorias(self):
        """Caso de dos estados usado en tests_dtmdp.py del repositorio.

        Estados 0 y 1; acciones '0' y '1'; costos [[3, 1], [2, 3]].
        El estado siguiente es aleatorio y depende del estado y de la accion.
        Objetivo: minimizar, beta = 0.8.
        Referencia: valores 7.8125 y 8.75; poli­tica {0: '1', 1: '0'}.
        Comprueba el muestreo estocÃ¡stico con tres semillas reproducibles.
        """
        # DATOS DEL MDP, COMPARTIDOS POR AMBAS REPRESENTACIONES.
        states = np.array([0, 1])
        actions = np.array(["0", "1"])
        matrices = {
            "0": np.array([[0.5, 0.5], [1/3, 2/3]]),
            "1": np.array([[0.25, 0.75], [2/3, 1/3]])
        }
        costs = np.array([[3.0, 1.0], [2.0, 3.0]])

        # 1. FUNCIONES QUE RECIBIRa Q-LEARNING.
        def initial_state(rng):
            return 0

        def feasible_actions(state):
            return ("0", "1")

        def step(state, action, rng):
            # Obtener la fila de probabilidades del estado y accion actuales.
            probabilities = matrices[action][state]
            # Sortear 0 o 1 segun esas probabilidades.
            next_state = int(rng.choice([0, 1], p=probabilities))
            # La accion '0' corresponde a columna 0 y '1' a columna 1.
            action_position = int(action)
            cost = costs[state, action_position]
            return next_state, cost

        # 2. CALCULAR UNA SOLUCION DE REFERENCIA CON DTMDP.
        reference = dtmdp(states, actions, matrices, costs, 0.8)
        reference_values, reference_policy = reference.solve(
            tolerance=1e-10, minimize=True, method="value_iteration")

                # 3. ELEGIR DIEZ SEMILLAS DISTINTAS AL AZAR.
        # Este generador permite repetir la misma selección de semillas.
        selector = np.random.default_rng(20261015)
        seeds = selector.choice(1_000_000, size=10, replace=False)

        # Guardar el error promedio de cada ejecución.
        mean_errors = []

        # Guardar todos los errores individuales para obtener el máximo.
        all_errors = []

        # Guardar cuántas ejecuciones producen la política de referencia.
        matching_policies = 0

        # 4. ENTRENAR Y MEDIR ERRORES PARA CADA SEMILLA.
        for seed in seeds:
            with self.subTest(seed=int(seed)):
                model = qlearning(
                    initial_state, feasible_actions, step, 0.8
                )

                values, policy = model.solve(
                    n_steps=100_000,
                    alpha=0.01,
                    epsilon=0.2,
                    minimize=True,
                    seed=int(seed)
                )

                print("\nSemilla:", int(seed))
                print("Q-learning:", values, policy)
                print("Value Iteration:",
                      reference_values, reference_policy)

                # Errores de los estados de esta ejecución.
                state_errors = []

                for position, state in enumerate(states):
                    learned_value = values[state]
                    reference_value = reference_values[position]

                    # Diferencia absoluta entre ambos valores.
                    error = abs(learned_value - reference_value)

                    state_errors.append(error)
                    all_errors.append(error)

                    print(
                        "Estado:", state,
                        "| Error absoluto:", error
                    )

                # Promediar los errores de los dos estados.
                mean_error = float(np.mean(state_errors))
                mean_errors.append(mean_error)

                print("Error absoluto medio:", mean_error)

                # Registrar si coincide la política completa.
                same_policy = all(
                    policy[state] == reference_policy[state]
                    for state in states
                )

                if same_policy:
                    matching_policies += 1

                # Mantener las comprobaciones automáticas de la prueba.
                self.assertEqual(policy, reference_policy)
                self.assertEqual(
                    sum(model.visit_counts.values()),
                    100_000
                )

        # 5. RESUMIR LOS ERRORES DE LAS DIEZ EJECUCIONES.
        print("\nRESUMEN DE LAS DIEZ SEMILLAS")
        print("Promedio de los errores medios:",
              float(np.mean(mean_errors)))
        print("Máximo error absoluto:",
              float(np.max(all_errors)))
        print("Políticas coincidentes:",
              matching_policies, "de", len(seeds))


# Ejecutar automaticamente los tres metodos cuyo nombre empieza por test_.
if __name__ == "__main__":
    unittest.main()