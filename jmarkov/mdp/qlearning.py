
"""Tabular Q-learning for discrete-time Markov decision processes.

Learns an optimal policy from simulated transitions, without transition
matrices, for continuing discounted problems.

Author: Juanita Valentina Cortés Vargas
"""

import numpy as np
class qlearning:
    """MDP represented through sampled transitions for tabular Q-learning."""

    def __init__(self, initial_state, feasible_actions, step,
                 discount_factor):
        if not callable(initial_state):
            raise TypeError("initial_state must be callable")

        if not callable(feasible_actions):
            raise TypeError("feasible_actions must be callable")

        if not callable(step):
            raise TypeError("step must be callable")

        if not 0 <= discount_factor < 1:
            raise ValueError("discount_factor must be between 0 and 1")

        self.initial_state = initial_state
        self.feasible_actions = feasible_actions
        # step es una función proporcionada por quien define el MDP.
        # Recibe (state, action, rng) y devuelve:
        # (next_state, immediate_return).
        # Describe cómo cambia el sistema y qué recompensa o costo produce.
        
        self.step = step
        self.discount_factor = discount_factor

        self.Q = {}

    def _initialize_state(self, state):
        """Initialize Q values for the feasible actions of a new state."""
        # Inicializar solo estados nuevos, conservando lo ya aprendido.
        if state not in self.Q:
            # Consultar las acciones permitidas y guardarlas en una tupla.
            actions = tuple(self.feasible_actions(state))

            if not actions:
                raise ValueError("state must have at least one feasible action")
            # Crear una estimación inicial de cero para cada acción factible.
            self.Q[state] = {action: 0.0 for action in actions}

    def _select_action(self, state, epsilon, rng):
        """Choose a feasible action using epsilon-greedy exploration."""

        # Verificar que epsilon sea una probabilidad válida.
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

        # Si el estado es nuevo, crear sus valores Q.
        # Si ya existe, conservar los valores aprendidos.
        self._initialize_state(state)

        # Obtener las acciones factibles guardadas para este estado.
        actions = tuple(self.Q[state])

        # Explorar con probabilidad epsilon.
        # por ejemplo si epsilon es 0.1 un número generado es 0.04 activa la exploración; uno como 0.72  selecciona  la mejor acción estimada hasta el momento.
        if rng.random() < epsilon:
            # Generar una posición aleatoria y devolver esa acción.
            return actions[int(rng.integers(len(actions)))]

        # Encontrar el mayor valor Q estimado en este estado.
        best_value = max(self.Q[state].values())

        # Reunir todas las acciones que alcanzan ese valor.
        best_actions = tuple(action for action in actions if self.Q[state][action] == best_value)

        # Elegir al azar entre las mejores acciones para resolver empates.
        return best_actions[int(rng.integers(len(best_actions)))]   
        
    # Ahora la actualización de Q-learning, en este caso es 
    # cómo una experiencia modifica el valor de la acción que tomé.
    def _update_q(self, state, action, reward, next_state, alpha):
        """Update one state-action value using the Q-learning rule."""

        # Verificar que la tasa de aprendizaje sea válida.
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 (excluded) and 1")

        # Crear las entradas de los estados si aún no existen.
        self._initialize_state(state)
        self._initialize_state(next_state)

        # Verificar que la acción sea factible en el estado actual.
        if action not in self.Q[state]:
            raise ValueError("action is not feasible in this state")

        # Recuperar la estimación actual del par estado-acción.
        current_value = self.Q[state][action]

        # Obtener el mejor valor estimado del estado siguiente.
        best_next_value = max(self.Q[next_state].values())

        # Construir el objetivo: recompensa más valor futuro descontado.
        target = reward + self.discount_factor * best_next_value

        # Calcular la diferencia entre el objetivo y la estimación actual.
        # td_Error= r+Beta * MaxQ(s',a')-Q(s,a)
        td_error = target - current_value

        # Mover la estimación actual hacia el objetivo.
        self.Q[state][action] = current_value + alpha * td_error

        return td_error
        
    def get_policy(self):
        """Return a greedy policy for the states stored in Q."""

        # Crear el diccionario que asignará una acción a cada estado.
        policy = {}

        # Recorrer los estados y sus valores Q.
        for state, action_values in self.Q.items():

            # Elegir una acción que tenga el mayor valor Q.
            #Si varias acciones empatan, este método devuelve
            #  la primera según el orden del diccionario. Durante el entrenamiento,
            #  _select_action sigue resolviendo empates al azar.
            best_action = max(action_values, key=action_values.get)

            # Guardar esa acción para el estado.
            policy[state] = best_action

        return policy
    
    def solve(self, n_steps, alpha=0.1, epsilon=0.1,
              minimize=False, seed=None):
        """Train Q-learning for a continuing discounted MDP."""

        # Validar el número de pasos de entrenamiento.
        #el numero de pasos es cuantos pasos utilizará el Qlearning para aprender
    
        if isinstance(n_steps, bool) or not isinstance(
            n_steps, (int, np.integer)
        ):
            raise TypeError("n_steps must be an integer")

        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        # Validar los parámetros antes de comenzar.
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 (excluded) and 1")

        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

        # Crear un generador aleatorio reproducible mediante la semilla.
        rng = np.random.default_rng(seed)

        # Comenzar un entrenamiento nuevo.
        self.Q = {}
        state = self.initial_state(rng)
        self._initialize_state(state)

        # Registrar cuántas veces se actualiza cada par estado-acción.
        self.visit_counts = {}

        for _ in range(n_steps):

            # Elegir una acción factible mediante epsilon-greedy.
            action = self._select_action(state, epsilon, rng)

            # Obtener una experiencia dezl modelo.
            next_state, immediate_return = self.step(state, action, rng)

            # Verificar que el retorno sea un número finito.
            if not np.isscalar(immediate_return):
                raise ValueError("immediate_return must be a scalar")

            reward = (immediate_return)

            if not np.isfinite(reward):
                raise ValueError("immediate_return must be finite")

            # Convertir costos en recompensas si buscamos minimizar.
            if minimize:
                reward = -reward

            # Actualizar Q con la experiencia obtenida.
            self._update_q(state, action, reward, next_state, alpha)

            # Contar la visita al par utilizado.
            pair = (state, action)
            self.visit_counts[pair] = self.visit_counts.get(pair, 0) + 1

            # Continuar desde el estado siguiente.
            state = next_state

        # Extraer la política y los valores estimados.
        policy = self.get_policy()
        values = {
            state: max(action_values.values())
            for state, action_values in self.Q.items()
        }

        # Reportar valores como costos cuando se minimiza.
        if minimize:
            values = {state: -value for state, value in values.items()}

        return values, policy