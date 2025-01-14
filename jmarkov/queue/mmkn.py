import numpy as np
from jmarkov.finite_ctbd import finite_ctbd

class mmkn():
    """
    Implements an M/M/k/n queue and computes steady state metrics 
    
    The M/M/k/n queue has exponential interarrival times, with rate arr_rate,
    exponential service times, with ser_rate,
    k servers in parallel, and
    n places in total (n-k for waiting).

    The class builds a birth-death chain that models this queue and uses 
    the steady state probability distribution of the chain to compute
    measures of performance such as mean number of entities in the system,
    in queue, in service, and the mean time in the system, in queue, and in service.
    """
    # number of servers
    k:int

    # arrival rate
    arr_rate:np.float64

    # service rate
    ser_rate:np.float64

    # capacity
    n:int

    # effective arrival rate
    eff_arr_rate:np.float64

    # steady state probabilities
    probs:np.array

    # initializer 
    def __init__(self, k:int, arr_rate:np.float64, ser_rate:np.float64, n:int):
        """
        Creates an M/M/k/n queue with k servers, arr_rate arrival rate, ser_rate service rate, and capacity n
        """
        self.k=k
        self.arr_rate = arr_rate
        self.ser_rate = ser_rate
        self.n=n
        self.probs = np.full(n+1, np.nan)
        self.eff_arr_rate = 0

    def mean_number_entities(self)-> np.float64:
        """
        Computes the mean number of entities in the system in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the system in steady state
        """
        if np.isnan(self.probs).any():
            self._solve_bd_process()
        mean_num = 0 
        for i in range(self.n+1):
            mean_num += self.probs[i]*i

        return mean_num


    def mean_number_entities_queue(self)-> np.float64:
        """
        Computes the mean number of entities in queue in steady state
        
        A birth-death chain is built and its stattionary probability distribution is used 
        to compute the mean number of entities in queue in steady state
        """
        if np.isnan(self.probs).any():
            self._solve_bd_process()
        mean_num = 0 
        for i in range(self.k, self.n+1):
            mean_num += self.probs[i]*(i-self.k)

        return mean_num

    def mean_number_entities_service(self)-> np.float64:
        """
        Computes the mean number of entities in service in steady state
        
        A birth-death chain is built and its stattionary probability distribution is used 
        to compute the mean number of entities in the system in steady state
        """
        if np.isnan(self.probs).any():
            self._solve_bd_process()
        mean_num = 0 
        for i in range(self.k+1):
            mean_num += self.probs[i]*i
        for i in range(self.k+1, self.n+1):
            mean_num += self.probs[i]*self.k
        return mean_num
        
        
    def mean_time_system(self)-> np.float64:
        """
        Computes the mean time in the system in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the system in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        L = self.mean_number_entities()
        return L/self.eff_arr_rate


    def mean_time_queue(self)-> np.float64:
        """
        Computes the mean time in the queue in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the queue in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        Lq = self.mean_number_entities_queue()
        return Lq/self.eff_arr_rate


    def mean_time_service(self)-> np.float64:
        """
        Computes the mean time in service
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the queue in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        Ls = self.mean_number_entities_service()
        return Ls/self.eff_arr_rate
   
    def _solve_bd_process(self):
        """
        Computes the steady state probabilities of the Markov chain that models the M/M/k/n queue
        
        A birth-death chain is built and its stationary probability distribution is obtained
        """
        birth = np.ones(self.n)*self.arr_rate
        death = np.append(np.arange(1,self.k+1)*self.ser_rate, np.ones(self.n-self.k)*self.ser_rate*self.k)
        bd = finite_ctbd(birth, death)
        self.probs = bd.steady_state()
        self.effective_arrival_rate()

    def effective_arrival_rate(self)-> np.float64:
        """
        Computes the effective arrival rate
        
        The effective arrival rate in steady state is computed considering the steady state probabilities 
        of the underlying Markov chain
        """
        if np.isnan(self.probs).any():
            self.probs = self._solve_bd_process()
        self.eff_arr_rate = self.arr_rate*(1-self.probs[-1])

    def utilization(self)-> np.float64:
        """
        Computes the mean server utilization
        
        The mean server utilization is computed as the ratio between the arrival rate
        and the servce rate times the number of servers
        """
        rho = self.arr_rate/(self.k*self.ser_rate)
        return rho
    
    def is_stable(self):
        """
        Computes whether the queue is stable or not
        
        This queue is always stable as it has finite capacity
        """
        return True