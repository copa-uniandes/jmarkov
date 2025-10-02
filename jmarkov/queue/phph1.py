import numpy as np
from jmarkov.ctbd import ctbd
from jmarkov.phase.ctph import ctph

class phph1():
    """
    Implements a PH/PH/1 queue and computes steady state metrics 
    
    The PH/PH/1 queue has phase type interarrival times, with parameters (alpha,T) of size ma,
    phase type service times, with parameters (beta,S) of size ms, 
    and 1 server.

    The class builds a quasi-birth-death chain that models this queue and uses 
    the steady state probability distribution of the chain to compute
    measures of performance such as mean number of entities in the system,
    in queue, in service, and the mean time in the system, in queue, and in service.
    """

    # phase-type representation of the inter-arrival time distribution
    IAT:ctph

    # phase-type representation of the service time distribution
    ST:ctph

    # steady state probabilities
    probs:np.array

    # number of entries in the steady state vector to compute
    n:int

    # phase-type representation of the waiting time distribution
    WT:ctph

    # phase-type representation of the response time distribution
    RT:ctph


    # initializer 
    def __init__(self, IAT:ctph, ST:ctph):
        """
        Creates a PH/PH/1 queue with 1 server, IAT inter-arrival times and ST service times
        """
        self.IAT=IAT
        self.ST = ST

        mean_iat = IAT.expected_value()
        mean_st = ST.expected_value()
        rho = mean_st/mean_iat

        if rho >= 1:
            raise ValueError(f'The load {rho} exceeds one, que queue is unstable')   

        self.n=100
        self.probs = np.full(self.n+1, np.nan)
        

    def _solve_mc(self, verbose=False):
        """
        Solves the underlying Markov chain to obtain the queue-length distribution

        Employs a method by Latouche and Ramaswami* especially devised for the PH/PH/1 queue,
        where the blocks in the underlying quasi-birth-death Markov chain are further divided in 
        subblocks.

        G. Latouche and V. Ramaswami. The PH/PH/1 queue at epochs of queue size change. 
        Queueing Systems, 25:97–114, 1997.
        """

        # arrivals
        ma = self.IAT.n_phases
        alpha = self.IAT.alpha
        # convert alpha to row vector
        alpha = alpha[np.newaxis,:]
        T = self.IAT.T
        t = -np.sum(T,axis=1)
        # convert t to column vector
        t = t[:,np.newaxis]
        avgt = self.IAT.expected_value()

        # service
        ms = self.ST.n_phases
        beta = self.ST.alpha
        # convert beta to row vector
        beta = beta[np.newaxis,:]
        S = self.ST.T
        s = -np.sum(S,axis=1)
        # convert s to column vector
        s = s[:,np.newaxis]
        avgs = self.ST.expected_value()

        mtot = ms*ma
        rho = avgs/avgt
        if rho >= 1:
            raise ValueError(f'The load {rho} exceeds one, que queue is unstable')   


        # Compute classic QBD blocks A0, A1 and A2
        A0 = np.kron(np.outer(t,alpha), np.eye(ms))
        A1 = np.kron(T,np.eye(ms))+ np.kron(np.eye(ma),S)

        # Compute QBD blocks in approach Latouche & Ramaswami
        invmA1 = np.linalg.inv(-A1)

        A0pp = np.kron(alpha,np.eye(ms))@invmA1@np.kron(t,np.eye(ms))
        A0mp = np.kron(np.eye(ma),beta)@invmA1@np.kron(t,np.eye(ms))
        A2pm = np.kron(alpha,np.eye(ms))@invmA1@np.kron(np.eye(ma),s)
        A2mm = np.kron(np.eye(ma),beta)@invmA1@np.kron(np.eye(ma),s)

        #A0n = [A0pp, np.zeros((ms,ma))], [A0mp, np.zeros((ma,ma))]]
        A0n = np.concatenate((
                np.concatenate((A0pp, np.zeros((ms,ma))), axis=1),
                np.concatenate((A0mp, np.zeros((ma,ma))), axis=1)
                ),
                axis=0
            )
        #A2n = [[np.zeros((ms,ms)), A2pm], [np.zeros((ma,ms)), A2mm]]
        A2n = np.concatenate((
                np.concatenate((np.zeros((ms,ms)), A2pm), axis=1),
                np.concatenate((np.zeros((ma,ms)), A2mm), axis=1)
                ),
                axis=0
            )

        # Compute matrix Gamma: NE corner of matrix G
        itB0 = A0n
        itB2 = A2n
        Gamma = itB2[0:ms,ms:ms+ma]
        itT = itB0
        check=1
        numit = 1
        while check > 10e-14 and numit < 10:
            itA1 = itB0@itB2 + itB2@itB0
            itB0 = np.linalg.inv(np.eye(ma+ms)-itA1)@itB0@itB0
            itB2 = np.linalg.inv(np.eye(ma+ms)-itA1)@itB2@itB2
            tmp = itT@itB2
            Gamma = Gamma + tmp[0:ms,ms:ms+ma]
            itT = itT@itB0
            check = np.linalg.norm(np.ones((ms,1))-Gamma@np.ones((ma,1)))
            numit=numit+1
            if verbose==True:
                print(f'Check after {numit} iterations: {check}')

        Gm = np.linalg.inv(np.eye(ma)-A0mp@Gamma)@A2mm
        R_Gam = A0pp@np.linalg.inv(np.eye(ms)-Gamma@A0mp)


        # Compute pi and queue length distribution
        Gstar = invmA1@np.kron(np.eye(ma),s@beta) + invmA1@np.kron(t,np.eye(ms))@Gamma@Gm@np.kron(np.eye(ma),beta)
        Rstar = A0@np.linalg.inv(-A1-A0@Gstar)

        # Compute pi_0
        pi = np.kron((1-rho)*np.linalg.inv(beta@Gamma@np.linalg.inv(-T)@np.ones((ma,1)))@beta@Gamma@np.linalg.inv(-T),beta)
        # Compute pi_1,...
        sumpi=np.sum(pi)
        numit=0
        while sumpi < 1-10**(-10) and numit < 1+self.n:
            tmp = pi[numit,:]@Rstar; #compute pi_(numit+1)
            pi = np.append(pi,[tmp],axis=0)
            pi[numit+1] = tmp
            numit=numit+1
            sumpi=sumpi+sum(pi[numit,:])
            if verbose==True:
                print(f'Accumulated mass after {numit} iterations: {sumpi}')

        self.probs[0:pi.shape[0]] = np.sum(pi,axis=1)
        self.probs[pi.shape[0]:self.n+1] = 0
        pi=np.reshape(pi, pi.shape[0]*pi.shape[1])
        if numit == 1+self.n:
            raise Warning(f'Maximum Number of Components {numit-1} reached')
        
        # wait time distribution
        sigtilde = np.linalg.inv(beta@np.linalg.inv(S)@np.ones((ms,1)))@beta@np.linalg.inv(S)
        # turn 2d array into 1d array 
        sigtilde = sigtilde[0]
        Delta = np.diag(sigtilde)
        
        wait_T = np.linalg.inv(Delta)@np.transpose(S+R_Gam@s@beta)@Delta
        theta = (-beta@np.linalg.inv(S)@np.ones((ms,1)))@np.transpose(s)@Delta
        D = np.linalg.inv(Delta)@np.transpose(R_Gam)@Delta
        wait_alpha = (1-np.linalg.inv(beta@np.linalg.inv(np.eye(ms)-R_Gam)@np.ones((ms,1)))@beta@np.ones((ms,1)))@np.linalg.inv(theta@D@np.ones((ms,1)))@theta@D
        # turn 2d array into 1d array 
        wait_alpha = wait_alpha[0]
        self.WT = ctph(wait_alpha, wait_T)
        mw = len(wait_alpha)
        # build ph representation of the response time distribution
        norm_beta = (1-wait_alpha.sum())*beta
        norm_beta = norm_beta[0]
        resp_alpha = np.concatenate((wait_alpha, norm_beta), axis=0)
        resp_exit = -wait_T@np.ones((mw,1))
        resp_T = np.concatenate((
                np.concatenate((wait_T, resp_exit@beta ), axis=1),
                np.concatenate((np.zeros((ms,mw)), S), axis=1)
                ),
                axis=0
            )
        self.RT = ctph(resp_alpha,resp_T)
    
    def number_entities_dist(self)-> float:
        """
        Computes the distribution of the number of entities in the system in steady state
        
        A quasi-birth-death chain is built and its stationary probability distribution is used 
        to compute the distribution of the number of entities in the system in steady state
        """
        if self.is_stable():
            if np.isnan(self.probs).any():
                self._solve_mc()
            return self.probs
        else:
            print('Unstable queue')
            return 0.0
    

    def mean_number_entities(self)-> np.float64:
        """
        Computes the mean number of entities in the system in steady state
        
        A quasi-birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the system in steady state
        """
        if self.is_stable():
            if np.isnan(self.probs).any():
                self._solve_mc(self.n)
            mean_num = 0 
            for i in range(self.n):
                mean_num += self.probs[i]*i

            return mean_num
        else:
            print('Unstable queue')
            return 0


    def mean_number_entities_queue(self)-> np.float64:
        """
        Computes the mean number of entities in queue in steady state
        
        A quasi-birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in queue in steady state
        """
        if self.is_stable():
            if np.isnan(self.probs).any():
                self._solve_mc(self.n)
            mean_num = 0 
            for i in range(1, self.n):
                mean_num += self.probs[i]*(i-1)

            return mean_num
        else:
            print('Unstable queue')
            return 0

    def mean_number_entities_service(self)-> np.float64:
        """
        Computes the mean number of entities in service in steady state
        
        A birth-death chain is built and its stattionary probability distribution is used 
        to compute the mean number of entities in the system in steady state
        """
        if self.is_stable():
            if np.isnan(self.probs).any():
                self._solve_mc(self.n)
            return 1 - self.probs[0]
        else:
            print('Unstable queue')
            return 0
        
    def wait_time_dist(self)-> ctph:
        """
        Computes the phase-type representation of the distribution of the waiting time in steady state
        
        A quasi-birth-death chain is built and its stationary probability distribution is used 
        to compute the phase-type representation of the distribution of the waiting time in steady state
        """
        if self.is_stable():
            if np.isnan(self.probs).any():
                self._solve_mc()
            return self.WT
        else:
            print('Unstable queue')
            return 0.0
        
    def resp_time_dist(self)-> ctph:
        """
        Computes the phase-type representation of the distribution of the response time in steady state
        
        A quasi-birth-death chain is built and its stationary probability distribution is used 
        to compute the phase-type representation of the distribution of the waiting time in steady state,
        which is then combined with the service time distribution to obtain the response time distribution
        """
        if self.is_stable():
            if np.isnan(self.probs).any():
                self._solve_mc()
            return self.RT
        else:
            print('Unstable queue')
            return 0.0


    def mean_time_system(self)-> np.float64:
        """
        Computes the mean time in the system in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the system in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        if self.is_stable():
            L = self.mean_number_entities()
            arr_rate = 1/self.IAT.expected_value()
            return L/arr_rate
        else:
            print('Unstable queue')
            return 0
    
    def mean_time_queue(self)-> np.float64:
        """
        Computes the mean time in the queue in steady state
        
        A birth-death chain is built and its stationary probability distribution is used 
        to compute the mean number of entities in the queue in steady state, which is 
        then used with Little's Law to obtain the mean time in the system in steady state
        """
        if self.is_stable():
            Lq = self.mean_number_entities_queue()
            arr_rate = 1/self.IAT.expected_value()
            return Lq/arr_rate
        else:
            print('Unstable queue')
            return 0
    
    def mean_time_service(self)-> np.float64:
        """
        Computes the mean time in service
        
        A simple relation is used to obtain the mean service time
        """
        if self.is_stable():
            return self.ST.expected_value()
        else:
            print('Unstable queue')
            return 0
    

    def utilization(self)-> np.float64:
        """
        Computes the mean server utilization
        
        The mean server utilization is computed as the ratio between the arrival rate
        and the service rate times the number of servers
        """
        rho = self.ST.expected_value()/self.IAT.expected_value()
        return rho
    
    def is_stable(self)-> bool:
        """
        Returns True if the queue is stable, False otherwise
        
        This queue is stable if the arrival rate is smaller than the 
        maximum service rate
        """
        return self.ST.expected_value() < self.IAT.expected_value() 
        