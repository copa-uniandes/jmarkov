import os.path
import sys
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import simpy
import numpy as np
import pandas as pd
from scipy.stats import norm, entropy, skew, kurtosis
from jmarkov.phase.fit.moments_ctph2 import moments_ctph2
from jmarkov.phase.ctph import ctph
from jmarkov.queue.phph1 import phph1
import itertools
import time



# ---------------------------
# LAS OPERACIONES SE ENCUENTRAN ENTRE "Debug1", "Debug2" y "Debug3"
# ---------------------------

np.random.seed(123)
def generate_correlated_exponentials(n, mean, rho):
    cov = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    mv_norm = np.random.multivariate_normal(mean=[0]*n, cov=cov)
    uniforms = norm.cdf(mv_norm)
    exponentials = -mean * np.log(uniforms)
    return exponentials


def estimate_rho_from_corr_matrix(corr_matrix):
    n = corr_matrix.shape[0]
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)


def simulate_queue(N, INTER_ARRIVAL_TIME, TASK_MEAN_DURATION, RHO, SIM_TIME):
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    all_task_durations, waiting_times, system_times = [], [], []

    def job(env, job_id, N, task_mean, rho, server):
        arrival_time = env.now
        durations = generate_correlated_exponentials(N, task_mean, rho)
        all_task_durations.append(durations)
        with server.request() as req:
            yield req
            start_service = env.now
            waiting_times.append(start_service - arrival_time)
            service_time = durations.max()
            yield env.timeout(service_time)
            completion_time = env.now
            system_times.append(completion_time - arrival_time)

    def job_generator(env, N, task_mean, interarrival, rho, server):
        job_id = 0
        while True:
            yield env.timeout(np.random.exponential(interarrival))
            env.process(job(env, job_id, N, task_mean, rho, server))
            job_id += 1

    env.process(job_generator(env, N, TASK_MEAN_DURATION, INTER_ARRIVAL_TIME, RHO, server))
    env.run(until=SIM_TIME)

    return np.array(all_task_durations), np.array(waiting_times), np.array(system_times)


def fit_phase_type_and_compare(task_matrix, waiting_times, system_times, N, INTER_ARRIVAL_TIME):

    empirical_corr_matrix = np.corrcoef(task_matrix.T)
    estimated_rho = estimate_rho_from_corr_matrix(empirical_corr_matrix)

    mean_all_tasks = task_matrix.mean()
    l1 = mean_all_tasks
    l2 = task_matrix.var() + l1*l1
    l12 = estimated_rho
    b = np.sqrt((1 - l12) * (l2 - l1*l1))
    m1 = l1 - b
    m2 = l2 - l1*l1 + m1*m1 - b*b
    #print("Llegué0")
    fitter = moments_ctph2(m1, m2)
    #print("Llegué0")
    print(m1)
    print(m2)
    PH = fitter.get_ph()
    #print("Llegué10")

    lda = 1 / INTER_ARRIVAL_TIME
    alpha = np.array([1])
    T = np.array([[-lda]])
    IAT = ctph(alpha, T)

    mu1 = 1/b
    num_tasks = N
    beta = np.zeros(num_tasks + 2)
    beta[0], beta[1] = PH.alpha[0], PH.alpha[1]
    S = np.zeros((num_tasks + 2, num_tasks + 2))
    S[0:2, 0:2] = PH.T
    S[0:2, 2:3] = -PH.T @ np.ones((2, 1))
    for i in range(num_tasks - 1):
        S[i+2, i+2] = -(num_tasks - i) * mu1
        S[i+2, i+3] = (num_tasks - i) * mu1
    S[num_tasks+1, num_tasks+1] = -mu1
    #print("Llegué1")
    ST = ctph(beta, S)
    #print("Llegué2")
    q = phph1(IAT, ST)



    WT = q.wait_time_dist()

    RT = q.resp_time_dist()



    # Compare empirical vs theoretical CDFs
    #t_max = max(system_times.max())
    t_max = system_times.max()
    dt = 0.05
    ts = np.arange(0, t_max, dt)
    print("Debug1")
    start_time = time.time()
    #theo_wait_cdf = np.array([WT.cdf(t) for t in ts])
    print("Wait time CDF comp: %s seconds" % (time.time() - start_time))

    start_time = time.time()
    theo_wait_cdf = WT.cdf(0, t_max, dt)
    print("Wait time CDF comp 2: %s seconds" % (time.time() - start_time))

    #diff_wait = np.max(theo_wait_cdf - theo_wait_cdf2)
    #print(f'Max diff wait: {diff_wait}')

    print("Debug2")
    start_time = time.time()
    #theo_resp_cdf = np.array([RT.cdf(t) for t in ts])
    print("Resp time CDF comp: %s seconds" % (time.time() - start_time))

    start_time = time.time()
    theo_resp_cdf = RT.cdf(0, t_max, dt)
    print("Resp time CDF comp: %s seconds" % (time.time() - start_time))

    #diff_resp = np.max(theo_resp_cdf - theo_resp_cdf2)
    #print(f'Max diff resp: {diff_resp}')


    print("Debug3")
    #print("A ver acá:")
    #print(len(theo_wait_cdf))
    #print(len(waiting_times))
    counts, bin_edges = np.histogram(waiting_times, bins=len(ts), density=False)

    # Convert to cumulative distribution
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]  # Normalize to 1


    # Evaluate the CDF at points ts using interpolation
    cdf_values = np.interp(ts, bin_edges[1:], cdf)  # Use upper edges for alignment


    #print(len(cdf_values))






    counts, bin_edges = np.histogram(system_times, bins=len(ts), density=False)

    # Convert to cumulative distribution
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]  # Normalize to 1

    # Evaluate the CDF at points ts using interpolation
    cdf_values2 = np.interp(ts, bin_edges[1:], cdf)  # Use upper edges for alignment
    #print("AQUÍ 1")
    #print(ts)
    #print("AQUÍ 2")
    #print(bin_edges[1:])
    #print("AQUÍ 3")
    #print(cdf)
    #print("AQUÍ 4")
    #print(cdf_values2)


    #print(len(cdf_values2))
    #print(len(theo_resp_cdf))


    ###################### Corregir las métricas #######
    ################### Utilizar métodos del RT ######################
    def metrics(sim_data, theo_cdf,indicador):
        hist, edges = np.histogram(sim_data, bins=100, range=(0, t_max), density=True)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        theo_pdf = np.gradient(theo_cdf, dt)
        theo_pdf = np.interp(bin_centers, ts, theo_pdf)
        eps = 1e-10
        p, q = hist + eps, theo_pdf + eps
        kl_div = entropy(p, q)
        if(indicador==1):
          return {
            'KL': kl_div,
            'Mean1': np.mean(sim_data),
            'Std1': np.std(sim_data),
            'Skew1': skew(sim_data),
            'Kurtosis1': kurtosis(sim_data),
            'Mean2': np.mean(theo_cdf),
            'Std2': np.std(theo_cdf),
            'Skew2': skew(theo_cdf),
            'Kurtosis2': kurtosis(theo_cdf),
            'hist': cdf_values,
            'theo_pdf': theo_wait_cdf
        }
        else:
          return {
            'KL': kl_div,
            'Mean1': np.mean(sim_data),
            'Std1': np.std(sim_data),
            'Skew1': skew(sim_data),
            'Kurtosis1': kurtosis(sim_data),
            'Mean2': np.mean(theo_cdf),
            'Std2': np.std(theo_cdf),
            'Skew2': skew(theo_cdf),
            'Kurtosis2': kurtosis(theo_cdf),
            'hist': cdf_values2,
            'theo_pdf': theo_resp_cdf
        }



    wait_metrics = metrics(waiting_times, theo_wait_cdf,1)
    resp_metrics = metrics(system_times, theo_resp_cdf,2)

    return wait_metrics, resp_metrics


# ---------------------------
# Parameter grid
# ---------------------------
#N_values = [5,10,50,100]
N_values = [1000]

#INTER_ARRIVAL_VALUES = [3,4,5]
TASK_MEAN_VALUES = [1]
#RHO_VALUES = [0.3,0.5,0.7,0.9]
RHO_VALUES = [0.3]
SIM_TIME_VALUES = [10000]
#utilizacion=[0.1,0.3,0.5,0.7,0.9]
utilizacion=[0.5]

#N_values = [3]
#INTER_ARRIVAL_VALUES = [3]
#TASK_MEAN_VALUES = [0.2]
#RHO_VALUES = [0.5]
#SIM_TIME_VALUES = [10000]



results = []

for N, ut, TM, RHO, SIM_T in itertools.product(
        N_values, utilizacion, TASK_MEAN_VALUES, RHO_VALUES, SIM_TIME_VALUES):
    HN=sum(1/i for i in range(1, N+1))
    #print(IAT)
    #print(HN)
    IAT=1/(ut/HN)
    start_time = time.time()
    task_matrix, waits, systems = simulate_queue(N, IAT, TM, RHO, SIM_T)
    print("Sim time: %s seconds" % (time.time() - start_time))
    
    print("Simulación terminada")
    wait_m, resp_m= fit_phase_type_and_compare(task_matrix, waits, systems, N, IAT)
    print("Ajuste terminado")
    results.append({
        'N': N,
        'InterArrival': IAT,
        'TaskMean': TM,
        'Rho': RHO,
        'SimTime': SIM_T,
        'KL_wait': wait_m['KL'],
        'Mean_wait1': wait_m['Mean1'],
        'Mean_wait2': wait_m['Mean2'],
        'Std_wait1': wait_m['Std1'],
        'Std_wait2': wait_m['Std2'],
        'Skew_wait1': wait_m['Skew1'],
        'Skew_wait2': wait_m['Skew2'],
        'Kurt_wait1': wait_m['Kurtosis1'],
        'Kurt_wait2': wait_m['Kurtosis2'],
        'KL_resp': resp_m['KL'],
        'Mean_resp1': resp_m['Mean1'],
        'Mean_resp2': resp_m['Mean2'],
        'Std_resp1': resp_m['Std1'],
        'Std_resp2': resp_m['Std2'],
        'Skew_resp1': resp_m['Skew1'],
        'Skew_resp2': resp_m['Skew2'],
        'Kurt_resp1': resp_m['Kurtosis1'],
        'Kurt_resp2': resp_m['Kurtosis2'],
        "counts_w": wait_m['hist'],
        "theo_w": wait_m['theo_pdf'],
        "counts_r": resp_m['hist'],
        "theo_r": resp_m['theo_pdf']
    })

df = pd.DataFrame(results)
print(df)
