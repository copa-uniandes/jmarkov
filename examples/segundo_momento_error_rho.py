#!/usr/bin/env python
import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import simpy
import numpy as np
from scipy.stats import norm

# ----------------
# Parameters
# ----------------
N = 3
INTER_ARRIVAL_TIME = 1.5
#INTER_ARRIVAL_TIME = 5
TASK_MEAN_DURATION = 0.8
print(f'base load: {TASK_MEAN_DURATION/INTER_ARRIVAL_TIME}')
SIM_TIME = 10000*5
RHO = 0.5

# To store results
all_task_durations = []
waiting_times = []
system_times = []

def generate_correlated_exponentials(n, mean, rho):
    cov = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    mv_norm = np.random.multivariate_normal(mean=[0]*n, cov=cov)
    uniforms = norm.cdf(mv_norm)
    exponentials = -mean * np.log(uniforms)
    return exponentials

def job(env, job_id, N, task_mean, rho, server):
    arrival_time = env.now
    durations = generate_correlated_exponentials(N, task_mean, rho)
    all_task_durations.append(durations)

    with server.request() as req:
        yield req
        start_service = env.now
        waiting_times.append(start_service - arrival_time)

        # service time = max of tasks (parallel execution)
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

# ----------------
# Run Simulation
# ----------------
env = simpy.Environment()
server = simpy.Resource(env, capacity=1)  # single server like in PH/PH/1
env.process(job_generator(env, N, TASK_MEAN_DURATION, INTER_ARRIVAL_TIME, RHO, server))
env.run(until=SIM_TIME)


# Convert to matrix
task_matrix = np.vstack(all_task_durations)



def estimate_rho_from_corr_matrix(corr_matrix):
    n = corr_matrix.shape[0]
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)

empirical_corr_matrix = np.corrcoef(task_matrix.T)
estimated_rho = estimate_rho_from_corr_matrix(empirical_corr_matrix)

mean_all_tasks = task_matrix.mean()
l1=mean_all_tasks
l2=task_matrix.var()+l1*l1
l12=estimated_rho
print(l1)
print(l2)
print(l12)

b=np.sqrt((1-l12)*(l2-l1*l1))
print(b)

m1=l1-b
print(m1)

m2=l2-l1*l1+m1*m1-b*b

print(m2)

from jmarkov.phase.fit.moments_ctph2 import moments_ctph2
from jmarkov.phase.ctph import ctph
print(m1)
print(m2)
fitter = moments_ctph2(m1,m2)
PH = fitter.get_ph()



from jmarkov.queue.phph1 import phph1
from jmarkov.phase.ctph import ctph

lda = 1 / INTER_ARRIVAL_TIME
alpha = np.array([1])
T = np.array([[-lda]])
IAT = ctph(alpha, T)

print(f'mean IAT: {IAT.expected_value()}')


num_tasks = N
mu1=1/b
beta = np.zeros(num_tasks + 2)
beta[0] = PH.alpha[0]
beta[1] = PH.alpha[1]
S = np.zeros((num_tasks + 2, num_tasks + 2))
S[0:2, 0:2] = PH.T
S[0:2, 2:3] = -PH.T@np.ones((2,1))

for i in range(num_tasks-1):
  S[i+2, i+2] = -(num_tasks - i) * mu1
  S[i+2, i + 3] = (num_tasks - i) * mu1


S[num_tasks+1, num_tasks+1] = -mu1


print(f'\nbeta: {beta}')
print(f'S:\n{S}')
ST = ctph(beta, S)

print(f'mean ST: {ST.expected_value()}')
print(f'load: {ST.expected_value()/IAT.expected_value()}')
q = phph1(IAT, ST)




WT = q.wait_time_dist()
print("Waiting time distribution:")
print(f'alpha: {WT.alpha}')
print(f'T: {WT.T}')

###########################################
RT= q.resp_time_dist()
print("Response time distribution:")
print(f'alpha: {RT.alpha}')
print(f'T: {RT.T}')






import matplotlib.pyplot as plt
import numpy as np
# ----------------
# Histogram of simulated waiting times
# ----------------
bins = 50




counts, edges = np.histogram(waiting_times, bins=bins, density=False)

# cumulative counts
cum_counts = np.cumsum(counts)

# normalize to turn it into a CDF (range [0,1])
cdf = cum_counts / cum_counts[-1]

counts=cdf



#counts/=(counts.sum()*dt)
bin_centers = (edges[:-1] + edges[1:]) / 2

# ----------------
# Theoretical distribution from jMarkov
# ----------------
t_max = max(max(waiting_times), 20)
dt = 0.05
ts = np.arange(0, t_max, dt)

#theo_pdf = np.array([WT.pdf(t) for t in ts])
theo_pdf = np.array([WT.cdf(t) for t in ts])
#theo_pdf /= np.sum(theo_pdf) * dt  # normalize
#theo_pdf /= INTENSIDAD_TRAFICO
# ----------------
# Plot
# ----------------
plt.figure(figsize=(8,5))
plt.bar(bin_centers, counts, width=edges[1]-edges[0], alpha=0.6,
        label="Simulated Waiting Times", color="orange", edgecolor="black")
plt.plot(ts, theo_pdf, 'b-', linewidth=2, label="jMarkov PDF")

plt.xlabel("Waiting Time")
plt.ylabel("Density")
plt.title("Comparison of Waiting Time Distributions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()













import matplotlib.pyplot as plt
import numpy as np
#max_task_durations = np.max(task_matrix, axis=1)  # From your simulation
#max_task_durations_ = np.pad(max_task_durations, (0, 8), constant_values=0)
sim_response_times=system_times

# ----------------
# Histogram of simulated waiting times
# ----------------
bins = 50




counts, edges = np.histogram(sim_response_times, bins=bins, density=False)

# cumulative counts
cum_counts = np.cumsum(counts)

# normalize to turn it into a CDF (range [0,1])
cdf = cum_counts / cum_counts[-1]

counts=cdf



#counts/=(counts.sum()*dt)
bin_centers = (edges[:-1] + edges[1:]) / 2

# ----------------
# Theoretical distribution from jMarkov
# ----------------
t_max = max(max(sim_response_times), 20)
dt = 0.05
ts = np.arange(0, t_max, dt)

#theo_pdf = np.array([WT.pdf(t) for t in ts])
theo_pdf = np.array([RT.cdf(t) for t in ts])
#theo_pdf /= np.sum(theo_pdf) * dt  # normalize
#theo_pdf /= INTENSIDAD_TRAFICO
# ----------------
# Plot
# ----------------
plt.figure(figsize=(8,5))
plt.bar(bin_centers, counts, width=edges[1]-edges[0], alpha=0.6,
        label="Simulated Service Times", color="orange", edgecolor="black")
plt.plot(ts, theo_pdf, 'b-', linewidth=2, label="jMarkov PDF")

plt.xlabel("Service Time")
plt.ylabel("Density")
plt.title("Comparison of Service Time Distributions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
