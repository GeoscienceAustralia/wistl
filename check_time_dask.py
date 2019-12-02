from dask.distributed import Client
from wistl.tests import test_line
import time
import sys

print(sys.argv[1])
print(sys.argv[2])
client = Client(sys.argv[1])

a = test_line.TestLine1()
a.setUpClass()

line1 = a.line
line1.no_sims = int(sys.argv[2])
line1._towers = None

line1.compute_damage_prob()

# big 
tic = time.time()
line1.compute_damage_prob_sim()
print(f'Elapsed time: {time.time() - tic}')

# dask
tic = time.time()
temp = []
for i in range(line1.no_sims):
    res = client.submit(line1.compute_damage_prob_sim_given_sim, id_sim=i)
    temp.append(res)

client.gather(temp)
print(f'Elapsed time dask: {time.time() - tic}')

# dask
tic = time.time()
temp = []
for i in range(line1.no_sims):
    res = client.submit(line1.compute_damage_prob_sim_given_sim1, id_sim=i)
    temp.append(res)

client.gather(temp)
client.close()
print(f'Elapsed time dask: {time.time() - tic}')


