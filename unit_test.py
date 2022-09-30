
from multiprocessing.dummy import Pool
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['OPENBLAS_NUM_THREADS'] = '8' 


print('run actor critic')
from actor_critic import Actor_Critic
ac = Actor_Critic(n_epi=100)
ac.run()
ac.test()