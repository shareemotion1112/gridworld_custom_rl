import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['OPENBLAS_NUM_THREADS'] = '8' 

from deep_q_learning import DQN



# from importlib import reload; reload(env)

dqn = DQN()

dqn.run()

