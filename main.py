import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['OPENBLAS_NUM_THREADS'] = '8' 

import argparse
parser = argparse.ArgumentParser(description='gridworld for various algorithms')
parser.add_argument('--type', '-t', default='dqn')
args = parser.parse_args()


from deep_q_learning import DQN


# from importlib import reload; reload(env)
if args.type == "dqn":
    dqn = DQN()
    dqn.run()

