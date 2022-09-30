from multiprocessing.dummy import Pool
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['OPENBLAS_NUM_THREADS'] = '8' 

import argparse
parser = argparse.ArgumentParser(description='gridworld for various algorithms')
parser.add_argument('--type', type=str, default='dqn', required=True)
args = vars(parser.parse_args())

print(f"args['type'] : {args['type'] == 'pg'} ")

# from importlib import reload; reload(env)
if args["type"] == "dqn":
    from deep_q_learning import DQN
    dqn = DQN()
    dqn.run()


if args["type"] == 'pg':
    print('run policy gradient')
    from policy_gradient import Policy_Gradient
    pg = Policy_Gradient(n_epi=100)
    pg.run()
    pg.test()

if args["type"] == 'ac':
    print('run actor critic')
    from actor_critic import Actor_Critic
    ac = Actor_Critic(n_epi=100)
    ac.run()
    ac.test()