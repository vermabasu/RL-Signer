import os
import sys
from models.utils.plotutils import *
from models.utils.utils import *
from mdp import shortestpath
from models.gail.algo.gailrnn_pytorch import GAILRNNTrain
from models.gail.network_models.infoq_rnn import infoQ_RNN
from models.gail.network_models.discriminator_rnn import Discriminator as Discriminator_rnn
from models.gail.network_models.discriminator_wgail import Discriminator as Discriminator_wgail
from models.gail.network_models.policy_net_rnn import Policy_net, Value_net, StateSeqEmb
import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
import time
import sys
 
 
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1), type=int)
    parser.add_argument('--n-episode', default=int(2000), type=int)
    parser.add_argument('--num-step1', default=int(1e4), type=int)
    parser.add_argument('--pretrain-step', default=int(0), type=int)
    parser.add_argument('-b', '--batch-size', default=int(8192), type=int)
    parser.add_argument('-nh', '--hidden', default=int(64), type=int)
    parser.add_argument('-ud', '--num-discrim-update',
                        default=int(2), type=int)
    parser.add_argument('-ug', '--num-gen-update', default=int(6), type=int)
    parser.add_argument('-lr', '--learning-rate',
                        default=float(5e-5), type=float)
    parser.add_argument('--c_1', default=float(1), type=float)
    parser.add_argument('--c_2', default=float(0.01), type=float)
    parser.add_argument('--eps', default=float(1e-6), type=float)
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--train-mode', default="value_policy", type=str)
    parser.add_argument('--data', default="data/test_data/x_1.csv", type=str)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('-w', '--wasser', action="store_true", default=False)
    parser.add_argument('-ml', '--max-length', default=int(400), type=int)
    parser.add_argument('--gangnam', default=False, action="store_true")
    parser.add_argument('--save_model', default="one_usr_all_sessions", type=str)
    parser.add_argument("--num-trajs",default=400,type=int)
    return parser.parse_args()
 
 
args = argparser()
 
 
def main(args):
    if args.gangnam:
        netins = [222, 223, 224, 225, 226, 227, 228,
                  214, 213, 212, 211, 210, 209, 208,
                  190, 189, 188, 187, 186, 185, 184,
                  167, 168, 169, 170, 171, 172, 173, 174, 175, 176]
        netouts = [191, 192, 193, 194, 195, 196, 197,
                   183, 182, 181, 180, 179, 178, 177,
                   221, 220, 219, 218, 217, 216, 215,
                   198, 199, 200, 201, 202, 203, 204, 205, 206, 207]
        env = shortestpath.ShortestPath(
            "data/gangnam_Network.txt", netins, netouts)
    else:
        start_time = time.time()
        netins = [8835]
        netouts = [3356]
 
 
        env = shortestpath.ShortestPath("data/Network_sign_v5.txt", netins, netouts)
    
    pad_idx = len(env.states)
    print("pad_idx", torch.cuda.is_available() & args.cuda)
 
    if torch.cuda.is_available() & args.cuda:
        device = torch.device("cuda")
       
    else:
        device = torch.device("cpu")
 
    ob_space = env.n_states
    act_space = env.n_actions
 
    state_dim = ob_space
    action_dim = act_space
 
    def find_state(x): return env.states.index(x) if x != -1 else pad_idx
    find_state = np.vectorize(find_state)
 
    origins = np.array(env.origins)
    origins = find_state(origins)
    origins = torch.Tensor(origins).long().to(device)
 
    policy = Policy_net(state_dim, action_dim,
                        hidden=args.hidden,
                        origins=origins,
                        start_code=env.states.index(env.start),
                        env=env,
                        disttype="categorical")
    print("policy", policy)
 
    value = Value_net(
        state_dim, origins.shape[0], hidden=args.hidden, num_layers=args.num_layers)
    print("value", value)

    if args.wasser:
        D = Discriminator_wgail(
            state_dim, origins.shape[0], hidden=args.hidden, disttype="categorical", num_layers=args.num_layers)
    else:
        D = Discriminator_rnn(
            state_dim, origins.shape[0], hidden=args.hidden, disttype="categorical", num_layers=args.num_layers)

    print(device.type == "cuda")
    if device.type == "cpu":
        policy = policy.cpu()
        value = value.cpu()
        D = D.cpu()
 
    GAILRNN = GAILRNNTrain(env=env,
                           Policy=policy,
                           Value=value,
                           Discrim=D,
                           pad_idx=pad_idx,
                           args=args)
    GAILRNN.set_device(device)

    if args.wasser:
        GAILRNN.train_discrim_step = GAILRNN.train_wasser_discrim_step
        GAILRNN.discrim_opt = torch.optim.RMSprop(
            GAILRNN.Discrim.parameters(), lr=GAILRNN.lr, eps=GAILRNN.eps)
 
    hard_update(GAILRNN.Value.StateSeqEmb, GAILRNN.Policy.StateSeqEmb)
    hard_update(GAILRNN.Discrim.StateSeqEmb, GAILRNN.Policy.StateSeqEmb)
    exp_trajs, all_states = env.generate_demonstrations( n_trajs=args.num_trajs, len_traj=400)
    model = 'one_usr_all_sessions/ModelParam_one_user_all_sessions_2400.pt'
    print("States", all_states)
    print("--- %s seconds ---" % (time.time() - start_time))
    learner_observations, learner_actions, learner_len, learner_rewards =\
            GAILRNN.unroll_trajectory2(
                num_trajs=args.n_episode, max_length=args.max_length)
    learner_observations = np.expand_dims(learner_observations, axis=1)
    print(np.shape(learner_observations))# these are predicted signature samples similary we need to train and generate for y
 
 
if __name__ == '__main__':
    args = argparser()
    main(args)
 