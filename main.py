import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="MMC", type=str)
parser.add_argument('--data', default="Single_OD", type=str)
args = parser.parse_args()

datalist = os.listdir("data/" + args.data)

if args.model == "IRL":
    # IRL Test
    for data0 in datalist:
        print(os.path.join('.','data',args.data,data0))
        subprocess.run(["python" , 
                        os.path.join('.','scripts','irl','demo_shortestpath.py'),
                        "--data",
                        os.path.join('.','data',args.data,data0)
                        ])
elif args.model == "MMC":
# MMC Test
    for data0 in datalist:
        print(os.path.join('.','data',args.data,data0))
        subprocess.run(["python" , 
                        os.path.join('.','scripts','behavior_clone','run_bc_mmc.py'),
                        "--data",
                        os.path.join('.','data',args.data,data0)
                        ])
elif args.model == "RNN":
    # RNN Test
    for data0 in datalist:
        print(os.path.join('.','data',args.data,data0))
        subprocess.run(["python" , 
                        os.path.join('.','scripts','behavior_clone','run_bc_rnn.py'),
                        "--data",
                        os.path.join('.','data',args.data,data0)
                        ])

