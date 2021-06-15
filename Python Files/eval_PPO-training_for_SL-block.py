import socket
import struct
import pickle
import numpy as np
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import argparse


class Connection:
    def __init__(self, s):
        self._socket = s
        self._buffer = bytearray()

    def receive_object(self):
        while len(self._buffer) < 4 or len(self._buffer) < struct.unpack("<L", self._buffer[:4])[0] + 4:
            new_bytes = self._socket.recv(16)
            if len(new_bytes) == 0:
                return None
            self._buffer += new_bytes
        length = struct.unpack("<L", self._buffer[:4])[0]
        header, body = self._buffer[:4], self._buffer[4:length + 4]
        obj = pickle.loads(body)
        self._buffer = self._buffer[length + 4:]
        return obj

    def send_object(self, d):
        body = pickle.dumps(d, protocol=2)
        header = struct.pack("<L", len(body))
        msg = header + body
        self._socket.send(msg)


class Env(gym.Env):
    def __init__(self, addr):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(addr)
        s.listen(1)
        clientsocket, address = s.accept()

        self._socket = clientsocket
        self._conn = Connection(clientsocket)

        self.action_space = gym.spaces.Discrete(6)# when you define the new action, change the action_space.
        self.observation_space = gym.spaces.Box(low=-2., high=2., shape=(25, ))#when you define a new observation, you should change the number of low,high and shape.

    def reset(self):
        self._conn.send_object("reset")
        msg = self._conn.receive_object()
        self.action_space = eval(msg["info"]["action_space"])
        self.observation_space = eval(msg["info"]["observation_space"])
        return msg["observation"]

    def step(self, action):
        self._conn.send_object(action.tolist())
        msg = self._conn.receive_object()
        obs = msg["observation"]
        rwd = msg["reward"]
        done = msg["done"]
        info = msg["info"]
        return obs, rwd, done, info

    def close(self):
        self._conn.send_object("close")
        self._socket.close()



parser = argparse.ArgumentParser()
parser.add_argument('--load-path', help='specify path to model that should be loaded')
args = parser.parse_args()


args.load_path = 'PPO-2D short curve open_100_3000_2__0\model_3000_steps'

addr = ("127.0.0.1", 50710)
env = Monitor(Env(addr))
obs = env.reset()

model = PPO.load(args.load_path, env=env)

cum_rwd = 0
assembly_seq = []

for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    assembly_seq.append(action)
    obs, reward, done, info = env.step(action)
    print(i, action, reward, done, info)
    if done:
        print("Return = ")
        break
        obs = env.reset()
        cum_rwd = 0
print (assembly_seq)
env.close()
