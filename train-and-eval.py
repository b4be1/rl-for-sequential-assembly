import socket
import struct
import pickle
import numpy as np
import os
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import argparse


class Connection:
    def __init__(self, s):
        self._socket = s
        self._buffer = bytearray()

    def receive_object(self):
        while len(self._buffer) < 4 or \
                len(self._buffer) < struct.unpack("<L", self._buffer[:4])[0] + 4:
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

        self.action_space = gym.spaces.Discrete(6)  # None
        self.observation_space = gym.spaces.Box(low=-2., high=2., shape=(37,))  # None

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, help='specify path where model should be saved',
                        default='')
    parser.add_argument('--load-path', type=str, help='specify which model should be loaded',
                        default='')
    parser.add_argument('--algorithm', type=str,
                        help='specify which algorithm should be used (PPO or DQN)', default='DQN')
    args = parser.parse_args()

    if args.save_path:
        addr = ("127.0.0.1", 50710)
        env = Monitor(Env(addr))
        env.reset()

        seed = np.random.randint(0, 1000)
        log_dir = args.save_path + '_seed' + str(seed)
        os.makedirs(log_dir)
        checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./' + str(log_dir),
                                                 name_prefix='model')
        if args.algorithm == 'DQN':
            model = DQN('MlpPolicy', env, learning_rate=3e-3, gamma=0.99, learning_starts=100,
                        train_freq=1,
                        target_update_interval=100, seed=seed, batch_size=64, verbose=1,
                        exploration_fraction=0.2, tensorboard_log="./" + str(log_dir))
            model.learn(total_timesteps=12000, log_interval=1, callback=checkpoint_callback)
        elif args.algorithm == 'PPO':
            model = PPO('MlpPolicy', env, n_steps=100, n_epochs=10, learning_rate=3e-3, gamma=0.99,
                        batch_size=64, verbose=1, tensorboard_log="./" + str(log_dir))
            model.learn(total_timesteps=12000, log_interval=1, callback=checkpoint_callback)
        else:
            raise ValueError("Invalid choice of the algorithm: use either PPO or DQN")

        env.close()

    elif args.load_path:
        addr = ("127.0.0.1", 50710)
        env = Monitor(Env(addr))
        obs = env.reset()
        if args.algorithm == 'DQN':
            model = DQN.load(args.load_path, env=env)
        elif args.algorithm == 'DQN':
            model = PPO.load(args.load_path, env=env)
        else:
            raise ValueError("Invalid choice of the algorithm: use either PPO or DQN")

        cum_rwd = 0
        for i in range(300):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cum_rwd += reward
            if done:
                obs = env.reset()
                print("Return = ", cum_rwd)
                cum_rwd = 0
        env.close()

    else:
        raise ValueError("Invalid program configuration: either specify save path to train a model "
              "or specify load path to evaluate a trained model")


if __name__ == "__main__":
    main()
