import gymnasium as gym
import numpy as np
import tqdm
import random
import math
from typing import Tuple, Dict, Any, List
import os
from collections import deque
import matplotlib.pyplot as plt


class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals=16):
        self._env = gym.make('CartPole-v1', render_mode="rgb_array")
        self.action_space = self._env.action_space
        self.intervals = intervals
        self.observation_space = gym.spaces.MultiDiscrete([intervals]*4)
        self._to_discrete = lambda x, a, b: int(min(max(0, (x-a)*self.intervals/(b-a)), self.intervals))
        
    def render(self):
        self._env.render()
    
    def reset(self):
        state, _ = self._env.reset()
        return self._discretize(state)

    def _discretize(self, state:np.array)->Tuple:
        cart_pos, cart_v, pole_angle, pole_v = state
        cart_pos = self._to_discrete(cart_pos, -2.4, 2.4)
        cart_v = self._to_discrete(cart_v, -3.0, 3.0)
        pole_angle = self._to_discrete(pole_angle, -0.5, 0.5)
        pole_v = self._to_discrete(pole_v, -2.0, 2.0)
        return (cart_pos, cart_v, pole_angle, pole_v)
    
    def step(self, action:int)->Tuple[Tuple, float, bool, Any]:
        state, reward, done, truncated, info = self._env.step(action)
        state = self._discretize(state)
        return state, reward, done, truncated, info


class QLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()
        self.buffer_pointer = 0

    
    def add_to_buffer(self, data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer]
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size
    
    def sample_batch(self):
        return random.sample(self.buffer, self.batch_size)
    
    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)
    
    def update_q(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        for state, action, reward, new_state in batch:
            self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])
    
    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                self.epsilon_decay(total_step)
                new_state, reward, done, truncated, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                if done:
                    reward = self.end_reward
                self.add_to_buffer((state, action, reward, new_state))
                self.update_q(total_step)
                self.lr_decay(total_step)
                self.save_model(i)
                state = new_state
    
    def save_model(self, i):
        if i % self.save_freq == 0:
            np.save(os.path.join(self.save_path, f'{i}.npy'), self.q)
      
          
class NStepQLearner(QLearner):
    def __init__(self, config: Dict, n: int):
        super().__init__(config)
        self.n = n
        self.trajectory_buffer = deque(maxlen=self.buffer_size)  # Buffer to store multiple trajectories
        self.current_trajectory = []  # Buffer for the current trajectory

    def pair_add_to_trajectory(self, data):
        self.current_trajectory.append(data)
            
    def trajectory_add_to_buffer(self, clear=True):
        self.trajectory_buffer.append(self.current_trajectory)
        if clear:
            self.current_trajectory = []
        if len(self.trajectory_buffer) >= self.buffer_size:
            while len(self.trajectory_buffer) >= self.buffer_size:
                self.trajectory_buffer.popleft()

    def update_q_from_batch(self):
        if len(self.trajectory_buffer) >= self.batch_size:
            batch = random.sample(self.trajectory_buffer, self.batch_size)
            for trajectory in batch:
                T = len(trajectory)
                for t in range(T):
                    if t + self.n < T:
                        G = sum(self.gamma**(i-t-1) * trajectory[i][2] for i in range(t, t+self.n))
                        G += self.gamma**self.n * self.q[trajectory[t+self.n][0]].max()
                    else:
                        G = sum(self.gamma**(i-t-1) * trajectory[i][2] for i in range(t, T))
                    state, action, _, _ = trajectory[t]
                    self.q[state][action] += self.lr * (G - self.q[state][action])
            for trajectory in batch:
                self.trajectory_buffer.remove(trajectory)

    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter, desc=f'Training with n={self.n}'):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                self.epsilon_decay(total_step)
                next_state, reward, done, truncated, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                if done:
                    reward = self.end_reward
                self.pair_add_to_trajectory((state, action, reward, next_state))
                self.trajectory_add_to_buffer(clear=False)
                self.update_q_from_batch()
                self.epsilon_decay(total_step)
                self.lr_decay(total_step)
                self.save_model(i)
                state = next_state
            self.trajectory_add_to_buffer(clear=True)

def train_q_learning():
    env_name = 'DiscreteCartPole'
    save_path = 'q_tables'
    intervals = 8
    
    env = DiscreteCartPoleEnv(intervals)
    
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    
    latest_checkpoint = 0
    
    if save_path not in os.listdir():
        os.mkdir(save_path)
    elif len(os.listdir(save_path)) != 0:
        latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
        print(f'{latest_checkpoint}.npy loaded')
        q_table = np.load(os.path.join(save_path, f'{latest_checkpoint}.npy'))
            
    trainer = QLearner({
        'env':env,
        'env_name':env_name,
        'render':True,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint+1000,
        'batch_size':128,
        'buffer_size':10000,
        'gamma':0.9,
        'update_freq':1,
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':50
    })
    trainer.train()
    
    num_episodes = 10
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = trainer.greedy(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()
        total_reward += episode_reward
    average_reward = total_reward / num_episodes
    print(f'Average Episode Reward over {num_episodes} episodes: {average_reward}')
    env.close()


def train_n_step_agents(n_values: List[int], num_runs: int = 5):
    env_name = 'DiscreteCartPole'
    save_path = 'n_step_q_tables'
    intervals = 8
    batch_size = 256

    env = DiscreteCartPoleEnv(intervals)
    
    if save_path not in os.listdir():
        os.mkdir(save_path)
    
    results = {}
    for n in n_values:
        best_average_reward = -float('inf')
        best_run = None
        print(f"Training for n={n} over {num_runs} runs...")
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs} for n={n}")
            q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
            trainer = NStepQLearner({
                'env': env,
                'env_name': env_name,
                'render': False,
                'end_reward': -1,
                'q': q_table,
                'start_iter': 0,
                'iter': 1000,
                'batch_size': batch_size,
                'buffer_size': 10000,
                'gamma': 0.9,
                'update_freq': 1,
                'epsilon_lower': 0.05,
                'epsilon_upper': 0.8,
                'epsilon_decay_freq': 200,
                'lr_lower': 0.05,
                'lr_upper': 0.5,
                'lr_decay_freq': 200,
                'save_path': save_path,
                'save_freq': 50
            }, n)
            trainer.train()
            
            # Evaluate the trained agent
            num_episodes = 10
            total_reward = 0
            for _ in range(num_episodes):
                state = trainer.env.reset()
                done = truncated = False
                episode_reward = 0
                while not done and not truncated:
                    action = trainer.greedy(state)
                    state, reward, done, truncated, _ = trainer.env.step(action)
                    episode_reward += reward
                total_reward += episode_reward
            average_reward = total_reward / num_episodes
            print(f"    Run {run} Average Episode Reward: {average_reward}")
            
            if average_reward > best_average_reward:
                best_average_reward = average_reward
                best_run = run
                best_q_table = trainer.q.copy()
        
        # Save the best q_table for current n
        if best_run is not None:
            np.save(os.path.join(save_path, f'n_{n}_best_run_{best_run}.npy'), best_q_table)
            print(f"  Best run for n={n} is run {best_run} with average reward {best_average_reward}\n")
            results[n] = best_average_reward

    # Visualization
    plt.figure(figsize=(10, 6))
    n_steps = list(results.keys())
    rewards = list(results.values())
    plt.bar(n_steps, rewards, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
    plt.xlabel('n-step')
    plt.ylabel('Average Episode Reward')
    plt.title('Best Average Rewards for n-step TD Learning')
    plt.savefig('n_step_q_learning_best_runs.png')
    plt.show()


if __name__ == '__main__':
    # train_q_learning()
    # exit()
    n_values = [1, 2, 4, 8, 16, 32]
    train_n_step_agents(n_values, num_runs=5)
