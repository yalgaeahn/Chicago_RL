# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import envelope
import matplotlib.pyplot as plt

class QuantumEnv(gym.Env):
    
    def __init__(self, Fs, N, initial_state,trans_info):
        """
        PARAMETER
            - Fs : sampling rate
            - N : total number of samples
            - initial_state : ndarray of shape (N,2)
        """
        self.initial_state = initial_state
        self.Fs = Fs
        self.N = N 
        self.trans_info = trans_info
        self.time = np.arange(N) / Fs  # (ns) 
        self.freq = np.fft.fftfreq(N, d=1/Fs) # (GHz)
        """
        Define action space and observation space
        They must be gymnasium.spaces objects
        """
        #Initialize the agent
        self.current_state = initial_state
        self.transformed_state = self.get_transformed_state(self.current_state)
        
        #continuous state and action space 일단 ndarray (sampling_rate, 2)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2*N,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2*N,))
    
    
    def get_transformed_state(self, state):
        """
        PARAMETER
            - state : ndarray of shape (2N,),  dtype=float32
        RETURN
            - transformed_state : ndarray of shape (2N), dtype=float32
            - 
        """
        A_ = np.fft.fft(state[:self.N])
        B_ = np.fft.fft(state[self.N:])
        transformer = envelope.ucsb_transformer(transfer_matrix_params=self.trans_info)

        h_matrix = np.array([[transformer.transfer_matrix[i][j](np.pi*2*f) for f in self.freq] for i in range(2) for j in range(2)]).reshape(2, 2, -1)
        h_11, h_12, h_21, h_22 = h_matrix[0][0], h_matrix[0][1], h_matrix[1][0], h_matrix[1][1]

        transformed_A_ = h_11 * A_ + h_12 * B_
        transformed_B_ = h_21 * A_ + h_22 * B_
        
        transformed_a_ = np.fft.ifft(transformed_A_)
        transformed_b_ = np.fft.ifft(transformed_B_)

        transformed_state = np.hstack((transformed_a_, transformed_b_))

        return transformed_state
    
    def step(self, action):
        #action을 return하는 get_action(observation)는 다른 곳에서 정의할거임
        """
        PARAMETERS
            -action: 
                an action provided by the agent to update the 
                environment state.
        
        RETURNS 

            -observation(ObsType):
                An element of the environment’s observation_space 
                as the next observation due to the agent actions. 
        
            -reward(SupportsFloat):
                The reward as a result of taking the action
        
            -terminated(bool): 
                Whether the agent reaches the terminal state,
                if true user needs to call `reset()`
            
            -truncated(bool):
                Whether the truncation condition outside the scope of the MDP is satisfied.
                if true user needs to call `reset()`
            -info(dict)
        
        """
     
        def get_reward(self,action): 
            """
            get_transformed_state(current_state,action)으로 얻은 녀석이 innitial state와 얼마나 다른지  state ndarray (100,2)
            """
            transformed_state = self.get_transformed_state(self.current_state + action)
            
            loss = nn.L1Loss()
        
            distance = loss(torch.from_numpy(transformed_state),torch.from_numpy(self.current_state))
    
            reward = -float(distance)
            
            return reward
        
        
        reward = get_reward(self,action)
        
        terminated = np.array_equal(self.current_state, self.transformed_state)
        
        next_state = self.current_state + action
        
        return next_state, reward, terminated, False, {} 
    
    def reset(self, seed=None,options=None):
        """
        option 이라는 좆같은 {} 도 return함 까먹으면안됨
        나중에는 이부분 generator()로 대체할거임 일단은 주어진 initial_state에 대한 deterministic policy 학습할 수 있는지 
        """
        if seed is not None:
            np.random.seed(seed)
        self.current_state = self.initial_state
        return self.initial_state, {}

