# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import envelope
import matplotlib.pyplot as plt

class QuantumEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    
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
        self.action_space = spaces.Tuple((spaces.Box(low=-1, high=1, shape=(N, 2)),))
        self.observation_space = spaces.Tuple((spaces.Box(low=-np.inf, high=np.inf, shape=(N, 2)),))
    
    def _get_obs(self):
        return self.current_state
    
    
    def get_transformed_state(self, state):
        """
        PARAMETER
            - state : ndarray of shape (N, 2)
            - trans_info: Placeholder for the transfer matrix parameters
            - freq: Placeholder for the frequencies
        RETURN
            - transformed_a_ : ndarray of shape (N,)
            - transformed_b_ : ndarray of shape (N,)
        """
        A_ = np.fft.fft(state[:,0])
        B_ = np.fft.fft(state[:,1])
        transformer = envelope.ucsb_transformer(transfer_matrix_params=self.trans_info)

        h_matrix = np.array([[transformer.transfer_matrix[i][j](np.pi*2*f) for f in self.freq] for i in range(2) for j in range(2)]).reshape(2, 2, -1)
        h_11, h_12, h_21, h_22 = h_matrix[0][0], h_matrix[0][1], h_matrix[1][0], h_matrix[1][1]

        transformed_A_ = h_11 * A_ + h_12 * B_
        transformed_B_ = h_21 * A_ + h_22 * B_
        
        transformed_a_ = np.fft.ifft(transformed_A_)
        transformed_b_ = np.fft.ifft(transformed_B_)
        
        return np.column_stack((transformed_a_, transformed_b_))
    
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
        
        """
     
        def get_reward(self,action): 
            """
            get_transformed_state(current_state,action)으로 얻은 녀석이 innitial state와 얼마나 다른지  state ndarray (100,2)
            """
            transformed_state = self.get_transformed_state(self.current_state +action)
            
            loss = nn.L1Loss()
        
            distance = loss(torch.from_numpy(transformed_state.real),torch.from_numpy(self.current_state.real))
    
            reward = -float(distance)
            
            return reward
        
        
        done = None
        
        next_state = self.current_state + action # next state 
        
        reward = get_reward(self,action)
        
        terminated = np.array_equal(self.current_state, self.transformed_state)
        
        if terminated==True:
            done = True
            self.current_state = self.reset() 
            self.transformed_state = self.get_transformed_state(self.current_state)
            
        else:
            done = False
            self.current_state = next_state
            self.transformed_state = self.get_transformed_state(self.current_state)
            
    
        
        return next_state, reward, done, {} # 마지막 dict는 gym API에는 있는데 나는 안쓸거임
    
    def reset(self):
        """
        나중에는 이부분 generator()로 대체할거임 일단은 주어진 initial_state에 대한 deterministic policy 학습할 수 있는지 
        """
        return self.initial_state

