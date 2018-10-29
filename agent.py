import numpy as np
from collections import defaultdict
import random

class Agent:
    
#     9.4 wit sarsa
#     eps = 0 / 20000
#     alpha = 0.4
#     gamma = 0.85
    
    eps = 100 / 20000
    alpha = 0.2
    gamma = 0.85
    
    def update_Q_sarsa(self, state, action, reward, next_state=None, next_action=None):
        current = self.Q[state][action]
        Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (self.gamma * Qsa_next)
        new_value = current + (self.alpha * (target - current))
        return new_value
    
    def update_Q_expsarsa(self, state, action, reward, next_state=None, next_action=None):
        current = self.Q[state][action]
        
        policy_s = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA) # greedy action
        Qsa_next = np.dot(self.Q[next_state], policy_s) 
    
        Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (self.gamma * Qsa_next)
        new_value = current + (self.alpha * (target - current))
        return new_value
    
    def update_with_Q_learing(self, state, action, reward, next_state=None, next_action=None):
        current = self.Q[state][action]        
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0    
        target = reward + (self.gamma * Qsa_next)
        new_value = current + (self.alpha * (target - current))
        return new_value
    
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        #Using Epsilon greedy approach
        if random.random() > self.eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return random.randint(0, self.nA -1 )
        
#         return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if not done:
            next_action = self.select_action(next_state) # epsilon-greedy action
#             print (state, action, reward, next_state, next_action)
            self.Q[state][action] = self.update_Q_expsarsa(state, action, reward, next_state, next_action)

            state = next_state     # S <- S'
            action = next_action   # A <- A'
        if done:
            self.Q[state][action] = self.update_Q_expsarsa(state, action, reward)
           
  
        
        
#         self.Q[state][action] += 1