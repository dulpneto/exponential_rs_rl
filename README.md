# Risk Sensitive with Exponential Functions in Reinforcement Learning: An Empirical Analysis


## Domains
Two different domains were implemented to analyze risk attitudes
- Two Arm Bandit Domain
- River Crossing Domain


### Implementation of Two-Arm Bandit Domain
A simple environment to evaluate how much risk the agent is willing to take by setting two different arms, Arm 0 (called Deterministic Arm) and Arm 1 (called Stochastic Arm)


### Implementation of River Crossing Domain
The River Crossing domain consists of a grid with the lower left corner as the initial state and the lower right corner as the final state. In this domain, the agent can perform 4 actions: moving north, south, east, or west.
The agent's goal is to reach the final state and for that, the agent can cross the river or walk along the river margin until reaching the bridge at the top of the grid. When the agent walks along the margin or over the bridge, she has a probability 1 of performing the chosen action. However, when the agent is in a state that represents the river, she has a chance of 75% to perform the selected action and a chance of 25% to be pushed by the river to the next state to the south.


The environment can be changed on the parameter env_type (=env). The default value is RIVER for the River Crossing domain and alternatively, TWO_ARMED can be set to run the Two-Armed domain.


## Exponential Utility Theory
Five main different exponential utility functions were implemented:
- Target - Applying exponential function on Q-Learning target
- TD - Applying exponential function on QLearning temporal difference
- SI - Applying two soft indicator function on QLearning temporal difference
- TD_TRUNC - Applying exponential function on QLearning temporal difference with truncate technique
- LSE - Applying LogSumExp strategy on target update to minimize overflow errors


This can be changed on argument bellman_update (-b)


## Risk Attitude
It can be controlled by changing the lambda argument (-l)
- Negative values are risk-averse
- Zero is risk neutral
- Positive risk prone

## Implementation of Value Iteration
Run: python3 VIRunner.py

## Implementation of Q-Learning
Run: python3 RLRunner.py -t QL -b Target -l -1.0

## Implementation of DeepQNetwork
Run: python3 RLRunner.py -t DQN -b Target -l -1.0
## Implementation of DeepQNetwork with Convolutional Network
Run: python3 RLRunner.py -t DQN_CONV -b Target -l -1.0

NOTE: This algorithm is available only for RIver Crossing Domain


## Alternative DeepQNetwork Implementation
### DQN with table cache
It refreshes a Q-table every time the target model is trained. It is useful to speed up the tests


Run: python3 RLRunner.py -t DQN_CACHED -b Target -l -1.0

Or

Run: python3 RLRunner.py -t DQN_CONV_CACHED -b Target -l -1.0




### DQN without model usage
It creates a Q-Table and uses it to learn, the model is trained by not used to update the Q-table. This implementation helps to find model approximation errors.


Run: python3 RLRunner.py -t DQN_SKIP -b Target -l -1.0

Or 

Run: python3 RLRunner.py -t DQN_CONV_SKIP -b Target -l -1.0


When running this implementation two Value tables are plotted. One from Q-table and one from model


## Execution parameters


The following parameters can be customized:


- -env or --env_type: Domain RIVER or TWO_ARMED, default RIVER
- -t or --type: The type of algorithm QL, DQN, DQN_CONV, DQN_CACHED, DQN_CONV_CACHED, DQN_SKIP, DQN_CONV_SKIP, default QL. CONV methods are not implemented in the Two Armed domain.
- -b or --bellman_update: The type of Bellman update Target, TD or LSE, default Target.
- -l or --lamb: The risk param, default to 0.0.
- -g or --gamma: The discount factor, default to 0.99.
- -a or --alpha: The learning rate, default to 0.1.
- -p or --epsilon: The epsilon, default to 0.3.
- -e or --episodes: The number of episodes, default to 150.


River exclusive
- -sh or --shape_h: The shape h size, default to 5.
- -sw or --shape_w: he shape w size, default to 4.


Two-Armed exclusive
- -a0 or --arm_0_r: The Arm 0 reward, default to 0.
- -a1 or --arm_1_mean: The Arm 1 mean reward, default to 0.