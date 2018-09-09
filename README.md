# rl_channel_allocation

Basic info
This is essentially a bin packing problem environment. It is in it's early stages of development but is fully operational.

There are two python scripts in this repistory: 
1) environment.py : This is the env which the agent exists in and implements selected actions and calculates the reward earned. 
2) qlearn.py : This is the Q-Learning algorithm and handles the acquisition of the calculation of the state-action values. 

TODO:
- Enable larger state spaces. 
- Allow for dynamic figuring out of actions. 
- Find what the best learning rate and discount factor are for this problem. 
- Add prioritisation in. 
- Add time-awareness. 
