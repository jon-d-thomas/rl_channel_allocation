"""
Author: Jonathan Thomas
Email: jonathan.david.thomas@bristol.ac.uk
Notes: If you want to play around with env and see what actions it selects using
python -i <python-exe>.exe then you can have a look.
"""
from environment import *
import matplotlib.pyplot as plt

def trainfn(lr1,y1,num_epis, num_iter):
    """
    This function facilitates the training of the state-action value function.
    It takes arguments of lr (learning rate), num_epis (number of episodes) and
    the num_iter (number of iterations). It returns the Q values, the reward
    improvement over time and the initiated environment.
    """
    reward_epis_list=[]
    for epis in range(num_epis):
        print(epis)#just so we know where we are. Maybe change this to a loading bar?
        a=Environment(2,2)#initiate env. 2,2 is all that is currently poss
        reward_iter_list=[]
        for iter1 in range(num_iter):
            s=a.convert_to_state()#convert initialised state to an integar
            b = np.argmax(Q[s,:] + np.random.randn(1,10)*(1./(epis+1)))#select action
            reward=a.perform_action(b)#perform action
            s_new=a.convert_to_state()
            Q[s,b]=Q[s,b] + lr1 * (reward+ y1* np.max(Q[s_new,:])-Q[s,b])#perform update
            reward_iter_list.append(reward)
        reward_epis_list.append(sum(reward_iter_list)/len(reward_iter_list))
    return Q,reward_epis_list,a

#initiating at 0 seems like a good idea, but it leads to undocumented behaviour
Q=np.zeros([256,10])#initiate Q-learning matrix which is total number of states, no actions
num_epis = 100
num_iter = 5000
lr = 0.10
y = 0.9#the discount factor
#create env
#initiate buffer
Q,improvement,myc=trainfn(lr,y,num_epis,num_iter)
#plot the results
plt.plot(improvement)
plt.title('Q-learning average reward over episodes')
plt.xlabel('Episodes')
plt.ylabel('Average reward per episode')
plt.show()
