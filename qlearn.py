from environment import *
import matplotlib.pyplot as plt

def trainfn(lr1,num_epis, num_iter):
    reward_epis_list=[]
    for epis in range(num_epis):
        print(epis)
        a=Environment(2,2)
        reward_iter_list=[]
        for iter1 in range(num_iter):
            s=a.convert_to_state()
            b = np.argmax(Q[s,:] + np.random.randn(1,10)*(1./(epis+1)))
            reward=a.perform_action(b)
            s_new=a.convert_to_state()
            Q[s,b]=Q[s,b] + lr1 * (reward+ y * np.max(Q[s_new,:])-Q[s,b])
            reward_iter_list.append(reward)
        reward_epis_list.append(sum(reward_iter_list)/len(reward_iter_list))
    return Q,reward_epis_list,a

#initiating at 0 seems like a good idea, but it leads to undocumented behaviour
Q=np.zeros([256,10])
num_epis = 100
num_iter = 5000
lr = 0.10
y = 0.9
#create env
#initiate buffer
Q,improvement,myc=trainfn(lr,num_epis,num_iter)

plt.plot(improvement)
plt.title('Q-learning average reward over episodes')
plt.xtitle('Episodes')
plt.ytitle('Average reward per episode')
plt.show()
