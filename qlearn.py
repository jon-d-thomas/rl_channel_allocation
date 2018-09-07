import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, cap, time):
        self.cap=cap
        self.time=time
        self.create_env()
        self.action_initiate()

    def create_env(self):
        self.env=np.random.randint(0,2,size=(self.cap+self.time)).reshape(self.cap,self.time)
        return self.env

    def add_block(self,c,t,x,y):
        x_poss=True
        y_poss=True
        block_pres=False
        poss=True
        if (x+c)>self.cap:
            x_poss=False
        if (t+y)>self.time:
            y_poss=True
        for j in range(x,x+c):
            for i in range(y,y+t):
                if self.env[i,j]!=0:
                    block_pres=True
                else:
                    self.env[i,j]=1
        if not x_poss or not y_poss or block_pres:
            poss=False
        return self.env,poss

    def buffer_job(self,jobs_in):
        """
        Jobs in should be a list of the dimensions, priority, and wait time
        will be in a numpy array. Where column 1=c,2=t,3=time4=time elap
        """
        self.buffer=jobs_in

    def action_initiate(self):
        """
        takes argument which is the index of the action
        for a 2x2 there are 9 possible actions.
        action space is a little messed up, coords in y,x order for some reason. I forget TODO:fix this.
        """
        self.actions_poss={0:np.array([2,2,0,0]),1:np.array([1,1,0,0]),2:np.array([1,1,0,1]),3:np.array([1,1,1,0]),4:np.array([1,1,1,1]),5:np.array([2,1,0,0]),6:np.array([1,2,0,0]),7:np.array([2,1,0,1]),8:np.array([1,2,1,0]),9:np.array([0,0,0,0])}
        return self.actions_poss

    def perform_action(self,index):
        c,d=self.buffer.shape
        done=False
        valid_q=True
        for i in range(c):
            if (np.array_equal(self.actions_poss[index][0:2],self.buffer[i,0:2])) and not done:
                state,valid_q=self.add_block(self.actions_poss[index][0],self.actions_poss[index][1],self.actions_poss[index][2],self.actions_poss[index][3])
                if valid_q==True:
                    self.buffer=np.delete(self.buffer,i,0)
                    done=True
                    break
            elif index==9:
                valid_q=True
                break
            else:#think this is the problem
                valid_q=False
        return valid_q

    def incr_time(self):
        temp=np.zeros([self.cap,self.time])
        temp[1:,:]=self.env[0:-1,:]
        self.env=temp
        self.buffer[:,2]=1
        return self.env

    def calc_reward(self):
        """
        The reward is dependent on the buffered pieces
        """
        temp=np.power(self.buffer[:,2],self.buffer[:,3])
        reward=-np.sum(temp)
        return reward

    def convert_to_state(self):
        """
        converts the state to a decimal representation
        where first 4 bits represent the state and the final 4 represented the 2 buffers.
        Related arrays are flatten and concatenated and then converted to decimal giving a
        unique repr for every possible state.
        """
        binaryrep=self.env.flatten()
        temp=self.buffer[:,0:2].flatten()
        temp[temp==1]=0
        temp[temp==2]=1
        state=np.append(binaryrep,temp)
        state=np.array_str(state)
        for i in ['[',']','.',' ']:
            state=state.replace(i,'')
        return int(state,2)

    def convert_from_state(self,state):
        state=np.fromstring(bin(state)[2:].zfill(8).replace('',' ')[1:-1],dtype=int,sep=' ')
        return state[0:4].reshape(2,2),state[4:6],state[6:]


def buffer_length_maintain(buff):
    c,d=buff.shape
    while c<2:
        buff=np.vstack([buff,[np.append(np.random.randint(1,3,size=2),np.array([1,1]))]])
        c,d=buff.shape
    return buff

def trainfn(lr1,num_epis, num_iter):
    reward_epis_list=[]
    for epis in range(num_epis):
        print(epis)
        a=Environment(2,2)
        buffer_2_load=np.array([np.append(np.random.randint(1,3,size=2),np.array([1,1])),np.append(np.random.randint(1,3,size=2),np.array([1,1]))])
        a.buffer_job(buffer_2_load)
        reward_iter_list=[]
        for iter1 in range(num_iter):
            s=a.convert_to_state()
            b = np.argmax(Q[s,:] + np.random.randn(1,10)*(1./(epis+1)))
            valid_q=a.perform_action(b)
            a.incr_time()
            reward=a.calc_reward()
            a.buffer=buffer_length_maintain(a.buffer)
            s_new=a.convert_to_state()
            if not valid_q:
                Q[s,b]=-50
                #Q[s,b]=Q[s,b] + lr1 * (-50+ y * np.max(Q[s_new,:])-Q[s,b])
            elif valid_q:
                reward=5
                if b==9:
                    reward=0
                Q[s,b]=Q[s,b] + lr1 * (reward+ y * np.max(Q[s_new,:])-Q[s,b])
                reward=Q[s,b]
            reward_iter_list.append(reward)
        reward_epis_list.append(sum(reward_iter_list)/len(reward_iter_list))
    return Q,reward_epis_list,a

Q=np.zeros([256,10])
num_epis = 5000
num_iter = 200
lr = 0.10
y = 0.9
#create env
#initiate buffer
Q,improvement,myc=trainfn(lr,100,5000)

plt.plot(improvement)
plt.show()
