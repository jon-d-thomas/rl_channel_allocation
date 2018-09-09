import numpy as np

class Environment:
    def __init__(self, cap, time):
        self.cap=cap
        self.time=time
        self.create_env()
        self.action_initiate()
        self.buffer_initiate()

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

    def buffer_initiate(self):
        """
        Jobs in should be a list of the dimensions, priority, and wait time
        will be in a numpy array. Where column 1=c,2=t,3=time4=time elap
        """
        self.buffer=np.array([np.append(np.random.randint(1,3,size=2),np.array([1,1])),np.append(np.random.randint(1,3,size=2),np.array([1,1]))])

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
            else:
                valid_q=False
        return self.incr_time(valid_q,index)


    def incr_time(self,valid,action):
        """
        on this happening, the env gets shifted down once. A reward gets calculated and the buffer gets refilled.
        """
        reward=self.calc_reward(valid,action)
        temp=np.zeros([self.cap,self.time])
        temp[1:,:]=self.env[0:-1,:]
        self.env=temp
        self.buffer[:,2]=1
        self.buffer_length_maintain()
        return reward

    def calc_reward(self,valid,action):
        """
        The reward is dependent on the buffered pieces and the capacity used
        """
        reward=(np.average(self.env[1])*1.4)+(np.average(self.env[0])*0.6)
        if valid and action!=9:
            reward+=1
        elif valid and action==9:
            reward+=-1
        elif not valid:
            reward=-50
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


    def buffer_length_maintain(self):
        c,d=self.buffer.shape
        while c<2:
            self.buffer=np.vstack([self.buffer,[np.append(np.random.randint(1,3,size=2),np.array([1,1]))]])
            c,d=self.buffer.shape
        return 0

