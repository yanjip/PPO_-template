

# time: 2023/12/4 9:46
# author: YanJP
# time: 2023/10/30 15:06
# author: YanJP

import  numpy as np
import para
import trans

def cal_r(p,h):
    return para.B*np.log2(1+p*(h**2)/para.n0)

def get_p_each(p,hk):
    D_m=0
    for h in hk:
        D_m+=cal_r(p,h)
    return D_m

def pre_process_action(hk, request_D):
    D_m=0
    a=0
    for h in hk:
        D_m+=cal_r(para.Pmax,h)
        a+=1
        if D_m>request_D:
            return a
    return a+100
def cal_p(D,hk):
    delta=2000
    pl=0
    ph=para.Pmax
    # print("二分法开始")
    if get_p_each(ph,hk)<D:
        return ph

    while True:
        pm=(pl+ph)/2
        D_m=get_p_each(pm,hk)
        if abs(D_m - D) <= delta:
            ans=pm
            break
        if D_m < D:
            pl=pm
        else:
            ph=pm
        # print("-----------now Power:{}---------".format(pm))
    # print("-----------now Power:{}---------".format(ans))
    return ans

#计算每个用户最多可以达到的传输速率
def get_maxrate(h_s):
    r=0
    for h in h_s:
        r+=para.B * np.log2(1 + para.Pmax * (h ** 2) / para.n0)
    return r

def test_p(p,h_s):
    r=0
    for h in h_s:
        r+=para.B * np.log2(1 + p * (h ** 2) / para.n0)
    return r
def get_quality():

    return 0
    pass

def pre_process_bitrate(link_vp,ul,ul_level):
    # if link_vp+ul/max(para.bitrates)>para.min_VP:
    #     res=ul
    # else:
    #     for i in range(ul_level,para.L):
    #         if link_vp+para.bitrates[i]/max(para.bitrates)>para.min_VP:
    #             res=i
    ul_min=np.exp(para.min_VP/link_vp)*max(para.bitrates)
    if ul*para.N_fov>=ul_min:
        return ul_level,True
    else:
        for i in range(ul_level+1,para.L):
                if para.bitrates[i]*para.N_fov>=ul_min:
                    return i,True
    return para.L-1,False#



class env_():
    def __init__(self):
        self.action_dim=para.action_dim
        self.observation_space=(para.state_dim,)
        self.UserAll=trans.generateU()
        self.reward=0
        self._max_episode_steps=para.K
        # self.request_tiles=np.random.randint(para.N_fov_low,para.N_fov)
        self.Links=para.Link
        pass
    def reset(self,):
        self.done=0
        self.index=0
        self.pos=0
        self.res_p=[]
        self.res_birate=[]
        self.carrier=[]
        self.now_h=para.h[0:para.action_dim+4,self.index]
        self.Nc_left=para.N_c
        self.Nc_left_norm = self.Nc_left/para.N_c
        obs=np.concatenate((self.now_h,np.array([self.Nc_left_norm])),axis=0)
        obs = np.concatenate((obs, np.array([self.Links.vq1[self.index]])))
        # obs=self.now_h
        self.VQcheck=[]
        return obs
        # state：[time_step, carrier_left, tile_number]  加不加上tilenumber呢，这很有影响
        pass
    def step(self,action,):
        # reward=0
        action_carrier=action[0]+1 +3 #至少选一个子载波
        bitrate_level=action[1]

        bitrate_level,VQ_check=pre_process_bitrate(self.Links.vq1[self.index],para.bitrates[bitrate_level],bitrate_level)
        self.VQcheck.append(VQ_check)
        bitrate_now=para.bitrates[bitrate_level] #in bps

        self.res_birate.append(bitrate_level)
        self.index+=1

        request_D=bitrate_now*para.N_fov

        # quality=get_quality()
        # if quality<para.quality_min:
        #     # reward-=quality
        #     pass
        maxRate=get_maxrate(self.now_h)
        # action_carrier_threshold=pre_process_action(para.h[self.pos:,self.index],request_D)  #最少的子载波数目，按照最大的发射功率算的(最大就是action_dim)
        action_carrier_threshold=pre_process_action(self.now_h,request_D)  #最少的子载波数目，按照最大的发射功率算的(最大就是action_dim)

        # if action_carrier>=action_carrier_threshold and para.action_dim >= action_carrier_threshold:
            # punish=0
            # action_carrier=
        if action_carrier<action_carrier_threshold and para.action_dim+4 >= action_carrier_threshold:
            action_carrier=action_carrier_threshold
            # else:
        if  para.action_dim+4 >= action_carrier_threshold:

            #     action_carrier=min(action_carrier,para.action_dim)
            p=cal_p(request_D,self.now_h[0:action_carrier]) #4.8M视频需要0.1245W
            self.res_p.append(p*action_carrier)
            # reward=para.get_object(p,action_carrier)  #p越小越好，子载波数越小越好  #还有种思路是固定子载波数量，然后把剩余的载波数量当做state的一部分
            # reward=-p*action_carrier
            reward=1
            # print("True")
        else:
            # punish=action-action_threshold
            # reward=punish*5
            # reward = -(para.Pmax * (action_carrier_threshold - action_carrier))  # 这样写会让agent认为子载波选的越多越好
            action_carrier=0
            # reward = -(para.Pmax * ( bitrate_level+1))  # 这样写会让agent认为子载波选的越多越好
            reward = 0 # 这样写会让agent认为子载波选的越多越好
            self.res_p.append(0)
            # action_carrier = action_carrier_threshold
        # if self.Links.get_VP_k(self.index-1,bitrate_now)<para.Vquality_min:
        #     reward=-para.Pmax*(para.L-bitrate_level)
        self.carrier.append(action_carrier)
        self.Nc_left -= action_carrier
        if self.Nc_left<0:
            reward += self.Nc_left
        # else:
        #     reward=-para.Pmax*action_carrier
        # if self.Nc_left<-0.1:
        #     p2=action_carrier
        #     reward-=p2
        self.pos+=action_carrier
        # self.now_h=para.h[self.pos:self.pos+para.action_carrier_dim,self.index]
        # self.carrier.append(action_carrier)
        # if self.index==22:
        #     x=1
        #     pass
        if self.index==para.K:
            self.done=1.0
            obs=np.array([0.0]*para.state_dim)
        else:
            self.now_h = para.h[self.pos:self.pos + para.action_dim+4, self.index]
            # obs=self.now_h
            obs = np.concatenate((self.now_h, np.array([self.Nc_left / para.N_c])), axis=0)
            obs=np.concatenate((obs,np.array([self.Links.vq1[self.index]])))
        # print(obs,self.index)
        self.action_carrier=action_carrier
        return obs, reward, self.done, None



        pass