#-*-coding:utf-8 -*-
import pdb
import numpy as np
import numpy as np
import matlab.engine
from matplotlib import pyplot as plt

ra=matlab.engine.start_matlab()
path='/root/ddpg/DDPG-master/env/'#'D:\cobalt_DDPG_env-small'  m文件所在路径
ra.path(ra.path(),path)

lx=5
ly=6
amax=np.array([1,1,1])
amin=np.array([0,0,0])
dtbmin=np.array([0,0,0])
dtbmax=np.array([1,1,1])

def show(sol,tspan):
	name=['Zn+','Cu+','Co+','H+','As+']
	for i in range(5):
		plt.figure(i);
		plt.plot(tspan,sol[:,i])
		plt.xlabel('time/min')
		plt.title(name[i])
	plt.figure(5)
	plt.plot(tspan,sol[:,-1])#E
	plt.xlabel('time/min')
	plt.title('E')
	plt.show()

class spaces:
	def __init__(self,high,low):
		self.high=high
		self.low=low
class spec:
	def __init__(self,limit):
		self.timestep_limit=limit

class cobalt_removal(object):
	def __init__(self):
		self.action_space=spaces(high=amax,low=amin)	
		self.t=0.0
		self.dt=10.0
		self.max_time=60*100000
		timestep_limit=10
		self.spec=spec(timestep_limit)
		zstart=np.loadtxt(path+'/state_initial.txt')
		self.state_zero=np.hstack((zstart,np.array([0.5,0.5,0.5])))
		self.state=self.state_zero
		obrange=np.loadtxt(path+'/obrange.txt')
		zmax,zmin=obrange[0,:],obrange[1,:]
		self.observation_space=spaces(high=np.hstack([zmax,dtbmax]),low=np.hstack([zmin,dtbmin]))

	def step(self,action):
		#这里仅考虑固定的扰动--扰动的改变和状态监测是同频率的-
		#-实际上是把扰动建模为不可观测的状态
		state=self.state
		t=self.t
		dt=self.dt
		dtb=state[lx+ly+1:]
		x=state[:lx+ly+1]
		x1,action1,dtb1=matlab.double(x.tolist()),matlab.double(action.tolist()),matlab.double(dtb.tolist())
		xnew,reward,sol,tspan=ra.cobalt(x1,action1,dtb1,dt,nargout=4)
		xnew,reward,sol,tspan=np.array(xnew).flatten(),reward,np.array(sol),np.array(tspan).flatten()
		eta=0.6
		dtbnew=(1-eta)*dtb+eta*(dtbmin+(dtbmax-dtbmin)*np.random.uniform(0,1,3))#平滑随机产生下一代数据--关键影响变量
		state_next=np.hstack((xnew,dtbnew))
		if self.t>=self.max_time:
			done=True
		else:
			done=False
		info=(sol,tspan)

		self.t+=self.dt
		self.state=state_next

		return state_next,reward,done,info

	def reset(self):
		self.state=self.state_zero
		self.t=0
		self.done=False
		return self.state

if False:
	env=cobalt_removal()
	#print 'cobalt_level=',cobalt_level
	for i in range(10):
		#pdb.set_trace()
		action=amin+(amax-amin)*np.random.uniform(0,1,3)
		state_next,reward,done,info=env.step(action)
		print reward
		sol,tspan=info
		if False:
			show(sol,tspan)
		name=['Zn+','Cu+','Co+','H+','As+']
		for i in [2]:
			plt.figure(i)
			plt.plot(tspan,sol[:,i])
			plt.xlabel('time/min')
			plt.title(name[i])
		plt.show()













	






	
