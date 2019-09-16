import numpy as np
from scipy import special as sp
from scipy import stats
import math
from cvxpy import*
from cvxopt import solvers,matrix,spmatrix,spdiag
import mosek
class init_conds:
 def __init__(self,base_v,base_w,base_x,base_y,base_gail,lanes,center_v=0.,scale_v=1.,center_w=-np.pi/8.,scale_w=np.pi/4.,center_x=90.,scale_x=20.,center_y=-1.5,scale_y=3.,exp_lb=1e-1,exp_ub=50.,beta_lb=.5,beta_ub=5.,norm_mean_radius=2,norm_var_radius=.5,mv_norm_mean_radius=1.):
  assert(base_x.shape[0]==base_y.shape[0]==base_v.shape[0]==base_w.shape[0])
  self.model_v='beta'
  self.model_w='beta'
  self.model_x='beta'
  self.model_y='beta'
  self.model_gail='mv_norm'
  self.center_v=center_v
  self.center_w=center_w 
  self.center_x=center_x 
  self.center_y=center_y
  self.scale_v=scale_v
  self.scale_w=scale_w
  self.scale_x=scale_x
  self.scale_y=scale_y
  self.lanes=lanes
  self.norm_mean_radius=norm_mean_radius 
  self.norm_var_radius=norm_var_radius 
  self.exp_lb=exp_lb
  self.exp_ub=exp_ub
  self.beta_lb=beta_lb
  self.beta_ub=beta_ub
  self.mv_norm_mean_radius=mv_norm_mean_radius 
  self.num_agents=base_x.shape[0]
  self.dim_gail=base_gail.shape[1]
  self.base_v=base_v 
  self.base_w=base_w 
  self.base_x=base_x 
  self.base_y=base_y 
  self.base_gail=base_gail 
  self.nat_base_v=self.exp_to_nat(self.model_v,base_v)
  self.nat_base_w=self.exp_to_nat(self.model_w,base_w)
  self.nat_base_x=self.exp_to_nat(self.model_x,base_x)
  self.nat_base_y=self.exp_to_nat(self.model_y,base_y)
  self.nat_base_gail=self.exp_to_nat(self.model_gail,base_gail,if_gail=True)
  self.base_log_partition_v,_ =self.compute_log_partition(self.model_v,self.nat_base_v)
  self.base_log_partition_w,_ =self.compute_log_partition(self.model_w,self.nat_base_w)
  self.base_log_partition_x,_ =self.compute_log_partition(self.model_x,self.nat_base_x)
  self.base_log_partition_y,_ =self.compute_log_partition(self.model_y,self.nat_base_y)
  self.base_log_partition_gail,_ =self.compute_log_partition(self.model_gail,self.nat_base_gail,if_gail=True)
 """ 
  exp_to_nat(self, model, param):
  Maps mean parametrized exponential familty parameters to their
  natural exponential family parametrizations.
  param is a (num_agents)-by-(num_param) matrix 
  (num_agents-by-(d+1)-by-d matrix for multivariate normal)
  returns param sized tensor
  """ 
 def exp_to_nat(self,model,param,if_gail=False):
  if model=='norm':
   eta_one=param[:,0]/param[:,1]
   eta_two=-.5/param[:,1]
   return np.vstack((eta_one,eta_two)).transpose()
  elif model=='beta':
   return param
  elif model=='expo':
   return-param
  elif model=='mv_norm':
   nat_param=np.empty(param.shape)
   if if_gail:
    nat_param[1:,:]=-0.5*np.linalg.inv(param[1:,:])
    nat_param[0,:]=np.matmul(-2*nat_param[1:,:],param[0,:])
   else:
    dd=param.shape[2]
    for a in xrange(self.num_agents):
     nat_param[a,1:,:]=-0.5*np.linalg.inv(param[a,1:,:])
     nat_param[a,0,:]=np.matmul(-2*nat_param[a,1:,:],param[a,0,:])
   return nat_param
  else:
   raise ValueError('model '+model+' type not supported') 
 def nat_to_exp(self,model,nat_param,if_gail=False):
  if model=='norm':
   return np.vstack((-.5*nat_param[:,0]/nat_param[:,1],-.5/nat_param[:,1])).transpose()
  elif model=='beta':
   return nat_param
  elif model=='expo':
   return-nat_param
  elif model=='mv_norm':
   param=np.empty(nat_param.shape)
   if if_gail:
    param[1:,:]=-.5*np.linalg.inv(nat_param[1:,:])
    param[0,:]=np.matmul(param[1:,:],nat_param[0,:])
   else:
    dd=nat_param.shape[2]
    for a in xrange(self.num_agents):
     param[a,1:,:]=-.5*np.linalg.inv(nat_param[a,1:,:])
     param[a,0,:]=np.matmul(param[a,1:,:],nat_param[a,0,:])
   return param
  else:
   raise ValueError('model '+model+' type not supported')
 """ compute_suff(self, model, obs):
  Compute sufficient statistics for the model families.  Given
  num_agents-by-N matrix of observations obs, returns
  num_agents-by-p-by-N tensor. 
  For gail, given dim_gail-by-N matrix of observations for gail
  weights, returns d+1-by-d-N matrix.
  Given num_agents-N-by-d matrix of
  observations obs, return num_agents-by-(d+1)-by-d-by-N
  """ 
 def compute_suff(self,model,obs,if_gail=False):
  num_agents=self.num_agents
  if model=='norm':
   sample_size=obs.shape[1]
   return np.concatenate((obs.reshape(num_agents,1,sample_size),np.power(obs,2.).reshape(num_agents,1,sample_size)),axis=1)
  elif model=='beta':
   sample_size=obs.shape[1]
   return np.concatenate((np.log(obs).reshape(num_agents,1,sample_size),np.log(1.-obs).reshape(num_agents,1,sample_size)),axis=1)
  elif model=='expo':
   sample_size=obs.shape[1]
   return obs.reshape(num_agents,1,sample_size)
  elif model=='mv_norm':
   if if_gail:
    sample_size=obs.shape[1]
    suff_stat=np.empty((self.dim_gail+1,self.dim_gail,sample_size))
    for s in xrange(sample_size):
     suff_stat[0,:,s]=obs[:,s]
     suff_stat[1:,:,s]=np.outer(obs[:,s],obs[:,s])
   else:
    sample_size=obs.shape[1]
    dd=obs.shape[2]
    suff_stat=np.empty((num_agents,dd+1,dd,sample_size))
    for a in xrange(num_agents):
     for s in xrange(sample_size):
      suff_stat[a,0,:,s]=obs[a,s,:]
      suff_stat[a,1:,:,s]=np.outer(obs[a,s,:],obs[a,s,:])
   return suff_stat
  else:
   raise ValueError('model type not supported')
 """
  compute_log_partition(self, model, nat_param)
  Compute log partition function at the natural parameter nat_param
  and return a num_agents-dim vector of log_partition function values
  grad_log_partition is a num_agents-by-p matrix
  """ 
 def compute_log_partition(self,model,nat_param,if_gail=False):
  if model=='norm':
   log_partition =-.25* np.power(nat_param[:,0],2.)/nat_param[:,1] -.5*np.log(-2*nat_param[:,1])
   grad_log_partition =np.vstack((-.5*nat_param[:,0]/nat_param[:,1],.25*np.power(nat_param[:,0]/nat_param[:,1],2.)-.5/nat_param[:,1])).transpose()
  elif model=='beta': 
   log_partition=np.log(sp.gamma(nat_param[:,0])) +np.log(sp.gamma(nat_param[:,1])) -np.log(sp.gamma(nat_param[:,0]+nat_param[:,1]))
   grad_log_partition =np.vstack((sp.polygamma(0,nat_param[:,0]),sp.polygamma(0,nat_param[:,1]))).transpose() -np.array(sp.polygamma(0,nat_param[:,0]+nat_param[:,1])).reshape(self.num_agents,1)
  elif model=='expo':
   if(nat_param>=0).any():
    raise ValueError('nat_param cannot be nonnegative for exponential')
   log_partition=-np.log(-nat_param).reshape(self.num_agents)
   grad_log_partition=(-1./nat_param)
  elif model=='mv_norm':
   param=self.nat_to_exp('mv_norm',nat_param,if_gail=if_gail)
   if if_gail:
    log_partition =np.array([.5*np.dot(nat_param[0,:],param[0,:])-.5*np.linalg.slogdet(-2*nat_param[1:,:])[1]])
    grad_log_partition =np.vstack((param[0,:],param[1:,:]+np.outer(param[0,:],param[0,:])))
   else:
    dd=param.shape[2]
    log_partition=np.empty(self.num_agents)
    grad_log_partition=np.empty((self.num_agents,dd+1,dd))
    for a in xrange(self.num_agents):
     log_partition[a] =.5*np.dot(nat_param[a,0,:],param[a,0,:]) -.5*np.slogdet(-2*nat_param[a,1:,:])[1]
     grad_log_partition[a,:,:] =np.vstack((param[a,0,:],param[a,1:,:]+np.outer(param[a,0,:],param[a,0,:])))
  else:
   raise ValueError('model type not supported')
  return(log_partition,grad_log_partition)
 def compute_hessian_log_partition(self,model,nat_param):
  dd=nat_param.size
  hessian=np.zeros((dd,dd))
  inds=np.arange(self.num_agents)*nat_param.shape[1]
  if model=='norm':
   first=-.5 /nat_param[:,1]
   common=.5*nat_param[:,0]/np.power(nat_param[:,1],2.)
   second=.5*(nat_param[:,1]-np.power(nat_param[:,0],2.)) /np.power(nat_param[:,1],3.)
   hessian[:self.num_agents,:self.num_agents]=np.diag(first)
   hessian[self.num_agents:,self.num_agents:]=np.diag(second)
   hessian[:self.num_agents,self.num_agents:]=np.diag(common)
   hessian[self.num_agents:,:self.num_agents]=np.diag(common)
  elif model=='beta':
   first=sp.polygamma(1,nat_param[:,0])
   second=sp.polygamma(1,nat_param[:,1])
   common=sp.polygamma(1,nat_param[:,0]+nat_param[:,1])
   hessian[:self.num_agents,:self.num_agents]=np.diag(first-common)
   hessian[self.num_agents:,self.num_agents:]=np.diag(second-common)
   hessian[:self.num_agents,self.num_agents:]=np.diag(-common)
   hessian[self.num_agents:,:self.num_agents]=np.diag(-common) 
  else:
   raise ValueError('model type not supported')
  return hessian
 def sample_obs(self,model,nat_param,sample_size,if_gail=False):
  param=self.nat_to_exp(model,nat_param,if_gail=if_gail)
  num_agents=self.num_agents
  obs=np.empty((num_agents,sample_size))
  if model=='norm':
   for a in xrange(num_agents):
    obs[a,:]=np.random.normal(param[a,0],np.sqrt(param[a,1]),size=sample_size)
  elif model=='beta':
   for a in xrange(num_agents):
    obs[a,:]=np.random.beta(param[a,0],param[a,1],size=sample_size)
  elif model=='expo':
   for a in xrange(num_agents):
    obs[a,:]=np.random.exponential(1/param[a,0],size=sample_size)
  elif model=='mv_norm':
   if if_gail:
    obs=np.random.multivariate_normal(param[0,:],param[1:,:],size=sample_size).transpose()
   else:
    dd=nat_param.shape[1]
    obs=np.empty((num_agents,sample_size,dd))
    for a in xrange(num_agents):
     obs[a,:,:] =np.random.multivariate_normal(param[a,0,:],param[a,1:,:],size=sample_size)
  else:
   raise ValueError('Option '+model+' not supported.')
  return obs
 """compute_pdf(self, model, nat_param, obs, log_partition):
  Given a num_agents-by-N matrix or d-by-N matrix of observations obs,
  return a num_agents-by-N matrix or N-dim vector of probability
  density function evaluated at each observations with the natural
  parameters given by nat_param.
  If if_log_pdf = True, then return log of pdf 
  (which is desired for numerical stability).
  """ 
 def compute_pdf(self,model,nat_param,obs,suff_stat,log_partition,if_gail=False,if_log_pdf=False):
  if model=='norm':
   base_measure=1/np.sqrt(2.*np.pi)
   nat_times_suff=(nat_param[:,:,np.newaxis]*suff_stat).sum(1)
  elif model=='beta':
   param=self.nat_to_exp('beta',nat_param)
   pdf=np.empty(obs.shape)
   for a in xrange(self.num_agents):
    pdf[a,:]=stats.beta.pdf(obs[a,:],param[a,0],param[a,1])
   if if_log_pdf:
    return np.log(pdf)
   else:
    return pdf
  elif model=='expo':
   base_measure=1.
   nat_times_suff=(nat_param[:,:,np.newaxis]*suff_stat).sum(1)
  elif model=='mv_norm':
   if if_gail:
    base_measure=np.power(2*np.pi,-.5*self.dim_gail)
    nat_times_suff=(nat_param[:,:,np.newaxis]*suff_stat).sum((0,1))
    if if_log_pdf:
     return np.log(base_measure)+nat_times_suff-log_partition
    else:
     return base_measure*np.exp(nat_times_suff-log_partition)
   else:
    dd=obs.shape[2]
    base_measure=np.power(2*np.pi,-.5*dd)
    nat_times_suff=(nat_param[:,:,:,np.newaxis]*suff_stat).sum((1,2))
  else:
   raise ValueError('model type not supported')
  if if_log_pdf:
   return np.log(base_measure)+nat_times_suff-log_partition[:,np.newaxis]
  else:
   return base_measure*np.exp(nat_times_suff-log_partition[:,np.newaxis])
 def optimize_cross_entropy(self,model,coeff,default_nat_param,if_gail=True,if_gail_covar=True,if_verbose=True,center=0):
  solvers.options['show_progress']=if_verbose
  if model=='norm':
   coeff_flat=coeff.T.flatten()
   eta_one=Variable(self.num_agents)
   eta_two=Variable(self.num_agents)
   temp=0.
   for ii in xrange(self.num_agents):
    temp+=quad_over_lin(eta_one[ii],-eta_two[ii])
   objective =Minimize(-sum_entries(mul_elemwise(coeff[:,0],eta_one))-sum_entries(mul_elemwise(coeff[:,1],eta_two))-sum_entries(log(-eta_two))+.25*temp)
   constraints=[eta_two<=0,norm(eta_one-center[:,0],2)<=self.norm_mean_radius,norm(eta_two-center[:,1],2)<=self.norm_var_radius]
   problem=Problem(objective,constraints)
   problem.solve(solver=CVXOPT,verbose=if_verbose)
   eta_one=np.array(eta_one.value).flatten()
   eta_two=np.array(eta_two.value).flatten()
   return np.vstack((eta_one,eta_two)).T
  elif model=='beta':
   coeff_flat=coeff.flatten()
   trunc=np.minimum(default_nat_param.flatten(),self.beta_ub)
   trunc=np.maximum(trunc,self.beta_lb)
   default_cvxopt=matrix(trunc)
   def beta_oracle(x=None,z=None):
    if x is None:
     return 0,default_cvxopt
    if min(x)<self.beta_lb or max(x)>self.beta_ub:
     return None,None
    nat_param=np.array(x).reshape(default_nat_param.shape)
    (log_partition,grad_log_partition) =self.compute_log_partition('beta',nat_param)
    f=coeff_flat.dot(np.array(x))-log_partition.sum()
    Df=coeff_flat-grad_log_partition.flatten()
    Df=matrix(Df.reshape(1,Df.size))
    if z is None:
     return-f,-Df
    H=self.compute_hessian_log_partition('beta',nat_param)
    H=matrix(H)
    return-f,-Df,z[0,0]*H
   num_var=default_nat_param.size
   identity=np.eye(num_var)
   G=matrix(np.vstack((-identity,identity)))
   h=matrix(np.append(np.full(num_var,-self.beta_lb),np.full(num_var,self.beta_ub))) 
   opt_var=solvers.cp(beta_oracle,G=G,h=h)['x']
   return np.array(opt_var).reshape(default_nat_param.shape)
  elif model=='expo':
   coeff_flat=coeff.flatten()
   var=Variable(default_nat_param.size)
   objective=Minimize(-sum_entries(mul_elemwise(coeff_flat,var))-sum_entries(log(-var)))
   constraints=[-self.exp_lb>=var,var>=-self.exp_ub]
   problem=Problem(objective,constraints)
   problem.solve(solver=ECOS,verbose=if_verbose,max_iters=1000)
   return np.array(var.value).reshape(default_nat_param.shape)
  elif model=='mv_norm':
   if if_gail:
    if if_gail_covar:
     dim_gail=default_nat_param.shape[1]
     eta_one=Variable(dim_gail)
     eta_two_bar=Semidef(dim_gail)
     t=Variable(1)
     objective =Minimize(-sum_entries(mul_elemwise(coeff[0,:],eta_one))+sum_entries(mul_elemwise(coeff[1:,:],eta_two_bar))+.25*t-log_det(2*eta_two_bar))
     constraints =[bmat([[eta_two_bar,eta_one],[eta_one.T,t]])>>0]
     problem=Problem(objective,constraints)
     problem.solve(solver=SCS,verbose=if_verbose)
     eta_one=np.array(eta_one.value).flatten()
     eta_two=-np.array(eta_two_bar.value)
     return np.vstack((eta_one,eta_two))
    else:
     dim_gail=default_nat_param.shape[1]
     eta_one=Variable(dim_gail)
     objective =Minimize(-sum_entries(mul_elemwise(coeff[0,:],eta_one))-.25*quad_form(eta_one,default_nat_param[1:,:]))
     constraints =[abs(eta_one-center)<=self.mv_norm_mean_radius]
     problem=Problem(objective,constraints)
     problem.solve(solver=MOSEK,verbose=if_verbose)
     eta_one=np.array(eta_one.value).flatten()
     return np.vstack((eta_one,default_nat_param[1:,:]))
   else:
    ValueError('model type not supported')
  else:
   raise ValueError('model type not supported')
 def print_value(self,log_value,header):
  print header+': \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.exp(np.median(log_value))),str(np.exp(np.max(log_value))),str(np.exp(np.min(log_value))),str(np.exp(sp.logsumexp(log_value)-np.log(log_value.size))))
 def print_log_value(self,log_value,header):
  print header+': \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(log_value)),str(np.max(log_value)),str(np.min(log_value)),str(np.mean(log_value)))
 def compute_like_ratio(self,obs_v,obs_w,obs_x,obs_y,obs_gail,nat_param_v,nat_param_w,nat_param_x,nat_param_y,nat_param_gail,if_long=False,if_verbose=True):
  (log_partition_v,grad_log_partition_v) =self.compute_log_partition(self.model_v,nat_param_v)
  (log_partition_w,grad_log_partition_w) =self.compute_log_partition(self.model_w,nat_param_w)
  (log_partition_x,grad_log_partition_x) =self.compute_log_partition(self.model_x,nat_param_x)
  (log_partition_y,grad_log_partition_y) =self.compute_log_partition(self.model_y,nat_param_y)
  (log_partition_gail,grad_log_partition_gail) =self.compute_log_partition(self.model_gail,nat_param_gail,if_gail=True)
  suff_v=self.compute_suff(self.model_v,obs_v)
  suff_w=self.compute_suff(self.model_w,obs_w)
  suff_x=self.compute_suff(self.model_x,obs_x)
  suff_y=self.compute_suff(self.model_y,obs_y)
  suff_gail=self.compute_suff(self.model_gail,obs_gail,if_gail=True)
  pdf_v=self.compute_pdf(self.model_v,nat_param_v,obs_v,suff_v,log_partition_v)
  pdf_w=self.compute_pdf(self.model_w,nat_param_w,obs_w,suff_w,log_partition_w)
  pdf_x=self.compute_pdf(self.model_x,nat_param_x,obs_x,suff_x,log_partition_x)
  pdf_y=self.compute_pdf(self.model_y,nat_param_y,obs_y,suff_y,log_partition_y)
  pdf_v=pdf_v.prod(axis=0)
  pdf_w=pdf_w.prod(axis=0)
  pdf_x=pdf_x.prod(axis=0)
  pdf_y=pdf_y.prod(axis=0)
  pdf_gail=self.compute_pdf(self.model_gail,nat_param_gail,obs_gail,suff_gail,log_partition_gail,if_gail=True)
  base_pdf_v=self.compute_pdf(self.model_v,self.nat_base_v,obs_v,suff_v,self.base_log_partition_v)
  base_pdf_w=self.compute_pdf(self.model_w,self.nat_base_w,obs_w,suff_w,self.base_log_partition_w)
  base_pdf_x=self.compute_pdf(self.model_x,self.nat_base_x,obs_x,suff_x,self.base_log_partition_x)
  base_pdf_y=self.compute_pdf(self.model_y,self.nat_base_y,obs_y,suff_y,self.base_log_partition_y)
  base_pdf_v=base_pdf_v.prod(axis=0)
  base_pdf_w=base_pdf_w.prod(axis=0)
  base_pdf_x=base_pdf_x.prod(axis=0)
  base_pdf_y=base_pdf_y.prod(axis=0)
  base_pdf_gail=self.compute_pdf(self.model_gail,self.nat_base_gail,obs_gail,suff_gail,self.base_log_partition_gail,if_gail=True)
  like_ratio_v=np.exp(np.log(base_pdf_v)-np.log(pdf_v))
  like_ratio_w=np.exp(np.log(base_pdf_w)-np.log(pdf_w))
  like_ratio_x=np.exp(np.log(base_pdf_x)-np.log(pdf_x))
  like_ratio_y=np.exp(np.log(base_pdf_y)-np.log(pdf_y))
  like_ratio_gail=np.exp(np.log(base_pdf_gail)-np.log(pdf_gail))
  if if_verbose:
   print '-------------------- next pdf -----------------------'
   print 'pdf_v: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_v)),str(np.max(pdf_v)),str(np.min(pdf_v)),str(np.mean(pdf_v)))
   print 'pdf_w: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_w)),str(np.max(pdf_w)),str(np.min(pdf_w)),str(np.mean(pdf_w)))
   print 'pdf_x: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_x)),str(np.max(pdf_x)),str(np.min(pdf_x)),str(np.mean(pdf_x)))
   print 'pdf_y: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_y)),str(np.max(pdf_y)),str(np.min(pdf_y)),str(np.mean(pdf_y)))
   print 'pdf_gail: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_gail)),str(np.max(pdf_gail)),str(np.min(pdf_gail)),str(np.mean(pdf_gail)))
   print '-------------------------------------------' 
   print '-------------------- base pdf -----------------------'
   print 'base_pdf_v: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_v)),str(np.max(base_pdf_v)),str(np.min(base_pdf_v)),str(np.mean(base_pdf_v)))
   print 'base_pdf_w: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_w)),str(np.max(base_pdf_w)),str(np.min(base_pdf_w)),str(np.mean(base_pdf_w)))
   print 'base_pdf_x: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_x)),str(np.max(base_pdf_x)),str(np.min(base_pdf_x)),str(np.mean(base_pdf_x)))
   print 'base_pdf_y: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_y)),str(np.max(base_pdf_y)),str(np.min(base_pdf_y)),str(np.mean(base_pdf_y)))
   print 'base_pdf_gail: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_gail)),str(np.max(base_pdf_gail)),str(np.min(base_pdf_gail)),str(np.mean(base_pdf_gail)))
   print '-------------------------------------------'
   print '-------------------- like ratio -----------------------'
   print 'like_ratio_v: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_v)),str(np.max(like_ratio_v)),str(np.min(like_ratio_v)),str(np.mean(like_ratio_v)))
   print 'like_ratio_w: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_w)),str(np.max(like_ratio_w)),str(np.min(like_ratio_w)),str(np.mean(like_ratio_w)))
   print 'like_ratio_x: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_x)),str(np.max(like_ratio_x)),str(np.min(like_ratio_x)),str(np.mean(like_ratio_x)))
   print 'like_ratio_y: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_y)),str(np.max(like_ratio_y)),str(np.min(like_ratio_y)),str(np.mean(like_ratio_y)))
   print 'like_ratio_gail: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_gail)),str(np.max(like_ratio_gail)),str(np.min(like_ratio_gail)),str(np.mean(like_ratio_gail)))
   print '-------------------------------------------'
  if if_long:
   return like_ratio_v,like_ratio_w,like_ratio_x, like_ratio_y,like_ratio_gail, suff_v,suff_w,suff_x,suff_y,suff_gail, grad_log_partition_v,grad_log_partition_w,grad_log_partition_x, grad_log_partition_y,grad_log_partition_gail
  else:
   return like_ratio_v,like_ratio_w,like_ratio_x, like_ratio_y,like_ratio_gail
 def compute_log_like_ratio(self,obs_v,obs_w,obs_x,obs_y,obs_gail,nat_param_v,nat_param_w,nat_param_x,nat_param_y,nat_param_gail,if_long=False,if_verbose=True):
  (log_partition_v,grad_log_partition_v) =self.compute_log_partition(self.model_v,nat_param_v)
  (log_partition_w,grad_log_partition_w) =self.compute_log_partition(self.model_w,nat_param_w)
  (log_partition_x,grad_log_partition_x) =self.compute_log_partition(self.model_x,nat_param_x)
  (log_partition_y,grad_log_partition_y) =self.compute_log_partition(self.model_y,nat_param_y)
  (log_partition_gail,grad_log_partition_gail) =self.compute_log_partition(self.model_gail,nat_param_gail,if_gail=True)
  suff_v=self.compute_suff(self.model_v,obs_v)
  suff_w=self.compute_suff(self.model_w,obs_w)
  suff_x=self.compute_suff(self.model_x,obs_x)
  suff_y=self.compute_suff(self.model_y,obs_y)
  suff_gail=self.compute_suff(self.model_gail,obs_gail,if_gail=True)
  log_pdf_v=self.compute_pdf(self.model_v,nat_param_v,obs_v,suff_v,log_partition_v,if_log_pdf=True)
  log_pdf_w=self.compute_pdf(self.model_w,nat_param_w,obs_w,suff_w,log_partition_w,if_log_pdf=True)
  log_pdf_x=self.compute_pdf(self.model_x,nat_param_x,obs_x,suff_x,log_partition_x,if_log_pdf=True)
  log_pdf_y=self.compute_pdf(self.model_y,nat_param_y,obs_y,suff_y,log_partition_y,if_log_pdf=True)
  log_pdf_v=log_pdf_v.sum(axis=0)
  log_pdf_w=log_pdf_w.sum(axis=0)
  log_pdf_x=log_pdf_x.sum(axis=0)
  log_pdf_y=log_pdf_y.sum(axis=0)
  log_pdf_gail=self.compute_pdf(self.model_gail,nat_param_gail,obs_gail,suff_gail,log_partition_gail,if_gail=True,if_log_pdf=True)
  base_log_pdf_v=self.compute_pdf(self.model_v,self.nat_base_v,obs_v,suff_v,self.base_log_partition_v,if_log_pdf=True)
  base_log_pdf_w=self.compute_pdf(self.model_w,self.nat_base_w,obs_w,suff_w,self.base_log_partition_w,if_log_pdf=True)
  base_log_pdf_x=self.compute_pdf(self.model_x,self.nat_base_x,obs_x,suff_x,self.base_log_partition_x,if_log_pdf=True)
  base_log_pdf_y=self.compute_pdf(self.model_y,self.nat_base_y,obs_y,suff_y,self.base_log_partition_y,if_log_pdf=True)
  base_log_pdf_v=base_log_pdf_v.sum(axis=0)
  base_log_pdf_w=base_log_pdf_w.sum(axis=0)
  base_log_pdf_x=base_log_pdf_x.sum(axis=0)
  base_log_pdf_y=base_log_pdf_y.sum(axis=0)
  base_log_pdf_gail=self.compute_pdf(self.model_gail,self.nat_base_gail,obs_gail,suff_gail,self.base_log_partition_gail,if_gail=True,if_log_pdf=True)
  log_like_ratio_v=base_log_pdf_v-log_pdf_v
  log_like_ratio_w=base_log_pdf_w-log_pdf_w
  log_like_ratio_x=base_log_pdf_x-log_pdf_x
  log_like_ratio_y=base_log_pdf_y-log_pdf_y
  log_like_ratio_gail=base_log_pdf_gail-log_pdf_gail
  if if_verbose:
   print '====================== PDF ========================'
   print '-------------------- next pdf -----------------------'
   self.print_value(log_pdf_v,'pdf_v')
   self.print_value(log_pdf_w,'pdf_w')
   self.print_value(log_pdf_x,'pdf_x')
   self.print_value(log_pdf_y,'pdf_y')
   self.print_value(log_pdf_gail,'pdf_gail')
   print '-------------------------------------------'
   print '-------------------- base pdf -----------------------'
   self.print_value(base_log_pdf_v,'base_pdf_v')
   self.print_value(base_log_pdf_w,'base_pdf_w')
   self.print_value(base_log_pdf_x,'base_pdf_x')
   self.print_value(base_log_pdf_y,'base_pdf_y')
   self.print_value(base_log_pdf_gail,'base_pdf_gail')
   print '-------------------------------------------'
   print '-------------------- like ratio -----------------------'
   self.print_value(log_like_ratio_v,'like_ratio_v')
   self.print_value(log_like_ratio_w,'like_ratio_w')
   self.print_value(log_like_ratio_x,'like_ratio_x')
   self.print_value(log_like_ratio_y,'like_ratio_y')
   self.print_value(log_like_ratio_gail,'like_ratio_gail') 
   print '-------------------------------------------'
   print '====================== LOG PDF ========================'
   print '-------------------- next log pdf -----------------------'
   self.print_log_value(log_pdf_v,'log_pdf_v')
   self.print_log_value(log_pdf_w,'log_pdf_w')
   self.print_log_value(log_pdf_x,'log_pdf_x')
   self.print_log_value(log_pdf_y,'log_pdf_y')
   self.print_log_value(log_pdf_gail,'log_pdf_gail')
   print '-------------------------------------------'
   print '-------------------- base log pdf -----------------------'
   self.print_log_value(base_log_pdf_v,'base_log_pdf_v')
   self.print_log_value(base_log_pdf_w,'base_log_pdf_w')
   self.print_log_value(base_log_pdf_x,'base_log_pdf_x')
   self.print_log_value(base_log_pdf_y,'base_log_pdf_y')
   self.print_log_value(base_log_pdf_gail,'base_log_pdf_gail')
   print '-------------------------------------------'
   print '-------------------- log like ratio -----------------------'
   self.print_log_value(log_like_ratio_v,'log_like_ratio_v')
   self.print_log_value(log_like_ratio_w,'log_like_ratio_w')
   self.print_log_value(log_like_ratio_x,'log_like_ratio_x')
   self.print_log_value(log_like_ratio_y,'log_like_ratio_y')
   self.print_log_value(log_like_ratio_gail,'log_like_ratio_gail') 
   print '-------------------------------------------'
  if if_long:
   return log_like_ratio_v,log_like_ratio_w,log_like_ratio_x, log_like_ratio_y,log_like_ratio_gail, suff_v,suff_w,suff_x,suff_y,suff_gail, grad_log_partition_v,grad_log_partition_w,grad_log_partition_x, grad_log_partition_y,grad_log_partition_gail
  else:
   return log_like_ratio_v,log_like_ratio_w,log_like_ratio_x, log_like_ratio_y,log_like_ratio_gail
 def compute_pair_pdfs(self,obs_v,obs_w,obs_x,obs_y,obs_gail,nat_param_v,nat_param_w,nat_param_x,nat_param_y,nat_param_gail,if_verbose=True,mixtures=np.array([0.]),if_include_gail=True):
  (log_partition_v,grad_log_partition_v) =self.compute_log_partition(self.model_v,nat_param_v)
  (log_partition_w,grad_log_partition_w) =self.compute_log_partition(self.model_w,nat_param_w)
  (log_partition_x,grad_log_partition_x) =self.compute_log_partition(self.model_x,nat_param_x)
  (log_partition_y,grad_log_partition_y) =self.compute_log_partition(self.model_y,nat_param_y)
  (log_partition_gail,grad_log_partition_gail) =self.compute_log_partition(self.model_gail,nat_param_gail,if_gail=True)
  suff_v=self.compute_suff(self.model_v,obs_v)
  suff_w=self.compute_suff(self.model_w,obs_w)
  suff_x=self.compute_suff(self.model_x,obs_x)
  suff_y=self.compute_suff(self.model_y,obs_y)
  suff_gail=self.compute_suff(self.model_gail,obs_gail,if_gail=True)
  pdf_v=self.compute_pdf(self.model_v,nat_param_v,obs_v,suff_v,log_partition_v)
  pdf_w=self.compute_pdf(self.model_w,nat_param_w,obs_w,suff_w,log_partition_w)
  pdf_x=self.compute_pdf(self.model_x,nat_param_x,obs_x,suff_x,log_partition_x)
  pdf_y=self.compute_pdf(self.model_y,nat_param_y,obs_y,suff_y,log_partition_y)
  pdf_v=pdf_v.prod(axis=0)
  pdf_w=pdf_w.prod(axis=0)
  pdf_x=pdf_x.prod(axis=0)
  pdf_y=pdf_y.prod(axis=0)
  pdf_gail=self.compute_pdf(self.model_gail,nat_param_gail,obs_gail,suff_gail,log_partition_gail,if_gail=True)
  if if_include_gail:
   pdf=pdf_v*pdf_w*pdf_x*pdf_y*pdf_gail
  else:
   pdf=pdf_v*pdf_w*pdf_x*pdf_y
  base_pdf_v=self.compute_pdf(self.model_v,self.nat_base_v,obs_v,suff_v,self.base_log_partition_v)
  base_pdf_w=self.compute_pdf(self.model_w,self.nat_base_w,obs_w,suff_w,self.base_log_partition_w)
  base_pdf_x=self.compute_pdf(self.model_x,self.nat_base_x,obs_x,suff_x,self.base_log_partition_x)
  base_pdf_y=self.compute_pdf(self.model_y,self.nat_base_y,obs_y,suff_y,self.base_log_partition_y)
  base_pdf_v=base_pdf_v.prod(axis=0)
  base_pdf_w=base_pdf_w.prod(axis=0)
  base_pdf_x=base_pdf_x.prod(axis=0)
  base_pdf_y=base_pdf_y.prod(axis=0)
  base_pdf_gail=self.compute_pdf(self.model_gail,self.nat_base_gail,obs_gail,suff_gail,self.base_log_partition_gail,if_gail=True)
  if if_include_gail:
   base_pdf=base_pdf_v*base_pdf_w*base_pdf_x* base_pdf_y*base_pdf_gail
  else:
   base_pdf=base_pdf_v*base_pdf_w*base_pdf_x*base_pdf_y
  if if_verbose:
   print '-------------------- next pdf -----------------------'
   print 'pdf_v: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_v)),str(np.max(pdf_v)),str(np.min(pdf_v)),str(np.mean(pdf_v)))
   print 'pdf_w: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_w)),str(np.max(pdf_w)),str(np.min(pdf_w)),str(np.mean(pdf_w)))
   print 'pdf_x: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_x)),str(np.max(pdf_x)),str(np.min(pdf_x)),str(np.mean(pdf_x)))
   print 'pdf_y: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_y)),str(np.max(pdf_y)),str(np.min(pdf_y)),str(np.mean(pdf_y)))
   if if_include_gail:
    print 'pdf_gail: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(pdf_gail)),str(np.max(pdf_gail)),str(np.min(pdf_gail)),str(np.mean(pdf_gail)))
   print '-------------------------------------------' 
   print '-------------------- base pdf -----------------------'
   print 'base_pdf_v: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_v)),str(np.max(base_pdf_v)),str(np.min(base_pdf_v)),str(np.mean(base_pdf_v)))
   print 'base_pdf_w: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_w)),str(np.max(base_pdf_w)),str(np.min(base_pdf_w)),str(np.mean(base_pdf_w)))
   print 'base_pdf_x: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_x)),str(np.max(base_pdf_x)),str(np.min(base_pdf_x)),str(np.mean(base_pdf_x)))
   print 'base_pdf_y: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_y)),str(np.max(base_pdf_y)),str(np.min(base_pdf_y)),str(np.mean(base_pdf_y)))
   if if_include_gail:
    print 'base_pdf_gail: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(base_pdf_gail)),str(np.max(base_pdf_gail)),str(np.min(base_pdf_gail)),str(np.mean(base_pdf_gail)))
   print '-------------------------------------------'
   like_ratio_v=np.exp(np.log(base_pdf_v)-np.log(pdf_v))
   like_ratio_w=np.exp(np.log(base_pdf_w)-np.log(pdf_w))
   like_ratio_x=np.exp(np.log(base_pdf_x)-np.log(pdf_x))
   like_ratio_y=np.exp(np.log(base_pdf_y)-np.log(pdf_y))
   like_ratio_gail=np.exp(np.log(base_pdf_gail)-np.log(pdf_gail))
   print '-------------------- like ratio -----------------------'
   print 'like_ratio_v: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_v)),str(np.max(like_ratio_v)),str(np.min(like_ratio_v)),str(np.mean(like_ratio_v)))
   print 'like_ratio_w: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_w)),str(np.max(like_ratio_w)),str(np.min(like_ratio_w)),str(np.mean(like_ratio_w)))
   print 'like_ratio_x: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_x)),str(np.max(like_ratio_x)),str(np.min(like_ratio_x)),str(np.mean(like_ratio_x)))
   print 'like_ratio_y: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_y)),str(np.max(like_ratio_y)),str(np.min(like_ratio_y)),str(np.mean(like_ratio_y)))
   if if_include_gail:
    print 'like_ratio_gail: \t median = %6s \t max = %6s \t min = %6s \t mean= %6s' %(str(np.median(like_ratio_gail)),str(np.max(like_ratio_gail)),str(np.min(like_ratio_gail)),str(np.mean(like_ratio_gail)))
   print '-------------------------------------------'
  return pdf,base_pdf
 def compute_pair_log_pdfs(self,obs_v,obs_w,obs_x,obs_y,obs_gail,nat_param_v,nat_param_w,nat_param_x,nat_param_y,nat_param_gail,if_verbose=True,mixtures=np.array([0.]),if_include_gail=True):
  (log_partition_v,grad_log_partition_v) =self.compute_log_partition(self.model_v,nat_param_v)
  (log_partition_w,grad_log_partition_w) =self.compute_log_partition(self.model_w,nat_param_w)
  (log_partition_x,grad_log_partition_x) =self.compute_log_partition(self.model_x,nat_param_x)
  (log_partition_y,grad_log_partition_y) =self.compute_log_partition(self.model_y,nat_param_y)
  (log_partition_gail,grad_log_partition_gail) =self.compute_log_partition(self.model_gail,nat_param_gail,if_gail=True)
  suff_v=self.compute_suff(self.model_v,obs_v)
  suff_w=self.compute_suff(self.model_w,obs_w)
  suff_x=self.compute_suff(self.model_x,obs_x)
  suff_y=self.compute_suff(self.model_y,obs_y)
  suff_gail=self.compute_suff(self.model_gail,obs_gail,if_gail=True)
  log_pdf_v=self.compute_pdf(self.model_v,nat_param_v,obs_v,suff_v,log_partition_v,if_log_pdf=True)
  log_pdf_w=self.compute_pdf(self.model_w,nat_param_w,obs_w,suff_w,log_partition_w,if_log_pdf=True)
  log_pdf_x=self.compute_pdf(self.model_x,nat_param_x,obs_x,suff_x,log_partition_x,if_log_pdf=True)
  log_pdf_y=self.compute_pdf(self.model_y,nat_param_y,obs_y,suff_y,log_partition_y,if_log_pdf=True)
  log_pdf_v=log_pdf_v.sum(axis=0)
  log_pdf_w=log_pdf_w.sum(axis=0)
  log_pdf_x=log_pdf_x.sum(axis=0)
  log_pdf_y=log_pdf_y.sum(axis=0)
  log_pdf_gail=self.compute_pdf(self.model_gail,nat_param_gail,obs_gail,suff_gail,log_partition_gail,if_gail=True,if_log_pdf=True)
  if if_include_gail:
   log_pdf=log_pdf_v+log_pdf_w+log_pdf_x+log_pdf_y+log_pdf_gail
  else:
   log_pdf=log_pdf_v+log_pdf_w+log_pdf_x+log_pdf_y
  base_log_pdf_v=self.compute_pdf(self.model_v,self.nat_base_v,obs_v,suff_v,self.base_log_partition_v,if_log_pdf=True)
  base_log_pdf_w=self.compute_pdf(self.model_w,self.nat_base_w,obs_w,suff_w,self.base_log_partition_w,if_log_pdf=True)
  base_log_pdf_x=self.compute_pdf(self.model_x,self.nat_base_x,obs_x,suff_x,self.base_log_partition_x,if_log_pdf=True)
  base_log_pdf_y=self.compute_pdf(self.model_y,self.nat_base_y,obs_y,suff_y,self.base_log_partition_y,if_log_pdf=True)
  base_log_pdf_v=base_log_pdf_v.sum(axis=0)
  base_log_pdf_w=base_log_pdf_w.sum(axis=0)
  base_log_pdf_x=base_log_pdf_x.sum(axis=0)
  base_log_pdf_y=base_log_pdf_y.sum(axis=0)
  base_log_pdf_gail=self.compute_pdf(self.model_gail,self.nat_base_gail,obs_gail,suff_gail,self.base_log_partition_gail,if_gail=True,if_log_pdf=True)
  if if_include_gail:
   base_log_pdf=base_log_pdf_v+base_log_pdf_w+base_log_pdf_x+ base_log_pdf_y+base_log_pdf_gail
  else:
   base_log_pdf=base_log_pdf_v+base_log_pdf_w+base_log_pdf_x+base_log_pdf_y
  if if_verbose:
   log_like_ratio_v=base_log_pdf_v-log_pdf_v
   log_like_ratio_w=base_log_pdf_w-log_pdf_w
   log_like_ratio_x=base_log_pdf_x-log_pdf_x
   log_like_ratio_y=base_log_pdf_y-log_pdf_y
   log_like_ratio_gail=base_log_pdf_gail-log_pdf_gail
   print '====================== PDF ========================'
   print '-------------------- next pdf -----------------------'
   self.print_value(log_pdf_v,'pdf_v')
   self.print_value(log_pdf_w,'pdf_w')
   self.print_value(log_pdf_x,'pdf_x')
   self.print_value(log_pdf_y,'pdf_y')
   self.print_value(log_pdf_gail,'pdf_gail')
   print '-------------------------------------------'
   print '-------------------- base pdf -----------------------'
   self.print_value(base_log_pdf_v,'base_pdf_v')
   self.print_value(base_log_pdf_w,'base_pdf_w')
   self.print_value(base_log_pdf_x,'base_pdf_x')
   self.print_value(base_log_pdf_y,'base_pdf_y')
   self.print_value(base_log_pdf_gail,'base_pdf_gail')
   print '-------------------------------------------'
   print '-------------------- like ratio -----------------------'
   self.print_value(log_like_ratio_v,'like_ratio_v')
   self.print_value(log_like_ratio_w,'like_ratio_w')
   self.print_value(log_like_ratio_x,'like_ratio_x')
   self.print_value(log_like_ratio_y,'like_ratio_y')
   self.print_value(log_like_ratio_gail,'like_ratio_gail') 
   print '-------------------------------------------'
   print '====================== LOG PDF ========================'
   print '-------------------- next log pdf -----------------------'
   self.print_log_value(log_pdf_v,'log_pdf_v')
   self.print_log_value(log_pdf_w,'log_pdf_w')
   self.print_log_value(log_pdf_x,'log_pdf_x')
   self.print_log_value(log_pdf_y,'log_pdf_y')
   self.print_log_value(log_pdf_gail,'log_pdf_gail')
   print '-------------------------------------------'
   print '-------------------- base log pdf -----------------------'
   self.print_log_value(base_log_pdf_v,'base_log_pdf_v')
   self.print_log_value(base_log_pdf_w,'base_log_pdf_w')
   self.print_log_value(base_log_pdf_x,'base_log_pdf_x')
   self.print_log_value(base_log_pdf_y,'base_log_pdf_y')
   self.print_log_value(base_log_pdf_gail,'base_log_pdf_gail')
   print '-------------------------------------------'
   print '-------------------- log like ratio -----------------------'
   self.print_log_value(log_like_ratio_v,'log_like_ratio_v')
   self.print_log_value(log_like_ratio_w,'log_like_ratio_w')
   self.print_log_value(log_like_ratio_x,'log_like_ratio_x')
   self.print_log_value(log_like_ratio_y,'log_like_ratio_y')
   self.print_log_value(log_like_ratio_gail,'log_like_ratio_gail') 
   print '-------------------------------------------'
  return log_pdf,base_log_pdf
def test_helpers():
 np.random.seed(1234)
 num_agents=5
 base_v=np.abs(np.random.randn(num_agents,1))
 base_w=10*np.random.rand(num_agents,2)
 base_x=np.column_stack((np.random.randn(num_agents),10*np.random.rand(num_agents)))
 base_y=10*np.random.rand(num_agents,2)
 dd=400
 mu=5.*np.random.randn(dd)
 sigma=np.diag(np.random.rand(dd))
 base_gail=np.vstack((mu,sigma))
 bd_y=5
 bd_w=np.pi
 ic=init_conds(base_v,base_w,base_x,base_y,base_gail)
 print('exp_to_nat')
 print('norm= ')
 print(ic.exp_to_nat('norm',base_x))
 print('beta= ')
 print(ic.exp_to_nat('beta',base_w))
 print('expo= ')
 print(ic.exp_to_nat('expo',base_v))
 print('mv_norm= ')
 print(ic.exp_to_nat('mv_norm',base_gail,if_gail=True))
 print('====================================================')
 print('==================== Testing Match =================')
 print('exp_to_nat')
 print('norm= ')
 nat_param=ic.exp_to_nat('norm',base_x)
 print(np.linalg.norm(base_x-ic.nat_to_exp('norm',nat_param)))
 print('beta= ')
 nat_param=ic.exp_to_nat('beta',base_w)
 print(np.linalg.norm(base_w-ic.nat_to_exp('beta',nat_param)))
 print('expo= ')
 nat_param=ic.exp_to_nat('expo',base_v)
 print(np.linalg.norm(base_v-ic.nat_to_exp('expo',nat_param)))
 print('mv_norm= ')
 nat_param=ic.exp_to_nat('mv_norm',base_gail,if_gail=True)
 print(np.linalg.norm(base_gail-ic.nat_to_exp('mv_norm',nat_param,if_gail=True)))
 print('====================================================')
 print('==================== Testing PDF Evaluation =================')
 sample_size=10
 model='norm'
 print(model+'= ')
 param=base_x
 obs=np.empty((num_agents,sample_size))
 pdf_true=np.empty((num_agents,sample_size))
 np.random.seed(12345)
 for a in xrange(num_agents):
  obs[a,:]=np.random.normal(param[a,0],np.sqrt(param[a,1]),size=sample_size)
  pdf_true[a,:]=stats.norm.pdf(obs[a,:],base_x[a,0],np.sqrt(param[a,1]))
 nat_param=ic.exp_to_nat(model,param)
 np.random.seed(12345)
 obs_ic=ic.sample_obs(model,nat_param,sample_size)
 print('----- test samples ------')
 print np.linalg.norm(obs-obs_ic)
 suff_stat=ic.compute_suff(model,obs)
 (log_partition,grad_log_partition) =ic.compute_log_partition(model,nat_param)
 pdf_ic=ic.compute_pdf(model,nat_param,obs,suff_stat,log_partition)
 print('----- test pdf ------')
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_true))
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_ic))
 print('-----------------------------------------------')
 model='beta'
 print(model+'= ')
 param=base_w
 obs=np.empty((num_agents,sample_size))
 pdf_true=np.empty((num_agents,sample_size))
 np.random.seed(12345)
 for a in xrange(num_agents):
  obs[a,:]=np.random.beta(param[a,0],param[a,1],size=sample_size)
  pdf_true[a,:]=stats.beta.pdf(obs[a,:],param[a,0],param[a,1])
 nat_param=ic.exp_to_nat(model,param)
 np.random.seed(12345)
 obs_ic=ic.sample_obs(model,nat_param,sample_size)
 print('----- test samples ------')
 print np.linalg.norm(obs-obs_ic)
 suff_stat=ic.compute_suff(model,obs)
 (log_partition,grad_log_partition) =ic.compute_log_partition(model,nat_param)
 pdf_ic=ic.compute_pdf(model,nat_param,obs,suff_stat,log_partition)
 print('----- test pdf ------')
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_true))
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_ic))
 print('-----------------------------------------------')
 model='expo'
 print(model+'= ')
 param=base_v
 obs=np.empty((num_agents,sample_size))
 pdf_true=np.empty((num_agents,sample_size))
 np.random.seed(12345)
 for a in xrange(num_agents):
  obs[a,:]=np.random.exponential(1./param[a,0],size=sample_size)
  pdf_true[a,:]=stats.expon.pdf(obs[a,:],0.,1./param[a,0])
 nat_param=ic.exp_to_nat(model,param)
 np.random.seed(12345)
 obs_ic=ic.sample_obs(model,nat_param,sample_size)
 print('----- test samples ------')
 print np.linalg.norm(obs-obs_ic)
 suff_stat=ic.compute_suff(model,obs)
 (log_partition,grad_log_partition) =ic.compute_log_partition(model,nat_param) 
 pdf_ic=ic.compute_pdf(model,nat_param,obs,suff_stat,log_partition)
 print('----- test pdf ------')
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_true))
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_ic))
 print('-----------------------------------------------')
 model='mv_norm'
 print(model+'= ')
 param=base_gail
 np.random.seed(12345)
 obs=np.random.multivariate_normal(param[0,:],param[1:,:],size=sample_size)
 pdf_true =stats.multivariate_normal.pdf(obs,param[0,:],param[1:,:])
 nat_param=ic.exp_to_nat(model,param,if_gail=True)
 np.random.seed(12345)
 obs_ic=ic.sample_obs(model,nat_param,sample_size,if_gail=True)
 print('----- test samples ------')
 print np.linalg.norm(obs-obs_ic)
 suff_stat=ic.compute_suff(model,obs,if_gail=True)
 (log_partition,grad_log_partition) =ic.compute_log_partition(model,nat_param,if_gail=True)
 pdf_ic=ic.compute_pdf(model,nat_param,obs,suff_stat,log_partition,if_gail=True)
 print('----- test pdf ------')
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_true))
 print(np.linalg.norm(pdf_true-pdf_ic)/np.linalg.norm(pdf_ic))
 return ic,base_v,base_w,base_x,base_y,base_gail,bd_y,bd_w
# Created by pyminifier (https://github.com/liftoff/pyminifier)

