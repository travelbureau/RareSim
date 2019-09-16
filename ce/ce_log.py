import numpy as np,numpy.matlib
from scipy import special as sp,stats
from scipy.special import logsumexp
import math
import init_conds
import argparse
import copy
import zmq
from joblib import Parallel,delayed
from multiprocessing import Process
import time
from cross_entropy_test import test_objective,test_objective_simple,test_gradient,test_ce,test_ce_simple
import h5py,os
import sys
sys.path.append('../Utils')
from Utils import send_array,recv_array,cd
tol=1e-8
def cross_entropy(ic,rho,level,num_iter,save_iter,sample_sizes,step_sizes,compute_objective,source_worker_socket,source_sink_req_socket,source_worker_direct_sockets,sockets=0,if_gail_covar=False,if_verbose=False,if_early_stop=False,if_update_gail=True):
 nat_param_v=ic.nat_base_v
 nat_param_w=ic.nat_base_w
 nat_param_x=ic.nat_base_x
 nat_param_y=ic.nat_base_y
 nat_param_gail=ic.nat_base_gail
 all_nat_param_v=np.empty((save_iter.size+1,)+nat_param_v.shape)
 all_nat_param_w=np.empty((save_iter.size+1,)+nat_param_w.shape)
 all_nat_param_x=np.empty((save_iter.size+1,)+nat_param_x.shape)
 all_nat_param_y=np.empty((save_iter.size+1,)+nat_param_y.shape)
 all_nat_param_gail=np.empty((save_iter.size+1,)+nat_param_gail.shape)
 print " ------------------------------------------------------------ "
 print " --------------- Starting Cross Entropy Loop----------------- "
 current_best=-1e5
 save_counter=0
 for iter in xrange(0,num_iter):
  print " ================= Iteration = "+str(iter) +" =================== "
  obs_v=ic.sample_obs(ic.model_v,nat_param_v,sample_sizes[iter])
  obs_w=ic.sample_obs(ic.model_w,nat_param_w,sample_sizes[iter])
  obs_x=ic.sample_obs(ic.model_x,nat_param_x,sample_sizes[iter])
  obs_y=ic.sample_obs(ic.model_y,nat_param_y,sample_sizes[iter])
  obs_gail=ic.sample_obs(ic.model_gail,nat_param_gail,sample_sizes[iter],if_gail=True)
  obj=compute_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,ic,iter,source_worker_socket,source_sink_req_socket,if_update_gail=if_update_gail)
  thresh_ind=np.int(np.floor(rho*sample_sizes[iter]))
  inds=np.argpartition(obj,thresh_ind)
  if obj[inds[thresh_ind]]>level:
   if if_early_stop:
    for ss in xrange(save_counter,save_iter.size):
     all_nat_param_v[ss,:]=nat_param_v
     all_nat_param_w[ss,:]=nat_param_w
     all_nat_param_x[ss,:]=nat_param_x
     all_nat_param_y[ss,:]=nat_param_y
     all_nat_param_gail[ss,:]=nat_param_gail
     print '=========== Cross Entropy Ended Early ==========='
    break
   else:
    print '========== Level Exceeded ==========='
    good_inds=obj>level
  else:
   good_inds=inds[thresh_ind:]
  print 'obj mean'
  print np.mean(obj)
  print 'obj median'
  print np.median(obj)
  print 'obj good mean'
  obj_good_mean=np.mean(obj[good_inds])
  if current_best<obj_good_mean:
   current_best=obj_good_mean
   all_nat_param_v[-1,:]=nat_param_v
   all_nat_param_w[-1,:]=nat_param_w
   all_nat_param_x[-1,:]=nat_param_x
   all_nat_param_y[-1,:]=nat_param_y
   all_nat_param_gail[-1,:]=nat_param_gail
   print 'best updated at iter = '+str(iter)
  print obj_good_mean
  log_like_ratio_v,log_like_ratio_w,log_like_ratio_x, log_like_ratio_y,log_like_ratio_gail, suff_v,suff_w,suff_x,suff_y,suff_gail, grad_log_partition_v,grad_log_partition_w,grad_log_partition_x, grad_log_partition_y,grad_log_partition_gail =ic.compute_log_like_ratio(obs_v,obs_w,obs_x,obs_y,obs_gail,nat_param_v,nat_param_w,nat_param_x,nat_param_y,nat_param_gail,True)
  log_like_ratio_v=log_like_ratio_v[good_inds]
  log_like_ratio_w=log_like_ratio_w[good_inds]
  log_like_ratio_x=log_like_ratio_x[good_inds]
  log_like_ratio_y=log_like_ratio_y[good_inds]
  log_like_ratio_gail=log_like_ratio_gail[good_inds]
  effective_sample_size=(sample_sizes[iter]*(1.-rho))
  trunc_suff_v=suff_v[:,:,good_inds]
  trunc_suff_w=suff_w[:,:,good_inds]
  trunc_suff_x=suff_x[:,:,good_inds]
  trunc_suff_y=suff_y[:,:,good_inds]
  trunc_suff_gail=suff_gail[:,:,good_inds]
  imp_surrogate_v =compute_imp_surrogate(ic,trunc_suff_v,log_like_ratio_v,effective_sample_size)
  imp_surrogate_w =compute_imp_surrogate(ic,trunc_suff_w,log_like_ratio_w,effective_sample_size)
  imp_surrogate_x =compute_imp_surrogate(ic,trunc_suff_x,log_like_ratio_x,effective_sample_size)
  imp_surrogate_y =compute_imp_surrogate(ic,trunc_suff_y,log_like_ratio_y,effective_sample_size)
  imp_surrogate_gail =compute_imp_surrogate(ic,trunc_suff_gail,log_like_ratio_gail,effective_sample_size)
  coeff_v=step_sizes[iter]*imp_surrogate_v +(1-step_sizes[iter])*grad_log_partition_v
  coeff_w=step_sizes[iter]*imp_surrogate_w +(1-step_sizes[iter])*grad_log_partition_w
  coeff_y=step_sizes[iter]*imp_surrogate_y +(1-step_sizes[iter])*grad_log_partition_y
  coeff_x=step_sizes[iter]*imp_surrogate_x +(1-step_sizes[iter])*grad_log_partition_x 
  coeff_gail=step_sizes[iter]*imp_surrogate_gail +(1-step_sizes[iter])*grad_log_partition_gail
  nat_param_v=ic.optimize_cross_entropy(ic.model_v,coeff_v,ic.nat_base_v,if_verbose=if_verbose,center=ic.nat_base_v)
  nat_param_w=ic.optimize_cross_entropy(ic.model_w,coeff_w,ic.nat_base_w,if_verbose=if_verbose,center=ic.nat_base_w)
  nat_param_y=ic.optimize_cross_entropy(ic.model_y,coeff_y,ic.nat_base_y,if_verbose=if_verbose,center=ic.nat_base_y)
  nat_param_x=ic.optimize_cross_entropy(ic.model_x,coeff_x,ic.nat_base_x,if_verbose=if_verbose,center=ic.nat_base_x)
  if if_update_gail:
   num_try=0
   while num_try<2:
    num_try+=1
    try:
     nat_param_gail=ic.optimize_cross_entropy(ic.model_gail,coeff_gail,ic.nat_base_gail,if_gail=True,if_gail_covar=if_gail_covar,if_verbose=if_verbose,center=ic.nat_base_gail[0,:])
     break
    except:
     print("Mosek Barfed")
  print 'nat_param_v'
  print nat_param_v
  print 'nat_param_w'
  print nat_param_w
  print 'nat_param_x'
  print nat_param_x
  print 'nat_param_y'
  print nat_param_y
  if iter in save_iter:
   all_nat_param_v[save_counter,:]=nat_param_v
   all_nat_param_w[save_counter,:]=nat_param_w
   all_nat_param_x[save_counter,:]=nat_param_x
   all_nat_param_y[save_counter,:]=nat_param_y
   all_nat_param_gail[save_counter,:]=nat_param_gail
   save_counter+=1
 print " ----------------------------------------------------------- "
 print " --------------- Ending Cross Entropy Loop ----------------- "
 return all_nat_param_v,all_nat_param_w, all_nat_param_x,all_nat_param_y,all_nat_param_gail
def compute_imp_surrogate(ic,trunc_suff,log_like_ratio,effective_sample_size):
 imp_surrogate=np.empty((trunc_suff.shape[0],trunc_suff.shape[1]))
 for a in xrange(ic.num_agents):
  for b in xrange(trunc_suff.shape[1]):
   pos_inds=trunc_suff[a,b,:]>0
   neg_inds=trunc_suff[a,b,:]<0
   if np.any(pos_inds):
    pos_part =logsumexp(log_like_ratio[pos_inds]+np.log(trunc_suff[a,b,pos_inds])) -np.log(effective_sample_size)
    pos_part=np.exp(pos_part)
   else:
    pos_part=0
   if np.any(neg_inds):
    neg_part =logsumexp(log_like_ratio[neg_inds]+np.log(-trunc_suff[a,b,neg_inds])) -np.log(effective_sample_size)
    neg_part=np.exp(neg_part)
   else:
    neg_part=0
   imp_surrogate[a,b]=pos_part-neg_part
 return imp_surrogate
"""cross_entropy_wass_tilt(ic, rho, num_iter, sample_sizes, step_sizes,                            lams, gd_step_sizes, gd_num_iters)
Runs the cross-entropy method with Wasserstein tilting with the
following inputs:
ic is the init_conds class containing the base parameters
rho is a number between (0, 1) specifying the cut-off quantile 
at each iteration
num_iter is a positive integer specifying number of cross entropy updates
sample_sizes is a num_iter dimensional vector of positive integers 
that represents the number of samples at each iteration
step_sizes is a num_iter dimensional vector with values in [0, 1] that 
determines the amount of weight on the current surrogate candidate q_k(x) 
\propto \indic{\obj(x) \ge \gamma_k} p_0(x)
lams is a dictionary containing lambdas for v, w, x, y, gail
gd_step_sizes is a dictionary containing stepsizes for v, w, x, y, gail
gd_num_iters is a num_iter dimensional vector
"""
def cross_entropy_wass_tilt(ic,rho,level,num_iter,save_iter,sample_sizes,step_sizes,lams,gd_step_sizes,gd_num_iters,compute_objective,compute_gradient,compute_hessian,sockets=0,if_gail_covar=False,if_verbose=False,if_early_stop=False,if_update_gail=True):
 nat_param_v=ic.nat_base_v
 nat_param_w=ic.nat_base_w
 nat_param_x=ic.nat_base_x
 nat_param_y=ic.nat_base_y
 nat_param_gail=ic.nat_base_gail
 all_nat_param_v=np.empty((save_iter.size,)+nat_param_v.shape)
 all_nat_param_w=np.empty((save_iter.size,)+nat_param_w.shape)
 all_nat_param_x=np.empty((save_iter.size,)+nat_param_x.shape)
 all_nat_param_y=np.empty((save_iter.size,)+nat_param_y.shape)
 all_nat_param_gail=np.empty((save_iter.size,)+nat_param_gail.shape)
 print " ------------------------------------------------------------ "
 print " --------------- Starting Cross Entropy Loop ---------------- "
 save_counter=0
 for iter in xrange(0,num_iter):
  print " ================= Iteration = "+str(iter) +" =================== "
  orig_obs_v=ic.sample_obs(ic.model_v,nat_param_v,sample_sizes[iter])
  orig_obs_w=ic.sample_obs(ic.model_w,nat_param_w,sample_sizes[iter])
  orig_obs_x=ic.sample_obs(ic.model_x,nat_param_x,sample_sizes[iter])
  orig_obs_y=ic.sample_obs(ic.model_y,nat_param_y,sample_sizes[iter])
  orig_obs_gail=ic.sample_obs(ic.model_gail,nat_param_gail,sample_sizes[iter],if_gail=True)
  obs_v,obs_w,obs_x,obs_y,obs_gail =compute_monge_map(ic,lams,gd_step_sizes,gd_num_iters[iter],orig_obs_v,orig_obs_w,orig_obs_x,orig_obs_y,orig_obs_gail,compute_gradient,sockets)
  obj=compute_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,ic,iter,source_worker_socket,source_sink_req_socket,if_update_gail=if_update_gail)
  thresh_ind=np.int(np.floor(rho*sample_sizes[iter]))
  inds=np.argpartition(obj,thresh_ind)
  if obj[inds[thresh_ind]]>level:
   if if_early_stop:
    for ss in xrange(save_counter,save_iter.size):
     all_nat_param_v[ss,:]=nat_param_v
     all_nat_param_w[ss,:]=nat_param_w
     all_nat_param_x[ss,:]=nat_param_x
     all_nat_param_y[ss,:]=nat_param_y
     all_nat_param_gail[ss,:]=nat_param_gail
     print '=========== Cross Entropy Ended Early ==========='
    break
   else:
    print '========== Level Exceeded ==========='
    good_inds=obj>level
  else:
   good_inds=inds[thresh_ind:]
  (log_partition_v,grad_log_partition_v) =ic.compute_log_partition(ic.model_v,nat_param_v)
  (log_partition_w,grad_log_partition_w) =ic.compute_log_partition(ic.model_w,nat_param_w)
  (log_partition_x,grad_log_partition_x) =ic.compute_log_partition(ic.model_x,nat_param_x)
  (log_partition_y,grad_log_partition_y) =ic.compute_log_partition(ic.model_y,nat_param_y)
  (log_partition_gail,grad_log_partition_gail) =ic.compute_log_partition(ic.model_gail,nat_param_gail,if_gail=True) 
  suff_v=ic.compute_suff(ic.model_v,obs_v)
  suff_w=ic.compute_suff(ic.model_w,obs_w)
  suff_x=ic.compute_suff(ic.model_x,obs_x)
  suff_y=ic.compute_suff(ic.model_y,obs_y)
  suff_gail=ic.compute_suff(ic.model_gail,obs_gail,if_gail=True)
  orig_suff_v=ic.compute_suff(ic.model_v,orig_obs_v)
  orig_suff_w=ic.compute_suff(ic.model_w,orig_obs_w)
  orig_suff_x=ic.compute_suff(ic.model_x,orig_obs_x)
  orig_suff_y=ic.compute_suff(ic.model_y,orig_obs_y)
  orig_suff_gail=ic.compute_suff(ic.model_gail,obs_gail,if_gail=True)
  pdf_v=ic.compute_pdf(ic.model_v,nat_param_v,orig_obs_v,orig_suff_v,log_partition_v)
  pdf_w=ic.compute_pdf(ic.model_w,nat_param_w,orig_obs_w,orig_suff_w,log_partition_w)
  pdf_x=ic.compute_pdf(ic.model_x,nat_param_x,orig_obs_x,orig_suff_x,log_partition_x)
  pdf_y=ic.compute_pdf(ic.model_y,nat_param_y,orig_obs_y,orig_suff_y,log_partition_y)
  pdf_v=pdf_v.prod(axis=0)
  pdf_w=pdf_w.prod(axis=0)
  pdf_x=pdf_x.prod(axis=0)
  pdf_y=pdf_y.prod(axis=0)
  pdf_gail=ic.compute_pdf(ic.model_gail,nat_param_gail,orig_obs_gail,orig_suff_gail,log_partition_gail,if_gail=True)
  base_pdf_v=ic.compute_pdf(ic.model_v,nat_param_v,obs_v,suff_v,ic.base_log_partition_v)
  base_pdf_w=ic.compute_pdf(ic.model_w,nat_param_w,obs_w,suff_w,ic.base_log_partition_w)
  base_pdf_x=ic.compute_pdf(ic.model_x,nat_param_x,obs_x,suff_x,ic.base_log_partition_x)
  base_pdf_y=ic.compute_pdf(ic.model_y,nat_param_y,obs_y,suff_y,ic.base_log_partition_y)
  base_pdf_v=base_pdf_v.prod(axis=0)
  base_pdf_w=base_pdf_w.prod(axis=0)
  base_pdf_x=base_pdf_x.prod(axis=0)
  base_pdf_y=base_pdf_y.prod(axis=0)
  base_pdf_gail=ic.compute_pdf(ic.model_gail,nat_param_gail,obs_gail,suff_gail,ic.base_log_partition_gail,if_gail=True)
  hessian_v,hessian_w,hessian_x,hessian_y,hessian_gail =compute_hessian(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets)
  normalizer_v=np.empty(sample_sizes[iter])
  normalizer_w=np.empty(sample_sizes[iter])
  normalizer_x=np.empty(sample_sizes[iter])
  normalizer_y=np.empty(sample_sizes[iter])
  normalizer_gail=np.empty(sample_sizes[iter])
  identity=np.eye(ic.num_agents)
  identity_gail=np.eye(ic.dim_gail)
  for nn in xrange(sample_sizes[iter]):
   normalizer_v[nn] =np.abs(np.linalg.det(identity-(1./lams['v'])*hessian_v[:,:,nn]))
   normalizer_w[nn] =np.abs(np.linalg.det(identity-(1./lams['w'])*hessian_w[:,:,nn]))
   normalizer_x[nn] =np.abs(np.linalg.det(identity-(1./lams['x'])*hessian_x[:,:,nn]))
   normalizer_y[nn] =np.abs(np.linalg.det(identity-(1./lams['y'])*hessian_y[:,:,nn]))
   normalizer_gail[nn] =np.abs(np.linalg.det(identity_gail-(1./lams['gail'])*hessian_gail[:,:,nn]))
  like_ratio_v=np.exp(np.log(base_pdf_v[good_inds])-np.log(normalizer_v[good_inds]*pdf_v[good_inds]))
  like_ratio_w=np.exp(np.log(base_pdf_w[good_inds])-np.log(normalizer_w[good_inds]*pdf_w[good_inds]))
  like_ratio_x=np.exp(np.log(base_pdf_x[good_inds])-np.log(normalizer_x[good_inds]*pdf_x[good_inds]))
  like_ratio_y=np.exp(np.log(base_pdf_y[good_inds])-np.log(normalizer_y[good_inds]*pdf_y[good_inds]))
  like_ratio_gail=np.exp(np.log(base_pdf_gail[good_inds])-np.log(normalizer_gail[good_inds]*pdf_gail[good_inds]))
  one_over_sample_size=1./(sample_sizes[iter]*(1.-rho))
  imp_surrogate_v =one_over_sample_size*(like_ratio_v[np.newaxis,np.newaxis,:]*suff_v[:,:,good_inds]).sum(2)
  imp_surrogate_w =one_over_sample_size*(like_ratio_w[np.newaxis,np.newaxis,:]*suff_w[:,:,good_inds]).sum(2)
  imp_surrogate_x =one_over_sample_size*(like_ratio_x[np.newaxis,np.newaxis,:]*suff_x[:,:,good_inds]).sum(2)
  imp_surrogate_y =one_over_sample_size*(like_ratio_y[np.newaxis,np.newaxis,:]*suff_y[:,:,good_inds]).sum(2)
  imp_surrogate_gail =one_over_sample_size*(like_ratio_gail[np.newaxis,np.newaxis,:]*suff_gail[:,:,good_inds]).sum(2)
  coeff_v=step_sizes[iter]*imp_surrogate_v +(1-step_sizes[iter])*grad_log_partition_v
  coeff_w=step_sizes[iter]*imp_surrogate_w +(1-step_sizes[iter])*grad_log_partition_w
  coeff_y=step_sizes[iter]*imp_surrogate_y +(1-step_sizes[iter])*grad_log_partition_y
  coeff_x=step_sizes[iter]*imp_surrogate_x +(1-step_sizes[iter])*grad_log_partition_x 
  coeff_gail=step_sizes[iter]*imp_surrogate_gail +(1-step_sizes[iter])*grad_log_partition_gail
  nat_param_v=ic.optimize_cross_entropy(ic.model_v,coeff_v,ic.nat_base_v,if_verbose=if_verbose,center=ic.nat_base_v)
  nat_param_w=ic.optimize_cross_entropy(ic.model_w,coeff_w,ic.nat_base_w,if_verbose=if_verbose,center=ic.nat_base_w)
  nat_param_y=ic.optimize_cross_entropy(ic.model_y,coeff_y,ic.nat_base_y,if_verbose=if_verbose,center=ic.nat_base_y)
  old_nat_param_x=copy.deepcopy(nat_param_x)
  nat_param_x=ic.optimize_cross_entropy(ic.model_x,coeff_x,ic.nat_base_x,if_verbose=if_verbose,center=ic.nat_base_x)
  if if_update_gail:
   nat_param_gail=ic.optimize_cross_entropy(ic.model_gail,coeff_gail,ic.nat_base_gail,if_gail=True,if_gail_covar=if_gail_covar,if_verbose=if_verbose,center=ic.nat_base_gail[0,:])
  if iter in save_iter:
   all_nat_param_v[save_counter,:]=nat_param_v
   all_nat_param_w[save_counter,:]=nat_param_w
   all_nat_param_x[save_counter,:]=nat_param_x
   all_nat_param_y[save_counter,:]=nat_param_y
   all_nat_param_gail[save_counter,:]=nat_param_gail
   save_counter+=1
 print " ------------------------------------------------------------ "
 print " --------------- Ending Cross Entropy Loop ------------------ "
 return nat_param_v,nat_param_w,nat_param_x,nat_param_y,nat_param_gail
"""compute_monge_map(lam, stepsize, iter,                       init_obs_v, init_obs_w, init_obs_x,                       init_obs_y, init_obs_gail):
Perform gd_num_iter number of gradient ascent steps on the objective
f(Z) - lam * .5 * norm(Z - Z_init, 2)^2
with respect to X. Each column of init_obs numpy array contains the
Z_init corresponding to the random variable.
obs_v, obs_w, obs_x, obs_y are num_agents-by-sample_size numpy arrays
obs_gail is a dim_gail-by-sample_size numpy array
lams is a dictionariy containing lambdas for v, w, x, y, gail
gd_step_sizes is a dictionary containing stepsizes for v, w, x, y, gail
"""
def compute_monge_map(ic,lams,gd_step_sizes,gd_num_iter,orig_obs_v,orig_obs_w,orig_obs_x,orig_obs_y,orig_obs_gail,compute_gradient,sockets):
 obs_v=orig_obs_v
 obs_w=orig_obs_w
 obs_x=orig_obs_x
 obs_y=orig_obs_y
 obs_gail=orig_obs_gail
 for iter in xrange(gd_num_iter):
  grad_v,grad_w,grad_x,grad_y,grad_gail =compute_gradient(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets)
  obs_v+=gd_step_sizes['v']*(grad_v-lams['v']*(obs_v-orig_obs_v))
  obs_w+=gd_step_sizes['w']*(grad_w-lams['w']*(obs_w-orig_obs_w))
  obs_x+=gd_step_sizes['x']*(grad_x-lams['x']*(obs_x-orig_obs_x))
  obs_y+=gd_step_sizes['y']*(grad_y-lams['y']*(obs_y-orig_obs_y))
  obs_gail+=gd_step_sizes['gail'] *(grad_gail-lams['gail']*(obs_gail-orig_obs_gail))
  if ic.model_v=='expo':
   obs_v[obs_v<0]=tol
  if ic.model_w=='beta':
   obs_w[obs_w<0]=tol
   obs_w[obs_w>1.]=1.-tol
  if ic.model_y=='beta':
   obs_y[obs_y<0]=tol
   obs_y[obs_y>1.]=1.-tol
 return obs_v,obs_w,obs_x,obs_y,obs_gail
"""
compute_objective(obs_initial, obs_gail)
This is a placeholder for Matt.
obs_v, obs_w, obs_x, obs_y are num_agents-by-sample_size numpy arrays
obs_gail is a dim_gail-by-sample_size numpy array
"""
def compute_objective_single(sample,gail_sample,numcars,lanes,index,run,socket,if_update_gail=False):
 stuffsize=4
 wholething=np.zeros(numcars*(stuffsize+1)+2)
 for i in xrange(stuffsize):
  wholething[i*numcars:(i+1)*numcars]=sample[i::stuffsize]
 wholething[stuffsize*numcars:(stuffsize+1)*numcars]=np.asarray(lanes)
 wholething[-2]=index
 wholething[-1]=run
 if if_update_gail:
  wholething=np.concatenate([wholething,gail_sample],axis=0)
 send_array(socket,np.ascontiguousarray(wholething))
def compute_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,ic,run,source_worker_socket,source_sink_req_socket,if_update_gail=False):
 numcars,sample_size=obs_v.shape
 obs_v=obs_v*ic.scale_v+ic.center_v
 obs_w=obs_w*ic.scale_w+ic.center_w
 obs_x=obs_x*ic.scale_x+ic.center_x
 obs_y=obs_y*ic.scale_y+ic.center_y
 source_sink_req_socket.send(str(sample_size))
 source_sink_req_socket.recv()
 for i in xrange(sample_size):
  sample=np.stack((obs_v[:,i],obs_w[:,i],obs_x[:,i],obs_y[:,i]),axis=1).flatten()
  compute_objective_single(sample,obs_gail[:,i],numcars,ic.lanes,i,run,source_worker_socket,if_update_gail=if_update_gail)
 source_sink_req_socket.send("done")
 objectives=np.copy(recv_array(source_sink_req_socket))
 return-1*objectives
"""
compute_gradient(obs_initial, obs_gail)
This is a placeholder for Aman.
"""
def placeholder_gradient(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0):
 num_agents=obs_v.shape[0]
 sample_size=obs_v.shape[1]
 dim_gail=obs_gail.shape[0]
 grad_v=np.random.rand(num_agents,sample_size)
 grad_w=np.random.rand(num_agents,sample_size)
 grad_x=np.random.rand(num_agents,sample_size)
 grad_y=np.random.rand(num_agents,sample_size)
 grad_gail=np.random.rand(dim_gail,sample_size)
 return grad_v,grad_w,grad_x,grad_y,grad_gail
def placeholder_hessian(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0):
 num_agents=obs_v.shape[0]
 sample_size=obs_v.shape[1]
 dim_gail=obs_gail.shape[0]
 hessian_v=np.empty((num_agents,num_agents,sample_size))
 hessian_w=np.empty((num_agents,num_agents,sample_size))
 hessian_x=np.empty((num_agents,num_agents,sample_size))
 hessian_y=np.empty((num_agents,num_agents,sample_size))
 hessian_gail=np.empty((dim_gail,dim_gail,sample_size))
 identity=np.eye(num_agents)
 identity_gail=np.eye(dim_gail)
 for nn in xrange(sample_size):
  hessian_v[:,:,nn]=identity
  hessian_w[:,:,nn]=identity
  hessian_x[:,:,nn]=identity
  hessian_y[:,:,nn]=identity
  hessian_gail[:,:,nn]=identity_gail
 return hessian_v,hessian_w,hessian_x,hessian_y,hessian_gail
# Created by pyminifier (https://github.com/liftoff/pyminifier)

