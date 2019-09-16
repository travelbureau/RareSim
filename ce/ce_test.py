def test_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0,ic=0):
 return .5*(np.sum(np.power(obs_v,2.)+np.power(obs_w,2.)+np.power(obs_x,2.)+np.power(obs_y,2.),0)+np.sum(np.power(obs_gail,2.),0))
def test_objective_simple(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0,ic=0):
 return obs_x[0,:]
def test_gradient(obs_v,obs_w,obs_x,obs_y,obs_gail,sockets=0):
 return obs_v,obs_w,obs_x,obs_y,obs_gail
def test_ce(seed_value):
 np.random.seed(seed_value)
 num_agents=2
 base_v=np.abs(np.random.randn(num_agents,1))
 base_w=np.array([[1.8,4.1],[2.1,2.8]])
 base_x=np.column_stack((np.random.randn(num_agents),10*np.random.rand(num_agents)))
 base_y=np.array([[1.2,2.3],[3.1,1.8]])
 print 'base_v'
 print base_v
 print 'base_w'
 print base_w
 print 'base_x'
 print base_x
 print 'base_y'
 print base_y
 dim_gail=20
 mu=5.*np.random.randn(dim_gail)
 sigma=np.diag(np.random.rand(dim_gail))
 base_gail=np.vstack((mu,sigma))
 print 'gail mean'
 print mu
 lanes=np.array([4,5])
 ic=init_conds.init_conds(base_v,base_w,base_x,base_y,base_gail,lanes)
 base_mean=np.power(base_v,2.).sum()+2.*num_agents +np.power(base_x[:,0],2.).sum() +np.power(mu,2.).sum()
 levels_mult=np.array([1.,3.,5.,10.])
 print 'levels'
 print levels_mult*base_mean
 sample_size=np.int(1000000.)
 num_iter=100
 save_iter=(np.array([30,70,100])-1).astype(int)
 rho=.95
 alpha=.9
 ce_sample_size=5000
 ns_naive=sample_size+np.int(ce_sample_size*(save_iter[-1]+1))
 obs_v=ic.sample_obs(ic.model_v,ic.nat_base_v,ns_naive)
 obs_w=ic.sample_obs(ic.model_w,ic.nat_base_w,ns_naive)
 obs_x=ic.sample_obs(ic.model_x,ic.nat_base_x,ns_naive)
 obs_y=ic.sample_obs(ic.model_y,ic.nat_base_y,ns_naive)
 obs_gail=ic.sample_obs(ic.model_gail,ic.nat_base_gail,ns_naive,if_gail=True)
 objs_naive=test_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
 objs_ce=np.empty((save_iter.size,sample_size))
 like_ratio=np.empty((save_iter.size,sample_size))
 sample_sizes=np.full(num_iter,ce_sample_size).astype(int)
 step_sizes=np.full(num_iter,alpha)
 all_nat_param_v,all_nat_param_w,all_nat_param_x, all_nat_param_y,all_nat_param_gail =cross_entropy(ic,rho,levels_mult[-1]*base_mean,num_iter,save_iter,sample_sizes,step_sizes,test_objective)
 counter=0
 for num_iter in save_iter:
  print '----------------------- Processing '+str(num_iter+1) +' -------------------'
  obs_v=ic.sample_obs(ic.model_v,all_nat_param_v[counter,:],sample_size)
  obs_w=ic.sample_obs(ic.model_w,all_nat_param_w[counter,:],sample_size)
  obs_x=ic.sample_obs(ic.model_x,all_nat_param_x[counter,:],sample_size)
  obs_y=ic.sample_obs(ic.model_y,all_nat_param_y[counter,:],sample_size) 
  obs_gail=ic.sample_obs(ic.model_gail,all_nat_param_gail[counter,:],sample_size,if_gail=True)
  objs_ce[counter,:]=test_objective(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
  print 'mean_ce'
  print np.mean(objs_ce[counter,:])
  like_ratio_v,like_ratio_w,like_ratio_x,like_ratio_y,like_ratio_gail =ic.compute_like_ratio(obs_v,obs_w,obs_x,obs_y,obs_gail,all_nat_param_v[counter,:],all_nat_param_w[counter,:],all_nat_param_x[counter,:],all_nat_param_y[counter,:],all_nat_param_gail[counter,:])
  like_ratio[counter,:]=like_ratio_v*like_ratio_w*like_ratio_x *like_ratio_y*like_ratio_gail
  print '---------------- nat_param ---------------------'
  print 'all_nat_param_v'
  print all_nat_param_v[counter,:]
  print 'all_nat_param_w'
  print all_nat_param_w[counter,:]
  print 'all_nat_param_x'
  print all_nat_param_x[counter,:]
  print 'all_nat_param_y'
  print all_nat_param_y[counter,:]
  counter+=1
 est_naive=np.empty((levels_mult.size,save_iter.size))
 std_naive=np.empty((levels_mult.size,save_iter.size))
 est_ce=np.empty((levels_mult.size,save_iter.size))
 std_ce=np.empty((levels_mult.size,save_iter.size))
 num_events_naive=np.zeros((levels_mult.size,save_iter.size))
 num_events_ce=np.zeros((levels_mult.size,save_iter.size))
 for ii in xrange(levels_mult.size):
  level=base_mean*levels_mult[ii]
  objs_naive_level=(objs_naive>level).astype(float)
  for jj in xrange(save_iter.size):
   ns_naive =np.minimum(np.int((save_iter[jj]+1)*ce_sample_size),sample_size)
   est_naive[ii,jj]=np.mean(objs_naive_level[:ns_naive])
   std_naive[ii,jj]=np.std(objs_naive_level[:ns_naive])
   num_events_naive[ii,jj]=np.sum(objs_naive_level[:ns_naive])
   objs_ce_level=like_ratio[jj,:] *(objs_ce[jj,:]>level).astype(float)
   est_ce[ii,jj]=np.mean(objs_ce_level[:sample_size])
   std_ce[ii,jj]=np.std(objs_ce_level[:sample_size])
   num_events_ce[ii,jj]=np.sum(objs_ce[jj,:]>level)
 filename=os.getcwd()+'/rho='+str(rho)+'_alpha='+str(alpha) +'_cesample='+str(ce_sample_size)+'.h5'
 with h5py.File(filename,'w')as f:
  f.create_dataset('levels_mult',data=levels_mult)
  f.create_dataset('save_iter',data=save_iter)
  f.create_dataset('rho',data=rho)
  f.create_dataset('alpha',data=alpha)
  f.create_dataset('ce_sample_size',data=ce_sample_size)
  f.create_dataset('all_nat_param_v',data=all_nat_param_v)
  f.create_dataset('all_nat_param_w',data=all_nat_param_w)
  f.create_dataset('all_nat_param_x',data=all_nat_param_x)
  f.create_dataset('all_nat_param_y',data=all_nat_param_y)
  f.create_dataset('all_nat_param_gail',data=all_nat_param_gail)
  f.create_dataset('base_v',data=base_v)
  f.create_dataset('base_w',data=base_w)
  f.create_dataset('base_x',data=base_x)
  f.create_dataset('base_y',data=base_y)
  f.create_dataset('base_gail',data=base_gail)
  f.create_dataset('objs_naive',data=objs_naive)
  f.create_dataset('objs_ce',data=objs_ce)
  f.create_dataset('like_ratio_ce',data=like_ratio)
 return est_naive,std_naive,est_ce,std_ce, num_events_naive,num_events_ce
def test_ce_simple(seed_value):
 np.random.seed(seed_value)
 num_agents=2
 base_v=np.abs(np.random.randn(num_agents,1))
 base_w=np.array([[1.8,4.1],[2.1,2.8]])
 base_x=np.column_stack((np.random.randn(num_agents),10*np.random.rand(num_agents)))
 base_y=np.array([[1.2,2.3],[3.1,1.8]])
 print 'base_v'
 print base_v
 print 'base_w'
 print base_w
 print 'base_x'
 print base_x
 print 'base_y'
 print base_y
 dim_gail=20
 mu=5.*np.random.randn(dim_gail)
 sigma=np.diag(np.random.rand(dim_gail))
 base_gail=np.vstack((mu,sigma))
 print 'gail mean'
 print mu
 lanes=np.array([4,5])
 ic=init_conds.init_conds(base_v,base_w,base_x,base_y,base_gail,lanes)
 base_mean=np.abs(base_x[0,0])
 levels_mult=np.array([2.,10.,14.,18.])
 print 'levels'
 print levels_mult*base_mean
 sample_size=np.int(1000000.)
 num_iter=100
 save_iter=(np.array([30,70,100])-1).astype(int)
 rho=.9
 alpha=.9
 ce_sample_size=10000
 ns_naive=sample_size+np.int(ce_sample_size*(save_iter[-1]+1))
 obs_v=ic.sample_obs(ic.model_v,ic.nat_base_v,ns_naive)
 obs_w=ic.sample_obs(ic.model_w,ic.nat_base_w,ns_naive)
 obs_x=ic.sample_obs(ic.model_x,ic.nat_base_x,ns_naive)
 obs_y=ic.sample_obs(ic.model_y,ic.nat_base_y,ns_naive)
 obs_gail=ic.sample_obs(ic.model_gail,ic.nat_base_gail,ns_naive,if_gail=True)
 objs_naive=test_objective_simple(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
 objs_ce=np.empty((save_iter.size,sample_size))
 like_ratio=np.empty((save_iter.size,sample_size))
 sample_sizes=np.full(num_iter,ce_sample_size).astype(int)
 step_sizes=np.full(num_iter,alpha)
 all_nat_param_v,all_nat_param_w,all_nat_param_x, all_nat_param_y,all_nat_param_gail =cross_entropy(ic,rho,levels_mult[-1]*base_mean,num_iter,save_iter,sample_sizes,step_sizes,test_objective_simple)
 print 'full likelihoods'
 counter=0
 for num_iter in save_iter:
  print '----------------------- Processing '+str(num_iter+1) +' -------------------'
  obs_v=ic.sample_obs(ic.model_v,all_nat_param_v[counter,:],sample_size)
  obs_w=ic.sample_obs(ic.model_w,all_nat_param_w[counter,:],sample_size)
  obs_x=ic.sample_obs(ic.model_x,all_nat_param_x[counter,:],sample_size)
  obs_y=ic.sample_obs(ic.model_y,all_nat_param_y[counter,:],sample_size) 
  obs_gail=ic.sample_obs(ic.model_gail,all_nat_param_gail[counter,:],sample_size,if_gail=True)
  objs_ce[counter,:]=test_objective_simple(obs_v,obs_w,obs_x,obs_y,obs_gail,0.)
  print 'mean_ce'
  print np.mean(objs_ce[counter,:])
  like_ratio_v,like_ratio_w,like_ratio_x,like_ratio_y,like_ratio_gail =ic.compute_like_ratio(obs_v,obs_w,obs_x,obs_y,obs_gail,all_nat_param_v[counter,:],all_nat_param_w[counter,:],all_nat_param_x[counter,:],all_nat_param_y[counter,:],all_nat_param_gail[counter,:])
  like_ratio[counter,:]=like_ratio_v*like_ratio_w*like_ratio_x *like_ratio_y*like_ratio_gail
  print '---------------- nat_param ---------------------'
  print 'all_nat_param_v'
  print all_nat_param_v[counter,:]
  print 'all_nat_param_w'
  print all_nat_param_w[counter,:]
  print 'all_nat_param_x'
  print all_nat_param_x[counter,:]
  print 'all_nat_param_y'
  print all_nat_param_y[counter,:]
  counter+=1
 est_naive=np.empty((levels_mult.size,save_iter.size))
 std_naive=np.empty((levels_mult.size,save_iter.size))
 est_ce=np.empty((levels_mult.size,save_iter.size))
 std_ce=np.empty((levels_mult.size,save_iter.size))
 num_events_naive=np.zeros((levels_mult.size,save_iter.size))
 num_events_ce=np.zeros((levels_mult.size,save_iter.size))
 actual=np.empty(levels_mult.size)
 for ii in xrange(levels_mult.size):
  level=base_mean*levels_mult[ii]
  objs_naive_level=(objs_naive>level).astype(float)
  actual[ii]=1-stats.norm.cdf((level-base_x[0,0])/np.sqrt(base_x[0,1]))
  for jj in xrange(save_iter.size):
   ns_naive =np.minimum(np.int((save_iter[jj]+1)*ce_sample_size),sample_size)
   est_naive[ii,jj]=np.mean(objs_naive_level[:ns_naive])
   std_naive[ii,jj]=np.std(objs_naive_level[:ns_naive])
   num_events_naive[ii,jj]=np.sum(objs_naive_level[:ns_naive])
   objs_ce_level=like_ratio[jj,:] *(objs_ce[jj,:]>level).astype(float)
   est_ce[ii,jj]=np.mean(objs_ce_level[:sample_size])
   std_ce[ii,jj]=np.std(objs_ce_level[:sample_size])
   num_events_ce[ii,jj]=np.sum(objs_ce[jj,:]>level)
 filename=os.getcwd()+'/rho='+str(rho)+'_alpha='+str(alpha) +'_cesample='+str(ce_sample_size)+'.h5'
 with h5py.File(filename,'w')as f:
  f.create_dataset('levels_mult',data=levels_mult)
  f.create_dataset('save_iter',data=save_iter)
  f.create_dataset('rho',data=rho)
  f.create_dataset('alpha',data=alpha)
  f.create_dataset('ce_sample_size',data=ce_sample_size)
  f.create_dataset('all_nat_param_v',data=all_nat_param_v)
  f.create_dataset('all_nat_param_w',data=all_nat_param_w)
  f.create_dataset('all_nat_param_x',data=all_nat_param_x)
  f.create_dataset('all_nat_param_y',data=all_nat_param_y)
  f.create_dataset('all_nat_param_gail',data=all_nat_param_gail)
  f.create_dataset('base_v',data=base_v)
  f.create_dataset('base_w',data=base_w)
  f.create_dataset('base_x',data=base_x)
  f.create_dataset('base_y',data=base_y)
  f.create_dataset('base_gail',data=base_gail)
  f.create_dataset('objs_naive',data=objs_naive)
  f.create_dataset('objs_ce',data=objs_ce)
  f.create_dataset('like_ratio_ce',data=like_ratio)
 return est_naive,std_naive,est_ce,std_ce, num_events_naive,num_events_ce,actual
# Created by pyminifier (https://github.com/liftoff/pyminifier)

