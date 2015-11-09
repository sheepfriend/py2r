from numpy import *
import numpy
import os
import scipy.stats

seq=arange

def matrix(num,**args):
	try:
		nrow=args['nrow']
		ncol=num.size/nrow
	except:
		try:
			ncol=args['ncol']
			nrow=num.size/ncol
		except:
			raise Exception('matrix','should contain nrow or ncol.')
	return num[0:(nrow*ncol)].copy().reshape(nrow,ncol)

def t(x):
	return numpy.matrix.transpose(x)

def solve(x):
	return numpy.linalg.inv(x)

def cbind(a,b):
	if a.ndim==1 and b.ndim==2:
		result=array(0.,dim=c(a.size,1+b.shape[1]))
		result[:,0]=a.copy()
		result[:,1:(b.shape[1]+1)]=b.copy()
		return result
	elif a.ndim==2 and b.ndim==1:
		result=array(0.,dim=c(b.size,1+a.shape[1]))
		result[:,0:(a.shape[1])]=a.copy()
		result[:,(a.shape[1])]=b.copy()
		return result
	else:
		return hstack((a.copy(),b.copy()))


def rbind(a,b):
	if a.ndim==1 and b.ndim==2:
		result=array(0.,dim=c(1+b.shape[0],a.size))
		result[0,:]=a.copy()
		result[1:(b.shape[0]+1),:]=b.copy()
		return result
	elif a.ndim==2 and b.ndim==1:
		result=array(0.,dim=c(1+a.shape[0],b.size))
		result[0:(a.shape[0]),:]=a.copy()
		result[(a.shape[0]),:]=b.copy()
		return result
	else:
		return vstack((a.copy(),b.copy()))


def c(*m):
	if len(m)==0:
		return numpy.array([])
	elif type(m[0])==type([]):
		k=[]
		for i in m:
			k+=i
		return numpy.array(m[0])
	elif type(m[0])==type(numpy.array([])):
		return m[0].copy().flatten()
	else:
		return numpy.array(m)

def rep(num,length):
	result=[]
	for i in range(0,length):
		result.append(num)
	return c(result)

def array(num,dim):	
	if type(num)==type(0):
		return numpy.array(rep(num,prod(dim))).reshape(dim)
	elif type(num)==type(0.):
		return numpy.array(rep(num,prod(dim))).reshape(dim)
	elif type(num)!=type(numpy.array([])):
		return numpy.array(num).reshape(dim)
	else:
		return num.copy().reshape(dim)

def compare(a,b,what):
	if what=='==':
		return a==b
	elif what=='>':
		return a>b
	elif what=='<':
		return a<b
	elif what=='>=':
		return a>=b
	elif what=='<=':
		return a<=b
	elif what=='!=':
		return a!=b
	else:
		raise Exception('compare','invalid syntax')


def which(a,b,what):
	if type(a)==type(0) and type(b)!=type(0):
		return [x for x in range(0,b.size) if compare(a,b[x],what)]
	elif type(b)==type(0) and type(a)!=type(0):
		return [x for x in range(0,a.size) if compare(a[x],b,what)]
	elif type(b)==type(0) and type(a)==type(0):
		return [x for x in range(0,1) if comapre(a,b,what)]
	else:
		if a.size==b.size:
			return [x for x in range(0,a.size) if compare(a[x],b[x],what)]
		else:
			raise Exception('which','mismatched arrays')


def which_or(*argv):
	if len(argv)==1:
		return argv[0]
	else:
		result=argv[0]
		for i in range(1,len(argv)):
			result=union1d(result,argv[i])
		return result

def which_and(*argv):
	if len(argv)==1:
		return argv[0]
	else:
		result=argv[0]
		for i in range(1,len(argv)):
			result=intersect1d(result,argv[i])
		return result

def apply_all(arr,fun):
	result=array(0,arr.shape)
	num=0
	for i in arr.flatten():
		result[num]=fun(i)
		num=num+1
	return result
	
def apply_row(arr,fun):
	size=arr[:,1].size
	result=rep(0,size)
	for i in range(0,size):
		result[i]=fun(arr[i,:])
	return result

def apply_col(arr,fun):
	size=arr[1,:].size
	result=rep(0,size)
	for i in range(0,size):
		result[i]=fun(arr[:,i])
	return result

def read_csv(filename,**argv):		
	f=open(filename,'r')
	try:
		argv['header']
	except:
		argv['header']=False
	try:
		argv['sep']
	except:
		argv['sep']='\t'
	content=f.read().split('\n')
	num=len(content[0].split(argv['sep']))
	result=[]
	if argv['header']:
		start=1
	else:
		start=0
	for i in range(start,len(content)):
		result+=content[i].split(argv['sep'])
	return matrix(c(result),ncol=num)

def write_csv(arr,filename,**argv):
	f=open(filename,'w')
	try:
		argv['sep']
	except:
		argv['sep']='\t'
	try:
		argv['header']
		f.write(argv['sep'].join(argv['header']).encode('utf-8'))
		f.write('\n')
		start=1
	except:
		start=0
	for i in range(start,arr[:,1].size):
		f.write(argv['sep'].join(arr[:,i]).encode('utf-8'))
		f.write('\n')
	f.close()

def dnorm(x,**argv):
	try:
		argv['sd']
	except:
		argv['sd']=1.
	try:
		argv['mean']
	except:
		argv['mean']=0.
	if argv['sd']<=0:
		raise Exception('norm','invalid sd')
	else:
		return scipy.stats.norm.pdf((x-argv['mean'])/argv['sd'])

def pnorm(x,**argv):
	try:
		argv['sd']
	except:
		argv['sd']=1.
	try:
		argv['mean']
	except:
		argv['mean']=0.
	if argv['sd']<=0:
		raise Exception('norm','invalid sd')
	else:
		return scipy.stats.norm.cdf((x-argv['mean'])/argv['sd'])


def qnorm(x,**argv):
	try:
		argv['sd']
	except:
		argv['sd']=1.
	try:
		argv['mean']
	except:
		argv['mean']=0.
	if argv['sd']<=0:
		raise Exception('norm','invalid sd')
	else:
		return scipy.stats.norm.ppf((x+argv['mean'])*argv['sd'])

def rnorm(x,**argv):
	try:
		argv['sd']
	except:
		argv['sd']=1.
	try:
		argv['mean']
	except:
		argv['mean']=0.
	if argv['sd']<=0:
		raise Exception('norm','invalid sd')
	else:
		y=c(scipy.stats.norm.rvs(size=x))
		return (y+argv['mean'])*argv['sd']

def dpois(x,mu):
	return scipy.stats.poisson.pmf(x,mu=mu)

def ppois(x,mu):
	return scipy.stats.poisson.cdf(x,mu=mu)

def qpois(x,mu):
	return scipy.stats.poisson.ppf(x,mu=mu)

def rpois(x,mu):
	return scipy.stats.poisson.rvs(mu,size=x)

def dexp(x,mu):	
	return scipy.stats.expon.pdf(x,scale=mu)

def pexp(x,mu):
	return scipy.stats.expon.cdf(x,scale=mu)

def qexp(x,mu):
	return scipy.stats.expon.ppf(x,scale=mu)

def rexp(x,mu):
	return scipy.stats.expon.rvs(scale,size=x)

def dchisq(x,df):	
	return scipy.stats.chi2.pdf(x,df)

def pchisq(x,df):
	return scipy.stats.chi2.cdf(x,df)

def qchisq(x,df):
	return scipy.stats.chi2.ppf(x,df)

def rchisq(x,df):
	return scipy.stats.chi2.rvs(df,size=x)

def dt(x,df):	
	return scipy.stats.t.pdf(x,df)

def pt(x,df):
	return scipy.stats.t.cdf(x,df)

def qt(x,df):
	return scipy.stats.t.ppf(x,df)

def rt(x,df):
	return scipy.stats.t.rvs(df,size=x)

def df(x,df1,df2):	
	return scipy.stats.f.pdf(x,df1,df2)

def pf(x,df1,df2):
	return scipy.stats.f.cdf(x,df1,df2)

def qf(x,df1,df2):
	return scipy.stats.f.ppf(x,df1,df2)

def rf(x,df1,df2):
	return scipy.stats.f.rvs(df1,df2,size=x)

def dgamma(x,df):	
	return scipy.stats.gamma.pdf(x,df)

def pgamma(x,df):
	return scipy.stats.gamma.cdf(x,df)

def qgamma(x,df):
	return scipy.stats.gamma.ppf(x,df)

def rgamma(x,df):
	return scipy.stats.gamma.rvs(df,size=x)

def dbeta(x,df1,df2):	
	return scipy.stats.beta.pdf(x,df1,df2)

def pbeta(x,df1,df2):
	return scipy.stats.beta.cdf(x,df1,df2)

def qbeta(x,df1,df2):
	return scipy.stats.beta.ppf(x,df1,df2)

def rbeta(x,df1,df2):
	return scipy.stats.beta.rvs(df1,df2,size=x)

def dlognorm(x,df):	
	return scipy.stats.lognorm.pdf(x,df)

def plognorm(x,df):
	return scipy.stats.lognorm.cdf(x,df)

def qlognorm(x,df):
	return scipy.stats.lognorm.ppf(x,df)

def rlognorm(x,df):
	return scipy.stats.lognorm.rvs(df,size=x)

def runif(x):
	return scipy.stats.uniform.rvs(size=x)

def lm(y,xs):
	if xs.ndim==1:
		slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(xs,y)
		return {"para": c(slope,intercept),
				"r_value":r_value,
				"p_value":p_value,
				"std_err":std_err
				}
	else:
		x=cbind(rep(1,xs[:,1].size),xs)
		return {"para":dot(dot(solve(dot(t(x),x)),t(x)),y)}
