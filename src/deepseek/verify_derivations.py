"""Independent tiny-graph numerical VJPs + ledger/scenario/display invariants."""
from collections import Counter
from copy import deepcopy
import json
import re
from pathlib import Path
import numpy as np
import derivation_ledger as L

ROOT=Path(__file__).resolve().parents[2]
pack=json.loads((ROOT/'src/deepseek/derivation_data.json').read_text(encoding='utf-8'))
rng=np.random.default_rng(20261015)
errors={}

def numeric(fn,x):
    result=np.zeros_like(x);eps=1e-6
    for ix in np.ndindex(x.shape):
        before=x[ix];x[ix]=before+eps;positive=fn();x[ix]=before-eps;negative=fn();x[ix]=before
        result[ix]=(positive-negative)/(2*eps)
    return result

def verify(name,fn,x,gradient):
    error=float(np.max(np.abs(numeric(fn,x)-gradient)))
    assert error<8e-6,(name,error)
    errors[name]=error

def sigmoid(x):return 1/(1+np.exp(-x))
def sm(x,axis=-1):
    p=np.exp(x-x.max(axis=axis,keepdims=True));return p/p.sum(axis=axis,keepdims=True)

# HC: source order, delayed pre as a separate output, and full Sinkhorn reverse.
c,h,it=2,3,3;n=c*h;mix=2*c+c*c;eps=1e-3
x=rng.normal(size=(c,h));w=rng.normal(scale=.15,size=(n,mix));sc=np.array([.3,.4,.2]);base=rng.normal(scale=.1,size=mix);pin=np.array([.4,.8])
gy=rng.normal(size=(c,h));gp=rng.normal(size=c)
def hc_forward():
    flat=x.ravel();r=(np.mean(flat*flat)+eps)**-.5;t=flat@w;m=t*r
    group=np.repeat(sc,[c,c,c*c]);a=m*group+base;p=sigmoid(a[:c]);post=2*sigmoid(a[c:2*c])
    prob=sm(a[2*c:].reshape(c,c));comb=prob+eps;states=[]
    for axis in [0]+[j for _ in range(it-1) for j in [1,0]]:
        den=comb.sum(axis=axis,keepdims=True)+eps;y=comb/den;states.append((y,den,axis));comb=y
    z=np.sum(pin[:,None]*x,axis=0);u=np.tanh(z);y=post[:,None]*u+comb.T@x
    return locals()
def hcloss():
    v=hc_forward();return np.sum(v['y']*gy)+np.sum((v['p']+eps)*gp)
v=hc_forward();du=(v['post'][:,None]*gy).sum(axis=0);dz=du*(1-v['u']**2)
db=(gy*v['u']).sum(axis=1);dc=x@gy.T
for y,den,axis in reversed(v['states']):dc=(dc-(dc*y).sum(axis=axis,keepdims=True))/den
dc=v['prob']*(dc-(dc*v['prob']).sum(axis=1,keepdims=True))
da=np.r_[gp*v['p']*(1-v['p']),db*2*(v['post']/2)*(1-v['post']/2),dc.ravel()]
dm=da*v['group'];dt=dm*v['r'];dr=np.sum(dm*v['t'])
dx=v['comb']@gy+pin[:,None]*dz+(dt@w.T).reshape(c,h)-dr*v['r']**3/n*x
verify('HC.full.dR',hcloss,x,dx);verify('HC.full.dW',hcloss,w,np.outer(x.ravel(),dt))
verify('HC.full.dbase',hcloss,base,da)
verify('HC.full.dscale',hcloss,sc,np.array([np.sum((da*v['m'])[i:j]) for i,j in [(0,c),(c,2*c),(2*c,mix)]]))
verify('HC.delayed.dpre_in',hcloss,pin,x@dz)

# MoE: continuous router through fixed Top-2, clamps, weights BEFORE Down.
m,dim,hidden,e,k=2,3,2,4,2
x=rng.normal(scale=.4,size=(m,dim));wr=rng.normal(scale=.2,size=(dim,e));bias=np.array([2.,1.,-1.,-2.]);rho=1.5
weights=[(rng.normal(scale=.3,size=(dim,hidden)),rng.normal(scale=.3,size=(hidden,dim)),rng.normal(scale=.3,size=(dim,hidden))) for _ in range(k+1)]
dy=rng.normal(size=(m,dim))
def moe_forward():
    z=x@wr;s=np.sqrt(np.logaddexp(0,z));indices=np.argsort(s+bias,axis=1)[:,-k:]
    # Both rows choose experts 0/1, but selection order can differ.
    selected=np.take_along_axis(s,indices,axis=1);den=selected.sum(axis=1,keepdims=True)+1e-20;p=rho*selected/den
    cache=[];out=np.zeros_like(x)
    for index,(w1,w2,w3) in enumerate(weights):
        gate=x@w1;up=x@w3;gc=np.minimum(gate,10);uc=np.clip(up,-10,10);sig=sigmoid(gc);t=gc*sig*uc
        route=np.ones((m,1)) if index==k else np.array([p[row,np.where(indices[row]==index)[0][0]] for row in range(m)])[:,None]
        mid=t*route if index<k else t;y=mid@w2;out+=y;cache.append(locals().copy())
    return z,s,indices,selected,den,p,cache,out
def moeloss():return np.sum(moe_forward()[-1]*dy)
z,s,indices,selected,den,p,cache,out=moe_forward();dx=np.zeros_like(x);dp=np.zeros((m,k));grads=[]
for index,((w1,w2,w3),a) in enumerate(zip(weights,cache)):
    dv=dy@w2.T;dt=dv*a['route'] if index<k else dv
    if index<k:
        dw=np.sum(dv*a['t'],axis=1)
        for row in range(m):dp[row,np.where(indices[row]==index)[0][0]]=dw[row]
    dg=dt*a['uc']*(a['sig']+a['gc']*a['sig']*(1-a['sig']))*(a['gate']<10)
    dup=dt*a['gc']*a['sig']*(np.abs(a['up'])<10)
    dx+=dg@w1.T+dup@w3.T;grads.append((x.T@dg,a['mid'].T@dy,x.T@dup))
ds_sel=rho/den*dp-rho/den**2*np.sum(dp*selected,axis=1,keepdims=True);ds=np.zeros_like(s)
np.put_along_axis(ds,indices,ds_sel,axis=1);dz=ds*sigmoid(z)/(2*s);dx+=dz@wr.T
verify('MoE.full.dX',moeloss,x,dx);verify('MoE.router.dW',moeloss,wr,x.T@dz)
for index,(triple,gradient) in enumerate(zip(weights,grads)):
    for name,t,g in zip(['W1','W2','W3'],triple,gradient):verify(f'MoE.expert{index}.{name}',moeloss,t,g)

# Compressor: per-feature softmax over tokens, then RMSNorm, fully differentiable.
x=rng.normal(size=(2,3));wc=rng.normal(size=(3,2));wg=rng.normal(size=(3,2));gamma=rng.normal(size=2);dy=rng.normal(size=2)
def closs():
    kv=x@wc;alpha=sm(x@wg,axis=0);v=np.sum(alpha*kv,axis=0);r=(np.mean(v*v)+.01)**-.5
    return np.sum(v*r*gamma*dy)
kv=x@wc;alpha=sm(x@wg,axis=0);v=(alpha*kv).sum(axis=0);r=(np.mean(v*v)+.01)**-.5
g=dy*gamma;dv=r*g-r**3/2*v*np.sum(g*v);dkv=alpha*dv;da=kv*dv;dz=alpha*(da-(alpha*da).sum(axis=0))
verify('Compressor.full.dX',closs,x,dkv@wc.T+dz@wg.T);verify('Compressor.dW_KV',closs,wc,x.T@dkv);verify('Compressor.dW_gate',closs,wg,x.T@dz)

# Engram continuous gating including signed sqrt (away from clamp boundaries).
m,c,h=2,2,3
x=rng.normal(size=(m,c,h));key=rng.normal(size=(m,c,h));value=rng.normal(size=(m,h));qw=rng.normal(size=(c,h));kw=rng.normal(size=(c,h));dy=rng.normal(size=(m,c,h));eps=.01
def estate():
    w=qw*kw;rh=(np.mean(x*x,axis=-1)+eps)**-.5;rk=(np.mean(key*key,axis=-1)+eps)**-.5
    raw=np.sum(x*w*key,axis=-1);dot=raw*rh*rk*h**-.5;root=np.sqrt(np.maximum(np.abs(dot),1e-6));gate=sigmoid(np.copysign(root,dot))
    return w,rh,rk,raw,dot,root,gate
def eloss():
    *_,gate=estate();return np.sum((x+gate[...,None]*value[:,None,:])*dy)
w,rh,rk,raw,dot,root,gate=estate();dg=np.sum(dy*value[:,None,:],axis=-1);ddot=dg*gate*(1-gate)/(2*root)*(np.abs(dot)>1e-6)
draw=ddot*rh*rk*h**-.5;drh=ddot*raw*rk*h**-.5;drk=ddot*raw*rh*h**-.5
dh=dy+draw[...,None]*w*key-(drh*rh**3/h)[...,None]*x
dk=draw[...,None]*x*w-(drk*rk**3/h)[...,None]*key
dw=np.sum(draw[...,None]*x*key,axis=0)
verify('Engram.full.dH',eloss,x,dh);verify('Engram.full.dKey',eloss,key,dk)
verify('Engram.full.dValue',eloss,value,np.sum(dy*gate[...,None],axis=1))
verify('Engram.full.dq_weight',eloss,qw,dw*kw);verify('Engram.full.dk_weight',eloss,kw,dw*qw)

# Indexer continuous score subgraph only; no pretend gradient through Top-K.
q=rng.uniform(.2,1,size=(2,3));keys=rng.uniform(.2,1,size=(4,3));w=rng.normal(size=2);g=rng.normal(size=4)
def iloss():return np.sum(np.sum(w[:,None]*np.maximum(q@keys.T,0),axis=0)*g)
z=q@keys.T;dz=w[:,None]*g*(z>0)
verify('Indexer.conditional.dQ',iloss,q,dz@keys)
verify('Indexer.conditional.dKey',iloss,keys,dz.T@q)
verify('Indexer.conditional.dHeadWeight',iloss,w,np.sum(np.maximum(z,0)*g,axis=1))

# Lookup repeats and real dequantization scales: scatter must SUM contributions.
table=rng.normal(size=(3,4));scales=rng.uniform(.5,1.5,size=(3,2));ids=np.array([0,2,0]);g=rng.normal(size=(3,4))
def lookup_loss():return np.sum(table[ids]*np.repeat(scales[ids],2,axis=1)*g)
dt=np.zeros_like(table);ds=np.zeros_like(scales)
np.add.at(dt,ids,np.repeat(scales[ids],2,axis=1)*g)
np.add.at(ds,ids,(table[ids]*g).reshape(3,2,2).sum(axis=2))
verify('Lookup.repeated_ids.dTable',lookup_loss,table,dt)
verify('Lookup.conditional.dScale',lookup_loss,scales,ds)

# Independent operation enumeration, including M=1 outer products.
for m,k,n in [(1,3,2),(2,3,2),(1,5120,512)]:
    f=L.linear_f('test',m,k,n).export()['totals']['main']; b=L.linear_b('test',m,k,n).export()
    assert f['mul']==m*k*n and f['add']==m*n*(k-1)
    assert b['totals']['main']['ordinary']==m*k*(2*n-1)+k*n*(2*m-1)
kv=L.linear_b('W_KV',1,5120,512).export()
assert kv['totals']['main']['ordinary']==7859200
assert kv['totals']['main']['approximate']==10485760
assert kv['steps'][1]['cost']['add']==0
assert pack['scenarios']['B']['length']==128 and pack['scenarios']['A']['prior']==1024
for fid,flow in pack['flows'].items():
    summed={'main':Counter(),'conditional':Counter()}
    for entry in flow['steps']:
        step=pack['steps'][entry['ref']];cost=step['cost'];acc=summed[step['path']]
        assert all(step[field] for field in ['purpose','inputs','outputs','formulas','work','why'])
        assert cost['ordinary']==cost['mul']+cost['add']+cost['sub']
        assert min(cost[k] for k in ['mul','add','sub','ordinary','approximate'])>=0
        for k in ['ordinary','approximate','mul','add','sub']:acc[k]+=cost[k]
        for k,v in cost['special'].items():acc['special:'+k]+=v
        assert all(entry['cumulative'][k]==v for k,v in acc.items())
    for path,t in flow['totals'].items():
        assert all(summed[path][k]==t[k] for k in ['ordinary','approximate','mul','add','sub'])
assert len(pack['modules'])==63
for mode in ['forward','backward']:
    assert pack['flows']['hc/B/'+mode]['totals']['main']['ordinary']>128*983040
assert any(pack['steps'][e['ref']]['status']=='Unknown' for e in pack['flows']['indexer/B/backward']['steps'])
assert pack['flows']['indexer/B/backward']['totals']['conditional']['ordinary']>0
assert pack['flows']['compressor/A/backward']['totals']['main']['ordinary']==0
assert pack['flows']['compressor/B/backward']['totals']['main']['ordinary']>0
index_scores=[pack['steps'][e['ref']] for e in pack['flows']['indexer/B/forward']['steps'] if '相似度点积' in pack['steps'][e['ref']]['title']]
assert index_scores[0]['cost']['ordinary']==128*32*128*(2*128-1)
merge=[pack['steps'][e['ref']] for e in pack['flows']['global/B/backward']['steps'] if '跨20个消费者层汇总' in pack['steps'][e['ref']]['title']]
assert merge[0]['cost']['add']==19*128*512
assert 'fp8_block_size = 32' in (ROOT/'references/deepseek/inference/model.py').read_text(encoding='utf-8')
# Original vector geometry is an immutable reference even though compute data changes.
import hashlib
assert hashlib.sha256((ROOT/'src/deepseek/architecture.svg').read_bytes()).hexdigest()=='8861c3b502047b9ba507645343bedf1494cc1719d10060c22ba67072542ec84c'
report=dict(passed=True,modules=len(pack['modules']),flows=len(pack['flows']),uniqueSteps=len(pack['steps']),finiteDifferenceMaxErrors=errors,outerProductStrict=7859200,outerProductApproximate=10485760,
  scope='Tiny real-valued graph tests, not official training or GPU kernels; fixed discrete choices; explicit scenario and special-op ledgers.')
(ROOT/'src/deepseek/qa/derivation_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
