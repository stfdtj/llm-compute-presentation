"""Independent arithmetic/gradient checks; tiny real-valued examples, no model run."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import audit_model as model
from teaching_data import DATA,check
from audited_data import apply

check()
apply(DATA)
model.validate_sources()
def number(s): return int(s.replace(',','').split()[0])
def amount(key,mode,label): return number(dict(DATA[key][mode]['rows'])[label])
assert amount('kv','parameter','本模块存储')==2621440
assert amount('kv','forward','当前 token · 一次计算')==5242880
assert amount('kv','backward','本次反向合计')==10485760
assert dict(DATA['kv']['forward']['totals'])['当前 token 经过 40 层']=='209,715,200 FLOPs'
assert DATA['encoder']['_audit']['decode']+DATA['decoder']['_audit']['decode']+2*model.engram()+7*model.H+model.norm(model.H)+2*model.H*model.V==model.FORWARD

rng=np.random.default_rng(31)
results={}
def numeric(fn,x):
    dx=np.zeros_like(x);eps=1e-6
    for ix in np.ndindex(x.shape):
        v=x[ix];x[ix]=v+eps;a=fn();x[ix]=v-eps;b=fn();x[ix]=v;dx[ix]=(a-b)/(2*eps)
    return dx
def verify(name,fn,x,analytic):
    error=float(np.max(np.abs(numeric(fn,x)-analytic)))
    print(('PASS ' if error<3e-6 else 'FAIL ')+name+': maximum error '+str(error))
    assert error<3e-6,(name,error)
    results[name]=error

# Two low-rank factors with the intervening RMSNorm: reverse-order gradients.
x=rng.normal(size=(1,4));wa=rng.normal(size=(4,2));wb=rng.normal(size=(2,6));gamma=rng.normal(size=2);dy=rng.normal(size=(1,6))
def qloss():
    z=x@wa;r=(np.mean(z*z,axis=1,keepdims=True)+1e-4)**-.5
    return np.sum(((z*r*gamma)@wb)*dy)
z=x@wa;r=(np.mean(z*z,axis=1,keepdims=True)+1e-4)**-.5;u=z*r*gamma
du=dy@wb.T;g=du*gamma;dz=r*g-r**3/2*z*np.sum(g*z,axis=1,keepdims=True)
verify('low_rank_query.dX',qloss,x,dz@wa.T);verify('low_rank_query.dW_QA',qloss,wa,x.T@dz);verify('low_rank_query.dW_QB',qloss,wb,u.T@dy);verify('RMSNorm.dGamma',qloss,gamma,np.sum(du*z*r,axis=0))

# SwiGLU, with source-specific clamps and values safely away from boundaries.
g=np.array([[-12.,1.,12.],[2.,-1.,5.]]);u=np.array([[2.,12.,-12.],[1.,2.,3.]]);dz=rng.normal(size=(2,3))
def siluloss():
    gc=np.minimum(g,10);uc=np.clip(u,-10,10);return np.sum(dz*gc/(1+np.exp(-gc))*uc)
gc=np.minimum(g,10);uc=np.clip(u,-10,10);sig=1/(1+np.exp(-gc))
verify('SwiGLU.gate',siluloss,g,dz*uc*(sig+gc*sig*(1-sig))*(g<10));verify('SwiGLU.up',siluloss,u,dz*gc*sig*(np.abs(u)<10))

# Attention shares K and V and includes a denominator-only sink.
q=rng.normal(size=(2,3));kv=rng.normal(size=(4,3));sink=np.array([.2]);do=rng.normal(size=(2,3));scale=3**-.5
def attloss():
    scores=q@kv.T*scale;p=np.exp(scores)/(np.exp(scores).sum(axis=1,keepdims=True)+np.exp(sink))
    return np.sum((p@kv)*do)
scores=q@kv.T*scale;p=np.exp(scores)/(np.exp(scores).sum(axis=1,keepdims=True)+np.exp(sink));ps=np.exp(sink)/(np.exp(scores).sum(axis=1,keepdims=True)+np.exp(sink));dp=do@kv.T;ds=p*(dp-(p*dp).sum(axis=1,keepdims=True))
verify('shared_KV_attention.dQ',attloss,q,ds@kv*scale);verify('shared_KV_attention.dKV',attloss,kv,ds.T@q*scale+p.T@do);verify('shared_KV_attention.dSink',attloss,sink,np.array([-np.sum(ps*(p*dp).sum(axis=1,keepdims=True))]))

# Normalized router weights on a fixed selected set, no gradient through IDs.
s=rng.uniform(.1,1,(2,6));dw=rng.normal(size=(2,6));den=s.sum(axis=1,keepdims=True)+1e-20
def routeloss():return np.sum(1.5*s/(s.sum(axis=1,keepdims=True)+1e-20)*dw)
verify('router_selected_scores',routeloss,s,1.5/den*dw-1.5/den**2*np.sum(dw*s,axis=1,keepdims=True))

# HC post mixing: four copies, coefficients treated as intermediate inputs.
res=rng.normal(size=(4,3));u=rng.normal(size=3);post=rng.normal(size=4);comb=rng.normal(size=(4,4));dy=rng.normal(size=(4,3))
def hcloss():return np.sum((post[:,None]*u+comb.T@res)*dy)
verify('HC.residual',hcloss,res,comb@dy);verify('HC.sublayer',hcloss,u,np.sum(post[:,None]*dy,axis=0));verify('HC.post_coeff',hcloss,post,np.sum(dy*u,axis=1));verify('HC.comb_coeff',hcloss,comb,res@dy.T)
report={'passed':True,'fullLogicalParameters':model.TOTALS['All logical weights'],'textAndEngramParameters':model.TOTALS['Text + Engram'],'currentToken':{'tokensProcessed':1,'priorCachedTokens':1024,'position':1025,'modeledOperations':model.FORWARD},'sourceLedgerReconciled':True,'encoderDecoderTotalsReconciled':True,'finiteDifferenceMaxErrors':results,'limit':'Tiny real-valued mathematical tests only. No checkpoint, GPU kernel or official training/backward validation.'}
(ROOT/'src/deepseek/qa/calculation_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
