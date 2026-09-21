"""Bounded sparse-DAG Frobenius objective and unpenalized covariance QML. Local numerical solvers."""
import numpy as np
from scipy.optimize import minimize, lsq_linear
CAP,FROB,WMAX,TMIN,TMAX=.995,2.,2.,1e-4,5.
THRESHOLD=.001

def mask(order):
    pos=np.argsort(order);return pos[:,None]>pos[None,:]

def phi(B):
    P=np.linalg.inv(np.eye(len(B))-B);return P,P@P.T

def scales(S,L,B):
    F=phi(B)[1]
    return lsq_linear(np.column_stack([L.ravel(),F.ravel()]),S.ravel(),bounds=([0.,TMIN],[WMAX,TMAX]),tol=1e-10).x

def fro_objective(S,L,B,w,t):
    P,F=phi(B);R=w*L+t*F-S
    return np.sum(R*R),4*t*P.T@R@F

def proximal_fit(S,L,order,lam,B0=None,maxiter=180):
    allowed=mask(order);B=np.zeros_like(S) if B0 is None else B0.copy();B[~allowed]=0
    B*=min(1,.99*CAP/max(np.linalg.norm(B,2),1e-15),.99*FROB/max(np.linalg.norm(B),1e-15))
    w,t=scales(S,L,B);obj,grad=fro_objective(S,L,B,w,t);obj+=lam*np.sum(abs(B));step=.1;converged=False
    for it in range(maxiter):
        old=obj
        for bt in range(35):
            trial=np.sign(B-step*grad)*np.maximum(abs(B-step*grad)-step*lam,0);trial[~allowed]=0
            if np.linalg.norm(trial)>FROB or np.linalg.norm(trial,2)>CAP:
                step*=.5;continue
            val,gg=fro_objective(S,L,trial,w,t);val+=lam*np.sum(abs(trial))
            if val<=obj+1e-12:break
            step*=.5
        else:break
        delta=np.linalg.norm(trial-B);B=trial;w,t=scales(S,L,B)
        obj,grad=fro_objective(S,L,B,w,t);obj+=lam*np.sum(abs(B))
        if delta<1e-6 or (abs(obj-old)<1e-8*max(1,old) and delta<1e-4):converged=True;break
        step=min(step*1.2,.5)
    return dict(B=B,w=w,tau=t,objective=obj,iterations=it+1,converged=converged)

def qml_fit(S,L,support,init):
    ix=np.where(support);m=len(ix[0]);N=len(S)
    def unpack(x):
        B=np.zeros_like(S);B[ix]=x[:m];return B,x[-2],x[-1]
    def fg(x):
        B,w,t=unpack(x);P,F=phi(B);C=w*L+t*F
        Ci=np.linalg.inv(C);R=Ci-Ci@S@Ci
        val=np.linalg.slogdet(C)[1]+np.trace(S@Ci)
        return val,np.r_[(2*t*P.T@R@F)[ix],np.sum(R*L),np.sum(R*F)]
    def constr(x):
        B,_,_=unpack(x);return np.array([CAP-np.linalg.norm(B,2),FROB**2-np.sum(B*B)])
    def jac(x):
        B,_,_=unpack(x);U,_,VT=np.linalg.svd(B)
        return np.array([np.r_[-np.outer(U[:,0],VT[0])[ix],0,0],np.r_[-2*B[ix],0,0]])
    B=init['B'].copy();B[~support]=0;w,t=scales(S,L,B)
    start=np.r_[B[ix],w,t]
    # L-BFGS solves the same smooth support likelihood when norm bounds are inactive.
    ans=minimize(fg,start,jac=True,method='L-BFGS-B',bounds=[(-FROB,FROB)]*m+[(0,WMAX),(TMIN,TMAX)],options={'ftol':1e-12,'gtol':1e-7,'maxiter':350})
    if not ans.success or min(constr(ans.x)) < -1e-7:
        ans=minimize(fg,start,jac=True,method='SLSQP',bounds=[(-FROB,FROB)]*m+[(0,WMAX),(TMIN,TMAX)],constraints={'type':'ineq','fun':constr,'jac':jac},options={'ftol':1e-10,'maxiter':600})
    B,w,t=unpack(ans.x)
    return dict(B=B,w=float(w),tau=float(t),q=float(ans.fun),success=bool(ans.success),feasible=bool(min(constr(ans.x))>=-1e-7),iterations=int(ans.nit))



def order_candidates(S,L,seed,n_random):
    """Same random/insertion screening, using S_pi = C C' Cholesky."""
    N=len(S);rng=np.random.default_rng(seed)
    _,U=np.linalg.eigh(L);u=U[:,-1]
    top=float(u@S@u);candidates=[]
    for frac in [.2,.4,.6,.8,.95]:
        R=S-frac*top*np.outer(u,u)
        if np.linalg.eigvalsh(R).min()<=1e-8:continue
        orders=np.array([list(range(N))]+[rng.permutation(N).tolist() for _ in range(n_random)])
        ch=np.linalg.cholesky(R[orders[:,:,None],orders[:,None,:]])
        d=np.diagonal(ch,axis1=1,axis2=2)**2
        scores=d.mean(1)/np.exp(np.log(d).mean(1))
        ix=int(np.argmin(scores));o=orders[ix].tolist();g=float(scores[ix])
        for _ in range(60):
            trials=[]
            for a in range(N):
                for b in range(N):
                    if a!=b:
                        t=o.copy();t.insert(b,t.pop(a));trials.append(t)
            arr=np.array(trials)
            ch=np.linalg.cholesky(R[arr[:,:,None],arr[:,None,:]])
            d=np.diagonal(ch,axis1=1,axis2=2)**2
            values=d.mean(1)/np.exp(np.log(d).mean(1));j=int(np.argmin(values))
            if values[j]>=g-1e-12:break
            g=float(values[j]);o=trials[j]
        C=np.linalg.cholesky(R[np.ix_(o,o)]);T=C/np.diag(C)[None,:]
        B=np.zeros_like(S);B[np.ix_(o,o)]=np.eye(N)-np.linalg.inv(T)
        candidates.append((g,o,B))
    unique=[]
    for g,o,B in sorted(candidates,key=lambda x:x[0]):
        if o not in [x[1] for x in unique]:unique.append((g,o,B))
    if not unique:raise RuntimeError('No positive-definite screening covariance')
    return unique[:3]
