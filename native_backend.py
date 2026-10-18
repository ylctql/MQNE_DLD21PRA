"""Float64/complex128 adjoint simulator, cross-checked against PyTorch autograd.

This changes execution cost, not the model, loss, batches, or optimizer.
"""
import math
import numpy as np
from numba import njit


def operations(library,path):
    kinds=[]; targets=[]; controls=[]; parameters=[]; offset=0
    for index in path:
        block=library.blocks[index]
        for wire in block.rotations:
            target=1<<(library.qubits-1-wire)
            kinds.append(3); targets.append(target); controls.append(0); parameters.append(offset)
            offset+=3
        for control,target in block.pairs:
            kinds.append(2); targets.append(1<<(library.qubits-1-target))
            controls.append(1<<(library.qubits-1-control)); parameters.append(offset); offset+=1
    return tuple(np.array(x,dtype=np.int64) for x in (kinds,targets,controls,parameters))


@njit(cache=True)
def forward(theta,x,kinds,targets,controls,parameters):
    batch=x.shape[0]; dimension=x.shape[1]*2
    state=np.zeros((batch,dimension),dtype=np.complex128)
    state[:,::2]=x
    for g in range(len(kinds)):
        angle=theta[parameters[g]]; c=math.cos(angle/2); s=math.sin(angle/2)
        target=targets[g]; control=controls[g]
        if kinds[g]==3:
            alpha,beta,gamma=theta[parameters[g]:parameters[g]+3]
            cb=math.cos(beta/2); sb=math.sin(beta/2)
            u00=cb*np.exp(-.5j*(alpha+gamma)); u01=-1j*sb*np.exp(.5j*(alpha-gamma))
            u10=-1j*sb*np.exp(-.5j*(alpha-gamma)); u11=cb*np.exp(.5j*(alpha+gamma))
        for b in range(batch):
            for i in range(dimension):
                if kinds[g]==3 and not i&target:
                    j=i|target; a=state[b,i]; z=state[b,j]
                    state[b,i]=u00*a+u01*z; state[b,j]=u10*a+u11*z
                elif kinds[g]==0:
                    phase=complex(c,s if i&target else -s)
                    state[b,i]*=phase
                elif kinds[g]!=3 and not i&target and (kinds[g]==1 or i&control):
                    j=i|target; a=state[b,i]; z=state[b,j]
                    state[b,i]=c*a-1j*s*z; state[b,j]=-1j*s*a+c*z
    result=np.zeros(batch)
    for b in range(batch):
        for i in range(1,dimension,2):result[b]+=abs(state[b,i])**2
    return result


@njit(cache=True)
def loss_gradient(theta,x,labels,kinds,targets,controls,parameters):
    batch=x.shape[0]; dimension=x.shape[1]*2; gates=len(kinds)
    state=np.zeros((batch,dimension),dtype=np.complex128)
    state[:,::2]=x
    for g in range(gates):
        angle=theta[parameters[g]]; c=math.cos(angle/2); s=math.sin(angle/2)
        target=targets[g]; control=controls[g]
        if kinds[g]==3:
            alpha,beta,gamma=theta[parameters[g]:parameters[g]+3]
            cb=math.cos(beta/2); sb=math.sin(beta/2)
            u00=cb*np.exp(-.5j*(alpha+gamma)); u01=-1j*sb*np.exp(.5j*(alpha-gamma))
            u10=-1j*sb*np.exp(-.5j*(alpha-gamma)); u11=cb*np.exp(.5j*(alpha+gamma))
        for b in range(batch):
            for i in range(dimension):
                if kinds[g]==3 and not i&target:
                    j=i|target; a=state[b,i]; z=state[b,j]
                    state[b,i]=u00*a+u01*z; state[b,j]=u10*a+u11*z
                elif kinds[g]==0:
                    state[b,i]*=complex(c,s if i&target else -s)
                elif kinds[g]!=3 and not i&target and (kinds[g]==1 or i&control):
                    j=i|target; a=state[b,i]; z=state[b,j]
                    state[b,i]=c*a-1j*s*z; state[b,j]=-1j*s*a+c*z
    adjoint=np.zeros((batch,dimension),dtype=np.complex128)
    loss=0.0
    for b in range(batch):
        probability=0.0
        for i in range(1,dimension,2):probability+=abs(state[b,i])**2
        clipped=min(max(probability,1e-10),1-1e-10)
        loss-= (labels[b]*math.log(clipped)+(1-labels[b])*math.log(1-clipped))/batch
        derivative=(clipped-labels[b])/(clipped*(1-clipped)*batch)
        if probability<1e-10 or probability>1-1e-10:derivative=0.0
        for i in range(1,dimension,2):adjoint[b,i]=derivative*state[b,i]
    gradient=np.zeros(len(theta))
    for g in range(gates-1,-1,-1):
        angle=theta[parameters[g]]; c=math.cos(angle/2); s=math.sin(angle/2)
        target=targets[g]; control=controls[g]; derivative=0.0
        da_total=0.0; db_total=0.0; dg_total=0.0
        if kinds[g]==3:
            alpha,beta,gamma=theta[parameters[g]:parameters[g]+3]
            cb=math.cos(beta/2); sb=math.sin(beta/2)
            u00=cb*np.exp(-.5j*(alpha+gamma)); u01=-1j*sb*np.exp(.5j*(alpha-gamma))
            u10=-1j*sb*np.exp(-.5j*(alpha-gamma)); u11=cb*np.exp(.5j*(alpha+gamma))
            eg=np.exp(1j*gamma)
        for b in range(batch):
            for i in range(dimension):
                if kinds[g]==3 and not i&target:
                    j=i|target; outa=state[b,i]; outb=state[b,j]
                    ga=adjoint[b,i]; gb=adjoint[b,j]
                    a=u00.conjugate()*outa+u10.conjugate()*outb
                    z=u01.conjugate()*outa+u11.conjugate()*outb
                    da0=-.5j*u00*a+.5j*u01*z; da1=-.5j*u10*a+.5j*u11*z
                    db0=-.5j*eg.conjugate()*outb; db1=-.5j*eg*outa
                    dg0=-.5j*outa; dg1=.5j*outb
                    da_total+=2*(ga.conjugate()*da0+gb.conjugate()*da1).real
                    db_total+=2*(ga.conjugate()*db0+gb.conjugate()*db1).real
                    dg_total+=2*(ga.conjugate()*dg0+gb.conjugate()*dg1).real
                    state[b,i]=a; state[b,j]=z
                    adjoint[b,i]=u00.conjugate()*ga+u10.conjugate()*gb
                    adjoint[b,j]=u01.conjugate()*ga+u11.conjugate()*gb
                elif kinds[g]==0:
                    sign=1.0 if i&target else -1.0
                    dstate=.5j*sign*state[b,i]
                    derivative+=2*(adjoint[b,i].conjugate()*dstate).real
                    adjoint[b,i]*=complex(c,-sign*s)
                    state[b,i]*=complex(c,-sign*s)
                elif kinds[g]!=3 and not i&target and (kinds[g]==1 or i&control):
                    j=i|target; a=state[b,i]; z=state[b,j]
                    ga=adjoint[b,i]; gz=adjoint[b,j]
                    da=-.5j*z; dz=-.5j*a
                    derivative+=2*(ga.conjugate()*da+gz.conjugate()*dz).real
                    state[b,i]=c*a+1j*s*z; state[b,j]=1j*s*a+c*z
                    adjoint[b,i]=c*ga+1j*s*gz; adjoint[b,j]=1j*s*ga+c*gz
        if kinds[g]==3:
            gradient[parameters[g]]=da_total; gradient[parameters[g]+1]=db_total; gradient[parameters[g]+2]=dg_total
        else:gradient[parameters[g]]+=derivative
    return loss,gradient


@njit(cache=True)
def adam_epoch(theta,x,y,batch_order,batch_size,first,second,step,rate,ops):
    kinds,targets,controls,parameters=ops
    for start in range(0,len(batch_order),batch_size):
        indexes=batch_order[start:start+batch_size]
        loss,gradient=loss_gradient(theta,x[indexes],y[indexes],kinds,targets,controls,parameters)
        step+=1
        for i in range(len(theta)):
            first[i]=.9*first[i]+.1*gradient[i]
            second[i]=.999*second[i]+.001*gradient[i]**2
            theta[i]-=rate*(first[i]/(1-.9**step))/(math.sqrt(second[i]/(1-.999**step))+1e-8)
    return step


def metrics(theta,data,ops,batch=256):
    x,y=data
    probabilities=np.concatenate([forward(theta,x[i:i+batch],*ops) for i in range(0,len(y),batch)])
    clipped=np.clip(probabilities,1e-10,1-1e-10)
    return {'loss':float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean()),
            'accuracy':float(((probabilities>=.5)==y.astype(bool)).mean())}


def train(library,path,config,train_data,val_data,seed):
    import time
    import torch
    from mqne import Circuit
    # Same initial random stream and parameter layout as the reference backend.
    model=Circuit(library,path,seed,crx_initialization=config.get('crx_initialization','random'))
    theta=model.theta.detach().numpy().copy(); ops=operations(library,path)
    train_data=tuple(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for v in train_data)
    val_data=tuple(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for v in val_data)
    x,y=train_data; first=np.zeros_like(theta); second=np.zeros_like(theta); updates=0
    generator=torch.Generator().manual_seed(seed+100000); started=time.perf_counter()
    history=[dict(epoch=0,train=metrics(theta,train_data,ops),validation=metrics(theta,val_data,ops))]
    for epoch in range(1,config['epochs']+1):
        order=torch.randperm(len(y),generator=generator).numpy()
        if config.get('training_unit','epoch')=='step':order=order[:config['batch_size']]
        updates=adam_epoch(theta,x,y,order,config['batch_size'],first,second,updates,config['learning_rate'],ops)
        if epoch%config.get('evaluation_interval',10)==0 or epoch==config['epochs']:
            history.append(dict(epoch=epoch,train=metrics(theta,train_data,ops),validation=metrics(theta,val_data,ops)))
            if config.get('progress',True):
                print(f"  epoch {epoch}: train {history[-1]['train']['accuracy']:.4f}, validation {history[-1]['validation']['accuracy']:.4f}",flush=True)
    final=history[-1]
    fitness=(final['train']['accuracy']*len(y)+final['validation']['accuracy']*len(val_data[1]))/(len(y)+len(val_data[1]))
    if config.get('fitness','combined')=='validation':fitness=final['validation']['accuracy']
    with torch.no_grad():model.theta.copy_(torch.tensor(theta,dtype=torch.float64))
    record=dict(fitness=float(fitness),final=final,history=history,training_seed=seed,
                seconds=time.perf_counter()-started,optimizer_updates=int(updates),
                path_1based=[i+1 for i in path],backend='numba_adjoint',**library.stats(path))
    return model,record
