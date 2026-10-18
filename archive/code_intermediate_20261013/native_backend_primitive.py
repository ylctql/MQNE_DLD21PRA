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
            for kind in (0,1,0):
                kinds.append(kind); targets.append(target); controls.append(0); parameters.append(offset)
                offset+=1
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
        for b in range(batch):
            for i in range(dimension):
                if kinds[g]==0:
                    phase=complex(c,s if i&target else -s)
                    state[b,i]*=phase
                elif not i&target and (kinds[g]==1 or i&control):
                    j=i|target; a=state[b,i]; z=state[b,j]
                    state[b,i]=c*a-1j*s*z; state[b,j]=-1j*s*a+c*z
    result=np.zeros(batch)
    for b in range(batch):
        for i in range(1,dimension,2):result[b]+=abs(state[b,i])**2
    return result


@njit(cache=True)
def loss_gradient(theta,x,labels,kinds,targets,controls,parameters):
    batch=x.shape[0]; dimension=x.shape[1]*2; gates=len(kinds)
    states=np.zeros((gates+1,batch,dimension),dtype=np.complex128)
    states[0,:,::2]=x
    for g in range(gates):
        angle=theta[parameters[g]]; c=math.cos(angle/2); s=math.sin(angle/2)
        target=targets[g]; control=controls[g]
        states[g+1]=states[g]
        for b in range(batch):
            for i in range(dimension):
                if kinds[g]==0:
                    states[g+1,b,i]=states[g,b,i]*complex(c,s if i&target else -s)
                elif not i&target and (kinds[g]==1 or i&control):
                    j=i|target; a=states[g,b,i]; z=states[g,b,j]
                    states[g+1,b,i]=c*a-1j*s*z; states[g+1,b,j]=-1j*s*a+c*z
    adjoint=np.zeros((batch,dimension),dtype=np.complex128)
    loss=0.0
    for b in range(batch):
        probability=0.0
        for i in range(1,dimension,2):probability+=abs(states[gates,b,i])**2
        clipped=min(max(probability,1e-10),1-1e-10)
        loss-= (labels[b]*math.log(clipped)+(1-labels[b])*math.log(1-clipped))/batch
        derivative=(clipped-labels[b])/(clipped*(1-clipped)*batch)
        if probability<1e-10 or probability>1-1e-10:derivative=0.0
        for i in range(1,dimension,2):adjoint[b,i]=derivative*states[gates,b,i]
    gradient=np.zeros(len(theta))
    for g in range(gates-1,-1,-1):
        angle=theta[parameters[g]]; c=math.cos(angle/2); s=math.sin(angle/2)
        target=targets[g]; control=controls[g]; derivative=0.0
        for b in range(batch):
            for i in range(dimension):
                if kinds[g]==0:
                    sign=1.0 if i&target else -1.0
                    dstate=.5j*sign*states[g+1,b,i]
                    derivative+=2*(adjoint[b,i].conjugate()*dstate).real
                    adjoint[b,i]*=complex(c,-sign*s)
                elif not i&target and (kinds[g]==1 or i&control):
                    j=i|target; a=states[g,b,i]; z=states[g,b,j]
                    ga=adjoint[b,i]; gz=adjoint[b,j]
                    da=-.5*s*a-.5j*c*z; dz=-.5j*c*a-.5*s*z
                    derivative+=2*(ga.conjugate()*da+gz.conjugate()*dz).real
                    adjoint[b,i]=c*ga+1j*s*gz; adjoint[b,j]=1j*s*ga+c*gz
        gradient[parameters[g]]+=derivative
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
