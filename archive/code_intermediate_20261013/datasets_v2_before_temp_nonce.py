"""Validated data downloads and persistent, nested experiment partitions."""
import gzip
import hashlib
import json
import os
import struct
import urllib.request
import uuid
import zlib
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import eigsh


MNIST_FILES = {
    'train-images-idx3-ubyte.gz': ('images',60000),
    'train-labels-idx1-ubyte.gz': ('labels',60000),
    't10k-images-idx3-ubyte.gz': ('images',10000),
    't10k-labels-idx1-ubyte.gz': ('labels',10000),
}


def read_idx(path,kind,count):
    with gzip.open(path,'rb') as stream:
        data=stream.read()  # Includes gzip CRC validation.
    if kind=='images':
        if len(data)<16 or struct.unpack('>IIII',data[:16])!=(2051,count,28,28):
            raise ValueError(f'Invalid MNIST image header: {path.name}')
        if len(data)!=16+count*28*28:
            raise ValueError(f'Invalid MNIST image length: {path.name}')
        return np.frombuffer(data[16:],dtype=np.uint8).reshape(count,28,28)
    if len(data)<8 or struct.unpack('>II',data[:8])!=(2049,count):
        raise ValueError(f'Invalid MNIST label header: {path.name}')
    labels=np.frombuffer(data[8:],dtype=np.uint8)
    if len(labels)!=count or np.any(labels>9):
        raise ValueError(f'Invalid MNIST label payload: {path.name}')
    return labels


def download_mnist(cache):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    ledger=cache/'sha256.json'
    expected=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {}
    actual={}; arrays={}
    for name,(kind,count) in MNIST_FILES.items():
        path=cache/name
        try:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if name in expected and digest!=expected[name]:
                raise ValueError('Cached hash mismatch')
            array=read_idx(path,kind,count)
        except (OSError,EOFError,ValueError,zlib.error):
            temporary=cache/(name+'.'+uuid.uuid4().hex+'.part')
            try:
                url='https://storage.googleapis.com/cvdf-datasets/mnist/'+name
                print('Fetching validated MNIST file',name,flush=True)
                urllib.request.urlretrieve(url,temporary)
                digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
                if name in expected and digest!=expected[name]:
                    raise ValueError(f'Download differs from recorded hash: {name}')
                array=read_idx(temporary,kind,count)
                os.replace(temporary,path)
            finally:
                temporary.unlink(missing_ok=True)
        actual[name]=digest; arrays[name]=array
    temporary=ledger.with_name(ledger.name+'.'+uuid.uuid4().hex+'.part')
    temporary.write_text(json.dumps(actual,indent=2),encoding='utf-8')
    os.replace(temporary,ledger)
    def binary(prefix):
        images=arrays[prefix+'-images-idx3-ubyte.gz']
        labels=arrays[prefix+'-labels-idx1-ubyte.gz']
        keep=(labels==1)|(labels==9)
        return images[keep],(labels[keep]==9).astype(np.int64)
    return binary('train'),binary('t10k')


def fingerprint(*arrays):
    digest=hashlib.sha256()
    for array in arrays:
        array=np.ascontiguousarray(array)
        digest.update(str((array.shape,array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def map_input_bits(values,order):
    """Keep physical wire order; map Yao's least-significant input wire to wire 0."""
    if order=='msb':return values
    if order!='yao_lsb':raise ValueError('Unknown input_bit_order')
    width=values.shape[1]; bits=width.bit_length()-1
    if width!=1<<bits:raise ValueError('Amplitude dimension must be a power of two')
    permutation=np.array([int(f'{i:0{bits}b}'[::-1],2) for i in range(width)])
    return values[:,permutation]


def fixed_partitions(labels,train_capacity,validation_capacity,seed,path):
    """One master partition, independent of pilot sample counts and search seed."""
    labels=np.asarray(labels); source=fingerprint(labels)
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if (saved['labels_sha256']!=source or saved['split_seed']!=seed
                or saved['train_capacity']!=train_capacity
                or saved['validation_capacity']!=validation_capacity):
            raise ValueError('Master partition does not match its source/protocol')
    else:
        tr,va=train_test_split(np.arange(len(labels)),train_size=train_capacity,
                               test_size=validation_capacity,random_state=seed,stratify=labels)
        # Keep order permanently; pilots take prefixes, not freshly shuffled splits.
        saved=dict(protocol='fixed_v2',split_seed=seed,labels_sha256=source,
                   train_capacity=train_capacity,validation_capacity=validation_capacity,
                   train=tr.tolist(),validation=va.tolist())
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
        temporary.write_text(json.dumps(saved,indent=2),encoding='utf-8')
        os.replace(temporary,path)
    train=np.array(saved['train'],dtype=int); validation=np.array(saved['validation'],dtype=int)
    if len(set(train)&set(validation)):
        raise ValueError('Overlapping master training/validation partitions')
    return train,validation,fingerprint(train,validation)


def ground_states(lambdas,cluster_ising):
    h0,h1=cluster_ising(); states=[]; residuals=[]
    v0=np.random.default_rng(314159).normal(size=256)
    for i,lam in enumerate(lambdas):
        h=h0+lam*h1
        energy,vectors=eigsh(h,k=1,which='SA',v0=v0,tol=1e-11)
        state=vectors[:,0]
        states.append(state); residuals.append(float(np.linalg.norm(h@state-energy[0]*state)))
        if (i+1)%200==0:print(f'Ground states {i+1}/{len(lambdas)}',flush=True)
    return np.array(states),np.array(residuals)


def dataset_v2(config,root,normalize,cluster_ising,manifest=None):
    task=config['task']; seed=manifest.get('split_seed',config.get('split_seed',20261013)) if manifest else config.get('split_seed',20261013)
    root=Path(root); master_dir=root/'data'/'partitions_v2'
    restored=manifest is not None
    train_n=config['train_samples']; val_n=config['validation_samples']
    if task=='cancer':
        raw=load_breast_cancer(); features=raw.data.copy(); y=1-raw.target
        source=fingerprint(features,y)
        if restored:
            tr=np.array(manifest['train_indices']); va=np.array(manifest['validation_indices'])
            split_id=manifest.get('master_split_sha256','legacy')
        else:
            tr_pool,va_pool,split_id=fixed_partitions(y,400,169,seed,master_dir/f'cancer_{seed}.json')
            if train_n>len(tr_pool) or val_n>len(va_pool):raise ValueError('Sample count exceeds fixed cancer pool')
            tr,va=tr_pool[:train_n],va_pool[:val_n]
        transform=config.get('feature_transform','raw')
        meta=dict(protocol='fixed_v2',task=task,split_seed=seed,source_sha256=source,
                  master_split_sha256=split_id,train_indices=tr.tolist(),validation_indices=va.tolist(),
                  preprocessing=transform+' 30 features, zero-pad to 64, L2 normalize',
                  label_1='malignant',independent_test=False)
        if transform=='standard':
            mean=np.array(manifest['training_feature_mean']) if restored else features[tr].mean(0)
            scale=np.array(manifest['training_feature_std']) if restored else features[tr].std(0)
            features=(features-mean)/np.where(scale==0,1,scale)
            meta.update(training_feature_mean=mean.tolist(),training_feature_std=scale.tolist())
        elif transform=='minmax':
            minimum=np.array(manifest['training_feature_min']) if restored else features[tr].min(0)
            span=np.array(manifest['training_feature_span']) if restored else features[tr].max(0)-minimum
            features=(features-minimum)/np.where(span==0,1,span)
            meta.update(training_feature_min=minimum.tolist(),training_feature_span=span.tolist())
        elif transform!='raw':raise ValueError('Unknown cancer transform')
        x=normalize(features,64); train=(x[tr],y[tr]); validation=(x[va],y[va]); test=None
    elif task=='spt':
        if restored and manifest.get('protocol')!='fixed_v2':
            lambdas=np.array(manifest['lambdas']); cache=root/'data'/f"spt_{len(lambdas)}_seed{config['seed']}.npz"
        else:
            # Independent 500-state test extension is never used for architecture selection.
            lambdas=np.random.default_rng(seed).uniform(0,2,2500)
            cache=root/'data'/f'spt_v2_2500_split{seed}.npz'
        cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():
            with np.load(cache) as loaded:
                x,y,stored,residuals=[loaded[k] for k in ('x','y','lambdas','residuals')]
            if not np.array_equal(stored,lambdas):raise ValueError('SPT cache lambda mismatch')
        else:
            x,residuals=ground_states(lambdas,cluster_ising); y=(lambdas>1).astype(np.int64)
            temporary=cache.with_suffix('.part.npz')
            np.savez_compressed(temporary,x=x,y=y,lambdas=lambdas,residuals=residuals)
            os.replace(temporary,cache)
        if x.shape!=(len(lambdas),256) or not np.all(np.isfinite(x)):
            raise ValueError('Invalid SPT states')
        if not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-10) or max(residuals)>1e-8:
            raise ValueError('Invalid SPT norms or eigenstate residuals')
        source=fingerprint(x,y,lambdas)
        if restored:
            tr=np.array(manifest['train_indices']); va=np.array(manifest['validation_indices'])
            te=np.array(manifest.get('test_indices',[]),dtype=int)
            split_id=manifest.get('master_split_sha256','legacy')
        else:
            tr_pool,va_pool,split_id=fixed_partitions(y[:2000],1600,400,seed,master_dir/f'spt_{seed}.json')
            if train_n>1600 or val_n>400:raise ValueError('Sample count exceeds fixed SPT pool')
            tr,va=tr_pool[:train_n],va_pool[:val_n]; te=np.arange(2000,2500)[:config.get('test_samples',500)]
        meta=dict(protocol='fixed_v2',task=task,split_seed=seed,source_sha256=source,
                  master_split_sha256=split_id,train_indices=tr.tolist(),validation_indices=va.tolist(),
                  test_indices=te.tolist(),lambdas=lambdas.tolist(),label_1='lambda>1',
                  max_eigen_residual=float(max(residuals)),independent_test=bool(len(te)),
                  Hamiltonian='-sum(X[j-1] Z[j] X[j+1])+lambda sum(Y[j]Y[j+1])',boundary='periodic')
        train,validation=(x[tr],y[tr]),(x[va],y[va]); test=(x[te],y[te]) if len(te) else None
    elif task=='mnist':
        from PIL import Image
        (images,y),(test_images,test_y)=download_mnist(root/'data'/'mnist')
        source=fingerprint(images,y,test_images,test_y)
        if restored:
            tr=np.array(manifest['train_indices_in_binary_subset'])
            va=np.array(manifest['validation_indices_in_binary_subset'])
            te=np.array(manifest['test_indices_in_binary_subset'])
            split_id=manifest.get('master_split_sha256','legacy')
        else:
            tr_pool,va_pool,split_id=fixed_partitions(y,2000,500,seed,master_dir/f'mnist_{seed}.json')
            if train_n>2000 or val_n>500:raise ValueError('Sample count exceeds fixed MNIST pool')
            tr,va=tr_pool[:train_n],va_pool[:val_n]
            test_n=config.get('test_samples',500)
            if test_n==len(test_y):te=np.arange(len(test_y))
            else:
                te,_=train_test_split(np.arange(len(test_y)),train_size=test_n,
                                      random_state=seed+1,stratify=test_y)
        def encode(imgs):
            values=np.array([np.asarray(Image.fromarray(im).resize((16,16),Image.Resampling.BILINEAR),
                                       dtype=float).ravel() for im in imgs])
            return normalize(values,256)
        meta=dict(protocol='fixed_v2',task=task,split_seed=seed,source_sha256=source,
                  master_split_sha256=split_id,train_indices_in_binary_subset=tr.tolist(),
                  validation_indices_in_binary_subset=va.tolist(),test_indices_in_binary_subset=te.tolist(),
                  preprocessing='Pillow bilinear 28x28 -> 16x16, row-major, L2 normalize',
                  label_1='digit 9',independent_test=True)
        train,validation,test=(encode(images[tr]),y[tr]),(encode(images[va]),y[va]),(encode(test_images[te]),test_y[te])
    else:raise ValueError(task)
    order=manifest.get('input_bit_order','msb') if restored else config.get('input_bit_order','msb')
    meta['input_bit_order']=order
    train=(map_input_bits(train[0],order),train[1])
    validation=(map_input_bits(validation[0],order),validation[1])
    if test is not None:test=(map_input_bits(test[0],order),test[1])
    if restored and 'source_sha256' in manifest and source!=manifest['source_sha256']:
        raise ValueError('Source fingerprint differs from the saved experiment')
    if task=='mnist':tk,vk='train_indices_in_binary_subset','validation_indices_in_binary_subset'
    else:tk,vk='train_indices','validation_indices'
    if set(meta[tk])&set(meta[vk]):raise ValueError('Training/validation overlap')
    if task=='spt' and set(meta.get('test_indices',[]))&(set(meta[tk])|set(meta[vk])):
        raise ValueError('SPT independent test overlap')
    return train,validation,test,meta
