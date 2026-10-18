"""Regression tests for review fixes and the numerically equivalent backend."""
import gzip
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import datasets_v2 as data
from mqne import Library,Circuit,train,dataset
from native_backend import operations,forward,loss_gradient,adam_epoch
from predict import verify_metrics


class ProtocolTests(unittest.TestCase):
    def test_yao_bit_mapping(self):
        # Little-endian |q3 q2 q1>=|0 0 1> maps to our |q1 q2 q3>=|1 0 0>.
        values=np.eye(8)[[1,2,3]]
        converted=data.map_input_bits(values,'yao_lsb')
        np.testing.assert_array_equal(converted,np.eye(8)[[4,2,6]])
        np.testing.assert_array_equal(data.map_input_bits(converted,'yao_lsb'),values)
        library=Library(4); model=Circuit(library,library.sample(np.random.default_rng(8),7),19)
        raw=np.arange(1,9,dtype=float)[None,:]; raw/=np.linalg.norm(raw)
        # Independent little-endian dense matrices: highest-index ancilla is |0>.
        state=np.pad(raw,((0,0),(0,8))).astype(complex); offset=0
        rz=lambda t:np.diag([np.exp(-.5j*t),np.exp(.5j*t)])
        rx=lambda t:np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
        def product(ops):
            matrix=np.array([[1]])
            for wire in (3,2,1,0):matrix=np.kron(matrix,ops.get(wire,np.eye(2)))
            return matrix
        theta=model.theta.detach().numpy()
        for block in model.blocks:
            for wire in block.rotations:
                a,b,c=theta[offset:offset+3]; offset+=3
                state=state@product({wire:rz(c)@rx(b)@rz(a)}).T
            for control,target in block.pairs:
                matrix=product({control:np.diag([1,0])})+product({control:np.diag([0,1]),target:rx(theta[offset])})
                offset+=1; state=state@matrix.T
        expected=np.abs(state[:,8:])**2
        actual=model(torch.tensor(data.map_input_bits(raw,'yao_lsb'),dtype=torch.float64)).detach()
        np.testing.assert_allclose(actual,expected.sum(1),atol=2e-14)

    def test_fixed_nested_partitions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            y=np.tile([0,1],50)
            tr,va,identity=data.fixed_partitions(y,60,20,44,root/'master.json')
            tr2,va2,identity2=data.fixed_partitions(y,60,20,44,root/'master.json')
            np.testing.assert_array_equal(tr,tr2); np.testing.assert_array_equal(va,va2)
            self.assertEqual(identity,identity2)
            self.assertFalse(set(tr[:10])&set(va))
            self.assertFalse(set(va[:5])&set(tr))

    def test_cancer_restore_uses_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            config=dict(task='cancer',seed=21,split_seed=44,train_samples=40,validation_samples=20,
                        feature_transform='standard')
            train1,validation1,_,manifest=dataset(config,root)
            changed=config|dict(seed=99,split_seed=888,train_samples=10,validation_samples=5)
            train2,validation2,_,_=dataset(changed,root,manifest)
            np.testing.assert_array_equal(train1[0],train2[0]); np.testing.assert_array_equal(validation1[0],validation2[0])
            full=config|dict(train_samples=400,validation_samples=169)
            _,_,_,full_manifest=dataset(full,root)
            self.assertFalse(set(manifest['train_indices'])&set(full_manifest['validation_indices']))

    def test_metrics_check_survives_optimization(self):
        with self.assertRaises(ValueError):
            verify_metrics(dict(loss=.3,accuracy=.7),dict(loss=.3,accuracy=0))
        with self.assertRaises(ValueError):
            verify_metrics(dict(loss=float('nan'),accuracy=.7),dict(loss=.3,accuracy=.7))
        with self.assertRaises(ValueError):
            verify_metrics(dict(loss=.3,accuracy=.7),dict(loss=float('nan'),accuracy=.7))

    def test_download_atomic_and_repairs_corruption(self):
        files={name:(kind,2) for name,(kind,count) in data.MNIST_FILES.items()}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); cache=root/'cache'; cache.mkdir()
            payloads={}
            for name,(kind,count) in files.items():
                raw=struct.pack('>IIII',2051,2,28,28)+np.ones((2,28,28),dtype=np.uint8).tobytes() if kind=='images' else struct.pack('>II',2049,2)+bytes([1,9])
                payloads[name]=gzip.compress(raw,mtime=0)
            def interrupted(url,path):
                Path(path).write_bytes(b'partial'); raise ConnectionError('Interrupted')
            with patch.dict(data.MNIST_FILES,files,clear=True),patch.object(data.urllib.request,'urlretrieve',side_effect=interrupted):
                with self.assertRaises(ConnectionError):data.download_mnist(cache)
            self.assertFalse((cache/'train-images-idx3-ubyte.gz').exists())
            self.assertFalse(list(cache.glob('*.part')))
            def complete(url,path):Path(path).write_bytes(payloads[url.rsplit('/',1)[-1]])
            with patch.dict(data.MNIST_FILES,files,clear=True),patch.object(data.urllib.request,'urlretrieve',side_effect=complete):
                data.download_mnist(cache)
                (cache/'train-images-idx3-ubyte.gz').write_bytes(b'bad cached data')
                with patch.object(data.urllib.request,'urlretrieve',side_effect=complete) as fetch:
                    data.download_mnist(cache)
                    self.assertEqual(fetch.call_count,1)
                self.assertEqual((cache/'train-images-idx3-ubyte.gz').read_bytes(),payloads['train-images-idx3-ubyte.gz'])


class BackendTests(unittest.TestCase):
    def test_native_probabilities_gradients_and_adam(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(17)
        library=Library(4); path=library.sample(rng,9); ops=operations(library,path)
        model=Circuit(library,path,37)
        x=rng.normal(size=(12,8)); x/=np.linalg.norm(x,axis=1,keepdims=True)
        y=rng.integers(0,2,size=12).astype(float)
        tensor=torch.tensor(x,dtype=torch.float64); labels=torch.tensor(y,dtype=torch.float64)
        probabilities=model(tensor)
        loss=torch.nn.functional.binary_cross_entropy(probabilities,labels)
        loss.backward(); theta=model.theta.detach().numpy().copy()
        native_loss,gradient=loss_gradient(theta,x,y,*ops)
        np.testing.assert_allclose(forward(theta,x,*ops),probabilities.detach(),atol=3e-14)
        np.testing.assert_allclose(native_loss,float(loss.detach()),atol=3e-14)
        np.testing.assert_allclose(gradient,model.theta.grad,atol=3e-12,rtol=3e-10)
        config=dict(device='cpu',learning_rate=.0015,epochs=3,batch_size=4,
                    evaluation_interval=1,fitness='combined',training_unit='epoch',progress=False)
        reference,record=train(library,path,config,(tensor,labels),(tensor,labels),37)
        native,nrecord=train(library,path,config|dict(backend='numba'),(tensor,labels),(tensor,labels),37)
        # Gauge directions have ~zero gradient; Adam's 1e-8 epsilon can amplify
        # float64 summation differences. Observables/gradients are checked tighter above.
        np.testing.assert_allclose(native.theta.detach(),reference.theta.detach(),atol=1e-9,rtol=1e-9)
        self.assertEqual(record['optimizer_updates'],nrecord['optimizer_updates'])
        self.assertAlmostEqual(record['final']['train']['loss'],nrecord['final']['train']['loss'],places=11)


if __name__=='__main__':unittest.main(verbosity=2)
