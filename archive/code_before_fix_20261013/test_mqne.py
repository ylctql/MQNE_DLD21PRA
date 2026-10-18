"""Physical and algorithmic cross-checks (not accuracy regression tests)."""
import math
import unittest

import numpy as np
import torch
from scipy.linalg import eigh

from mqne import Block, Circuit, Library, PAPER_PATHS, cluster_ising


class PhysicsTests(unittest.TestCase):
    def test_library_counts_and_start(self):
        libraries = {n:Library(n) for n in (7,9)}
        self.assertEqual(len(libraries[7].blocks),895)
        self.assertEqual(len(libraries[9].blocks),6687)
        for n,library in libraries.items():
            self.assertEqual(library.blocks[-1].rotations,tuple(range(n)))
            self.assertEqual(len(set(library.blocks)),len(library.blocks))
            rng=np.random.default_rng(41)
            for _ in range(20):
                path=library.sample(rng,15)
                self.assertTrue(all(library.allowed(library.blocks[a],library.blocks[b])
                                    for a,b in zip(path,path[1:])))

    def test_successors_against_scalar_rules(self):
        library = Library(4)
        for i,a in enumerate(library.blocks):
            expected = [j for j,b in enumerate(library.blocks) if library.allowed(a,b)]
            self.assertEqual(library.successors(i).tolist(),expected)

    def test_gates_unitarity_and_control(self):
        library = Library(2)
        crx = library.blocks.index(Block(((0,1),),()))
        circuit = Circuit(library,[crx])
        with torch.no_grad():
            circuit.theta.fill_(math.pi)
        inputs = torch.eye(2,dtype=torch.float64)
        states = circuit.state(inputs).detach().numpy()
        np.testing.assert_allclose(states,[[1,0,0,0],[0,0,0,-1j]],atol=1e-14)
        rng = np.random.default_rng(1)
        library = Library(4)
        circuit = Circuit(library,library.sample(rng,7),seed=23)
        with torch.no_grad():
            circuit.theta.copy_(torch.randn_like(circuit.theta))
        inputs = torch.tensor(rng.normal(size=(4,8)),dtype=torch.float64)
        inputs /= inputs.norm(dim=1,keepdim=True)
        np.testing.assert_allclose(circuit.state(inputs).abs().square().sum(1).detach(),1,atol=2e-14)
        # Compare to an independently assembled dense Kronecker-product circuit.
        state = np.stack((inputs.numpy(),np.zeros_like(inputs.numpy())),axis=-1).reshape(4,16)
        theta = circuit.theta.detach().numpy(); offset=0
        rz = lambda t:np.diag([np.exp(-.5j*t),np.exp(.5j*t)])
        rx = lambda t:np.array([[np.cos(t/2),-1j*np.sin(t/2)],[-1j*np.sin(t/2),np.cos(t/2)]])
        def product(ops):
            result = np.array([[1]])
            for q in range(4): result = np.kron(result,ops.get(q,np.eye(2)))
            return result
        for block in circuit.blocks:
            for q in block.rotations:
                a,b,c=theta[offset:offset+3]; offset+=3
                state = state@product({q:rz(c)@rx(b)@rz(a)}).T
            for control,target in block.pairs:
                gate = product({control:np.diag([1,0])})+product({control:np.diag([0,1]),target:rx(theta[offset])})
                offset+=1; state=state@gate.T
        np.testing.assert_allclose(circuit.state(inputs).detach().numpy(),state,atol=2e-14)

    def test_autograd_finite_difference(self):
        library = Library(3)
        path = library.sample(np.random.default_rng(4),5)
        circuit = Circuit(library,path,seed=5)
        x = torch.tensor([[1,2,3,4]],dtype=torch.float64)
        x /= x.norm(dim=1,keepdim=True)
        with torch.no_grad(): circuit.theta.add_(.17)
        circuit(x).sum().backward()
        gradient = circuit.theta.grad.detach().numpy().copy()
        for i in range(len(gradient)):
            with torch.no_grad():
                original=float(circuit.theta[i]); h=1e-6
                circuit.theta[i]=original+h; plus=float(circuit(x))
                circuit.theta[i]=original-h; minus=float(circuit(x))
                circuit.theta[i]=original
            self.assertAlmostEqual(gradient[i],(plus-minus)/(2*h),places=7)

    def test_cluster_ising_dense_spectrum(self):
        # Independent Kronecker construction, including periodic wraparound and signs.
        n=4
        pauli={'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
        def term(ops):
            result=np.array([[1]])
            for j in range(n): result=np.kron(result,pauli[ops[j]] if j in ops else np.eye(2))
            return result
        dense=sum(-term({(j-1)%n:'X',j:'Z',(j+1)%n:'X'})
                  +.7*term({j:'Y',(j+1)%n:'Y'}) for j in range(n))
        a,b=cluster_ising(n)
        np.testing.assert_allclose((a+.7*b).toarray(),dense,atol=1e-14)
        a,b=cluster_ising(8)
        eigenvalues=eigh(a.toarray(),eigvals_only=True)
        self.assertAlmostEqual(eigenvalues[0],-8,places=10)

    def test_zero_extension_preserves_parent(self):
        library=Library(4); rng=np.random.default_rng(20)
        parent=library.sample(rng,4); child=library.sample(rng,2,parent)
        a,b=Circuit(library,parent,31),Circuit(library,child,31)
        count=len(a.theta)
        with torch.no_grad():
            b.theta[:count].copy_(a.theta); b.theta[count:]=0
        x=torch.tensor([[1,2,3,4,5,6,7,8]],dtype=torch.float64)
        x/=x.norm(dim=1,keepdim=True)
        torch.testing.assert_close(a.state(x),b.state(x),atol=1e-14,rtol=1e-14)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
