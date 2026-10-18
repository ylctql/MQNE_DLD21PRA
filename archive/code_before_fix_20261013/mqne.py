"""Independent MQNE reproduction of Lu, Shen & Deng, PRApplied 16 044039.

Wire 0 is the most significant bit. Last wire is a |0> readout ancilla.
Gate-block order uses a lexicographic convention inspired by the author's
public notebook; published numbered paths are not assumed to be compatible.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import platform
import struct
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Block:
    pairs: tuple[tuple[int, int], ...]
    rotations: tuple[int, ...]

    @property
    def parameters(self):
        return len(self.pairs) + 3 * len(self.rotations)

    @property
    def support(self):
        return set(self.rotations) | {q for pair in self.pairs for q in pair}


class Library:
    def __init__(self, qubits):
        self.qubits = qubits
        self.blocks = []
        # Lexicographic prefixes of permutations, sorted by controlling wire.
        directed = [(a, b) for a in range(qubits) for b in range(qubits) if abs(a-b) == 1]
        for count in range(qubits // 2, -1, -1):
            matchings = [pairs for pairs in itertools.combinations(directed, count)
                         if len({q for p in pairs for q in p}) == 2 * count]
            matchings.sort(key=lambda pairs: tuple(q for p in pairs for q in p))
            for pairs in matchings:
                used = {q for pair in pairs for q in pair}
                free = [q for q in range(qubits) if q not in used]
                for bits in itertools.product((0, 1), repeat=len(free)):
                    rotations = tuple(q for q, bit in zip(free, bits) if bit)
                    if pairs or rotations:
                        self.blocks.append(Block(tuple(pairs), rotations))
        self._successors = {}
        self._supports = np.array([sum(1 << q for q in b.support) for b in self.blocks])
        self._rotations = np.array([sum(1 << q for q in b.rotations) for b in self.blocks])
        self._pair_bits = np.array([sum(1 << (2*min(a,b)+(a>b)) for a,b in b.pairs)
                                   for b in self.blocks])
        self._pair_supports = np.array([sum(1 << q for p in b.pairs for q in p)
                                       for b in self.blocks])

    @staticmethod
    def allowed(a, b):
        # Every new gate overlaps the previous block; no mergeable repetition.
        pair_support = {q for p in a.pairs for q in p}
        return (all(q in pair_support for q in b.rotations)
                and not set(a.pairs).intersection(b.pairs)
                and all(set(p).intersection(a.support) for p in b.pairs))

    def successors(self, index):
        if index not in self._successors:
            mask = (self._rotations & ~self._pair_supports[index]) == 0
            mask &= (self._pair_bits & self._pair_bits[index]) == 0
            for a in range(self.qubits-1):
                if not (self._supports[index] & ((1 << a) | (1 << (a+1)))):
                    mask &= (self._pair_bits & (3 << (2*a))) == 0
            self._successors[index] = np.flatnonzero(mask)
        return self._successors[index]

    def sample(self, rng, length, prefix=None):
        path = list(prefix) if prefix else [len(self.blocks)-1]
        target = len(path)+length if prefix else length
        while len(path) < target:
            choices = self.successors(path[-1])
            if not len(choices):
                raise RuntimeError('Dead end in gate-block graph')
            path.append(int(rng.choice(choices)))
        return path

    def stats(self, path):
        blocks = [self.blocks[i] for i in path]
        nr = sum(len(b.rotations) for b in blocks)
        nc = sum(len(b.pairs) for b in blocks)
        return dict(block_depth=len(path), R_gates=nr, elementary_single_qubit_gates=3*nr,
                    CRx_gates=nc, parameters=3*nr+nc)


class Circuit(torch.nn.Module):
    def __init__(self, library, path, seed=0, device='cpu', crx_initialization='random'):
        super().__init__()
        self.qubits = library.qubits
        self.path = path
        self.blocks = [library.blocks[i] for i in path]
        generator = torch.Generator().manual_seed(seed)
        values = torch.rand(sum(b.parameters for b in self.blocks), generator=generator,
                            dtype=torch.float64) * 2 * math.pi
        self.theta = torch.nn.Parameter(values.to(device))
        self.ops = []
        offset = 0
        indexes = np.arange(1 << self.qubits)
        for b in self.blocks:
            for wire in b.rotations:
                mask = 1 << (self.qubits-1-wire)
                lo = indexes[(indexes & mask) == 0]
                self.ops.append(('R', offset, torch.tensor(lo, device=device),
                                 torch.tensor(lo | mask, device=device)))
                offset += 3
            for control, target in b.pairs:
                cm = 1 << (self.qubits-1-control)
                tm = 1 << (self.qubits-1-target)
                lo = indexes[((indexes & tm) == 0) & ((indexes & cm) != 0)]
                self.ops.append(('CRx', offset, torch.tensor(lo, device=device),
                                 torch.tensor(lo | tm, device=device)))
                if crx_initialization == 'zero':
                    with torch.no_grad():
                        self.theta[offset] = 0
                offset += 1

    def state(self, inputs):
        state = torch.stack((inputs, torch.zeros_like(inputs)), dim=-1).flatten(1)
        state = state.to(torch.complex128)
        for kind, offset, lo, hi in self.ops:
            a, b = state[:, lo], state[:, hi]
            if kind == 'R':
                # Chronological Rz(alpha), Rx(beta), Rz(gamma).
                alpha, beta, gamma = self.theta[offset:offset+3]
                a = a * torch.exp(-.5j*alpha)
                b = b * torch.exp(.5j*alpha)
                c, s = torch.cos(beta/2), -1j*torch.sin(beta/2)
                out_a = (c*a+s*b)*torch.exp(-.5j*gamma)
                out_b = (s*a+c*b)*torch.exp(.5j*gamma)
            else:
                angle = self.theta[offset]
                c, s = torch.cos(angle/2), -1j*torch.sin(angle/2)
                out_a, out_b = c*a+s*b, s*a+c*b
            new = state.clone()
            new[:, lo], new[:, hi] = out_a, out_b
            state = new
        return state

    def forward(self, inputs):
        return self.state(inputs)[:, 1::2].abs().square().sum(dim=1)


def normalize(data, dimension):
    data = np.pad(data, ((0, 0), (0, dimension-data.shape[1])))
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Cannot amplitude-encode a zero vector')
    return data / norms


def pauli_term(n, operations):
    indices = np.arange(1 << n)
    mapped = indices.copy()
    phase = np.ones(len(indices), dtype=np.complex128)
    for wire, kind in operations:
        bit = (indices >> (n-1-wire)) & 1
        if kind in ('X', 'Y'):
            mapped ^= 1 << (n-1-wire)
        if kind == 'Y':
            phase *= 1j*(1-2*bit)
        elif kind == 'Z':
            phase *= 1-2*bit
    return sparse.csr_matrix((phase, (mapped, indices)), shape=(1 << n, 1 << n))


def cluster_ising(n=8):
    cluster = sparse.csr_matrix((1 << n, 1 << n), dtype=np.complex128)
    ising = cluster.copy()
    for j in range(n):
        cluster -= pauli_term(n, [((j-1)%n, 'X'), (j, 'Z'), ((j+1)%n, 'X')])
        ising += pauli_term(n, [(j, 'Y'), ((j+1)%n, 'Y')])
    return cluster.real, ising.real


def download_mnist(cache):
    names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in names:
        path = cache/name
        if not path.exists():
            url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'+name
            print('Downloading', url, flush=True)
            urllib.request.urlretrieve(url, path)
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (cache/'sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    def read(prefix):
        with gzip.open(cache/(prefix+'-images-idx3-ubyte.gz'), 'rb') as f:
            magic, count, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051 and (rows, cols) == (28, 28)
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
        with gzip.open(cache/(prefix+'-labels-idx1-ubyte.gz'), 'rb') as f:
            magic, count = struct.unpack('>II', f.read(8))
            assert magic == 2049
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        selected = (labels == 1) | (labels == 9)
        return images[selected], (labels[selected] == 9).astype(np.int64)
    return read('train'), read('t10k')


def dataset(config, root):
    task, seed = config['task'], config['seed']
    train_n, val_n = config['train_samples'], config['validation_samples']
    if task == 'cancer':
        raw = load_breast_cancer()
        y = 1-raw.target  # malignant=1
        tr, va = train_test_split(np.arange(len(y)), train_size=train_n,
                                  test_size=val_n, random_state=seed, stratify=y)
        features=raw.data.copy()
        transform=config.get('feature_transform','raw')
        if transform=='standard':
            mean=features[tr].mean(axis=0)
            scale=features[tr].std(axis=0)
            features=(features-mean)/np.where(scale==0,1,scale)
        elif transform!='raw':
            raise ValueError('Unsupported feature_transform')
        x = normalize(features, 64)
        meta = {'preprocessing': transform+' 30 features, zero-pad to 64, L2 normalize',
                'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                'label_1': 'malignant', 'independent_test': False}
        if transform=='standard':
            meta.update(training_feature_mean=mean.tolist(),training_feature_std=scale.tolist())
        return (x[tr], y[tr]), (x[va], y[va]), None, meta
    if task == 'spt':
        cache = root/'data'/f'spt_{train_n+val_n}_seed{seed}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            loaded = np.load(cache)
            x, y, lambdas, residuals = [loaded[k] for k in ('x','y','lambdas','residuals')]
        else:
            rng = np.random.default_rng(seed)
            lambdas = rng.uniform(0, 2, train_n+val_n)
            h0, h1 = cluster_ising()
            x, residuals = [], []
            # Deterministic solver start; finite-size degenerate states are not specified in SI.
            v0 = np.random.default_rng(314159).normal(size=256)
            for i, lam in enumerate(lambdas):
                h = h0+lam*h1
                eigenvalues, vectors = eigsh(h, k=1, which='SA', v0=v0, tol=1e-11)
                state = vectors[:, 0]
                residuals.append(float(np.linalg.norm(h@state-eigenvalues[0]*state)))
                x.append(state)
                if (i+1) % 200 == 0:
                    print(f'Ground states {i+1}/{len(lambdas)}', flush=True)
            x = np.array(x)
            residuals = np.array(residuals)
            y = (lambdas > 1).astype(np.int64)
            np.savez_compressed(cache, x=x, y=y, lambdas=lambdas, residuals=residuals)
        tr, va = train_test_split(np.arange(len(y)), train_size=train_n, test_size=val_n,
                                  random_state=seed, stratify=y)
        meta = {'Hamiltonian': '-sum(X[j-1] Z[j] X[j+1])+lambda sum(Y[j]Y[j+1])',
                'boundary': 'periodic', 'max_eigen_residual': float(max(residuals)),
                'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                'lambdas': lambdas.tolist(), 'label_1': 'lambda>1', 'independent_test': False}
        return (x[tr], y[tr]), (x[va], y[va]), None, meta
    if task == 'mnist':
        from PIL import Image
        (images, y), (test_images, test_y) = download_mnist(root/'data'/'mnist')
        selected, _ = train_test_split(np.arange(len(y)), train_size=train_n+val_n,
                                       random_state=seed, stratify=y)
        tr, va = train_test_split(selected, train_size=train_n, test_size=val_n,
                                  random_state=seed, stratify=y[selected])
        def encode(imgs):
            values = np.array([np.asarray(Image.fromarray(im).resize((16,16),
                              Image.Resampling.BILINEAR), dtype=float).ravel() for im in imgs])
            return normalize(values, 256)
        test_n = config.get('test_samples', 500)
        te, _ = train_test_split(np.arange(len(test_y)), train_size=test_n,
                                 random_state=seed, stratify=test_y)
        meta = {'preprocessing': 'Pillow bilinear 28x28 -> 16x16, row-major, L2 normalize',
                'train_indices_in_binary_subset': tr.tolist(),
                'validation_indices_in_binary_subset': va.tolist(),
                'test_indices_in_binary_subset': te.tolist(), 'label_1': 'digit 9',
                'independent_test': True}
        return (encode(images[tr]),y[tr]), (encode(images[va]),y[va]), (encode(test_images[te]),test_y[te]), meta
    raise ValueError(task)


def metrics(model, inputs, labels, batch=256):
    with torch.no_grad():
        probabilities = torch.cat([model(x) for x in inputs.split(batch)]).clamp(1e-10, 1-1e-10)
        loss = torch.nn.functional.binary_cross_entropy(probabilities, labels)
        accuracy = ((probabilities >= .5) == labels.bool()).double().mean()
    return {'loss': float(loss), 'accuracy': float(accuracy)}


def train(library, path, config, train_data, val_data, seed):
    model = Circuit(library, path, seed, device=config['device'],
                    crx_initialization=config.get('crx_initialization','random'))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    history = []
    generator = torch.Generator().manual_seed(seed+100000)
    started = time.perf_counter()
    updates = 0
    x, y = train_data
    history.append(dict(epoch=0, train=metrics(model,x,y), validation=metrics(model,*val_data)))
    for epoch in range(1, config['epochs']+1):
        if config.get('training_unit', 'epoch') == 'step':
            batches = [torch.randperm(len(y),generator=generator)[:config['batch_size']]]
        else:
            batches = torch.randperm(len(y), generator=generator).split(config['batch_size'])
        for indices in batches:
            optimizer.zero_grad()
            probabilities = model(x[indices]).clamp(1e-10,1-1e-10)
            loss = torch.nn.functional.binary_cross_entropy(probabilities,y[indices])
            loss.backward()
            optimizer.step()
            updates += 1
        if epoch % config.get('evaluation_interval', 10) == 0 or epoch == config['epochs']:
            history.append(dict(epoch=epoch, train=metrics(model,x,y),
                                validation=metrics(model,*val_data)))
    final = history[-1]
    # SI says fitness averaged on training and validation samples.
    fitness = ((final['train']['accuracy']*len(y) + final['validation']['accuracy']*len(val_data[1]))
               / (len(y)+len(val_data[1])))
    if config.get('fitness','combined') == 'validation':
        fitness = final['validation']['accuracy']
    record = dict(fitness=fitness, final=final, history=history, training_seed=seed,
                  seconds=time.perf_counter()-started, optimizer_updates=updates,
                  path_1based=[i+1 for i in path], **library.stats(path))
    return model, record


PAPER_PATHS = {
    'cancer': [895,31,801,273,404,643,423,593,225,481,673,792,609],
    'mnist': [6687,3969,4418,1321,6319,2817,5933,859,2183,5160,4641,4097,4388,2305,4388],
    'spt': [6687,1665,2177,2401,4899,3042,4901],
    'spt_fixed': [6687,3649,3173,737,1798,60,1217,4905,1826],
}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def plots(output, records, best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for record in records:
        ax.scatter(record['generation'], record['fitness'], color='#3274a1', alpha=.55)
    generations = sorted({r['generation'] for r in records})
    ax.plot(generations, [max(r['fitness'] for r in records if r['generation']==g)
                         for g in generations], 'o-', color='#d55e00')
    ax.set(xlabel='Generation', ylabel='Fitness', ylim=(0,1.02), xticks=generations)
    fig.tight_layout(); fig.savefig(output/'fitness.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for part in ('train','validation'):
        epochs = [r['epoch'] for r in best['history']]
        for axis, key in zip(axes,('accuracy','loss')):
            axis.plot(epochs,[r[part][key] for r in best['history']],label=part)
            axis.set(xlabel='Epoch / step',ylabel=key.capitalize())
            axis.legend()
    fig.tight_layout(); fig.savefig(output/'training.png',dpi=160); plt.close(fig)


def run(config, root, output, circuit_file=None):
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'config.json',config)
    torch.set_num_threads(config.get('threads',1))
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    device = config['device']
    print('Preparing',config['task'],'on',device,flush=True)
    tr, va, te, metadata = dataset(config,root)
    save_json(output/'data_manifest.json',metadata)
    def tensors(data):
        return tuple(torch.tensor(v,dtype=torch.float64,device=device) for v in data)
    train_data, val_data = tensors(tr), tensors(va)
    library = Library({'cancer':7,'spt':9,'mnist':9}[config['task']])
    rng = np.random.default_rng(config['seed'])
    records, population = [], []
    best, best_model = None, None
    started = time.perf_counter()
    for generation in range(1, (1 if circuit_file else config['generations'])+1):
        if circuit_file:
            saved = json.loads(circuit_file.read_text(encoding='utf-8'))
            paths = [[library.blocks.index(Block(tuple(tuple(p) for p in b['pairs']),
                                                 tuple(b['rotations']))) for b in saved['blocks']]]
        elif generation == 1:
            paths = [library.sample(rng,config['initial_length']) for _ in range(config['population'])]
        else:
            parents = sorted(population,key=lambda x:x['fitness'],reverse=True)[:config['survivors']]
            paths = [library.sample(rng,config['extension_length'], [i-1 for i in parent['path_1based']])
                     for parent in parents for _ in range(config['population'])]
        population = []
        for candidate,path in enumerate(paths):
            assert all(library.allowed(library.blocks[a],library.blocks[b]) for a,b in zip(path,path[1:]))
            seed = config['seed'] if config['initialization']=='fixed' else config['seed']+len(records)
            model, record = train(library,path,config,train_data,val_data,seed)
            record.update(generation=generation,candidate=candidate)
            records.append(record); population.append(record)
            if best is None or record['fitness'] > best['fitness']:
                best, best_model = record, model
                save_json(output/'best_circuit.json', dict(record=best,
                          blocks=[asdict(library.blocks[i]) for i in path],
                          parameters=model.theta.detach().cpu().tolist()))
                torch.save(model.state_dict(),output/'best_parameters.pt')
            save_json(output/'candidates.json',records)
            print(f"Generation {generation} candidate {candidate+1}/{len(paths)}: "
                  f"fitness={record['fitness']:.4f}, validation={record['final']['validation']['accuracy']:.4f}, "
                  f"parameters={record['parameters']}, {record['seconds']:.1f}s",flush=True)
        if max(r['fitness'] for r in population) >= config.get('fitness_threshold',1.01):
            break
    test_metrics = metrics(best_model,*tensors(te)) if te is not None else None
    summary = dict(task=config['task'],run_type='circuit_retrain' if circuit_file else 'MQNE_search',
                   best=best,independent_test=test_metrics,seconds=time.perf_counter()-started,
                   circuits_trained=len(records),optimizer_updates=sum(r['optimizer_updates'] for r in records),
                   library_nonempty=len(library.blocks),environment=dict(python=platform.python_version(),
                   torch=torch.__version__,numpy=np.__version__,device=device,platform=platform.platform()))
    save_json(output/'summary.json',summary)
    plots(output,records,best)
    print('Saved',output,flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--circuit',type=Path,help='Retrain an explicit saved best_circuit.json from scratch')
    parser.add_argument('--device',choices=['cpu','cuda'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device'] = args.device
    run(config,Path(__file__).resolve().parent,args.output,args.circuit)


if __name__ == '__main__':
    main()
