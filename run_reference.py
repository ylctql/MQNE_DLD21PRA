"""Train explicit published Fig. S11 or a stated fixed-architecture control."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mqne import Block,Library,dataset,train,metrics,plots,save_json


def paper_s11():
    # Transcribed from SI page 6; zero-based wires top to bottom.
    specifications=[
        ([],list(range(9))),
        ([(1,0),(7,8)],[]),
        ([(6,7)],[1,8]),
        ([(0,1),(5,6),(7,8)],[]),
        ([],[1,5,6,7,8]),
        ([(1,0),(5,6)],[]),
        ([(1,2)],[5,6]),
        ([(3,2),(5,4),(6,7)],[1]),
        ([(5,6),(7,8)],[2,3]),
        ([(4,5)],[6,7,8]),
        ([(7,6)],[]),
        ([(6,5),(7,8)],[]),
        ([(6,7)],[5,8]),
        ([(5,6),(8,7)],[]),
        ([(6,7)],[5,8]),
    ]
    return [Block(tuple(sorted(pairs)),tuple(rotations)) for pairs,rotations in specifications]


def brickwork(qubits,layers=10):
    blocks=[]
    for _ in range(layers):
        blocks.append(Block((),tuple(range(qubits))))
        for start in (0,1):
            blocks.append(Block(tuple((q,q+1) for q in range(start,qubits-1,2)),()))
    return blocks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--architecture',choices=['paper_s11','brickwork10','legacy_spt_control'],required=True)
    parser.add_argument('--seed',type=int,default=21)
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    config=json.loads(args.config.read_text(encoding='utf-8'))|dict(seed=args.seed)
    torch.set_num_threads(1)
    library=Library(7 if config['task']=='cancer' else 9)
    if args.architecture=='paper_s11' and config['task']!='mnist':raise ValueError('Fig. S11 is MNIST only')
    legacy=None
    if args.architecture=='legacy_spt_control':
        if config['task']!='spt':raise ValueError('Legacy SPT control is SPT only')
        legacy=json.loads((root/'references'/'spt_legacy_control.json').read_text(encoding='utf-8'))
        blocks=[Block(tuple(tuple(p) for p in b['pairs']),tuple(b['rotations'])) for b in legacy['blocks']]
    else:blocks=paper_s11() if args.architecture=='paper_s11' else brickwork(library.qubits)
    path=[library.blocks.index(b) for b in blocks]
    if args.architecture=='paper_s11':
        expected=dict(block_depth=15,R_gates=28,CRx_gates=22,parameters=106)
        stats=library.stats(path)
        if any(stats[k]!=v for k,v in expected.items()):raise ValueError('Fig. S11 resource transcription mismatch')
        if not all(library.allowed(a,b) for a,b in zip(blocks,blocks[1:])):raise ValueError('Fig. S11 contains illegal transition')
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/'config.json',config)
    tr,va,te,manifest=dataset(config,root)
    if legacy and set(legacy['old_lambdas'])&set(manifest['lambdas']):
        raise ValueError('Legacy architecture data overlap with new evaluation pool')
    save_json(args.output/'data_manifest.json',manifest)
    tensors=lambda values:tuple(torch.tensor(v,dtype=torch.float64) for v in values)
    started=time.perf_counter(); model,record=train(library,path,config,tensors(tr),tensors(va),args.seed)
    record.update(generation=1,candidate=0)
    test=metrics(model,*tensors(te)) if te is not None else None
    save_json(args.output/'best_circuit.json',dict(record=record,blocks=[asdict(b) for b in blocks],
                  parameters=model.theta.detach().tolist(),
                  provenance={k:manifest[k] for k in ('protocol','task','source_sha256','master_split_sha256')}))
    torch.save(model.state_dict(),args.output/'best_parameters.pt')
    save_json(args.output/'candidates.json',[record])
    summary=dict(task=config['task'],run_type=args.architecture,best=record,independent_test=test,
                 seconds=time.perf_counter()-started,circuits_trained=1,optimizer_updates=record['optimizer_updates'],
                 protocol='fixed_v2',backend='numba',
                 architecture_source='SI_MQNE.pdf page 6 Fig. S11, manually transcribed' if args.architecture=='paper_s11'
                                     else ('Historical spt_random6_seed21 architecture, fresh fixed_v2 data, no shared lambda; diagnostic control, not a new search'
                                           if legacy else '10 layers of full R rotations and two checkerboard CRx blocks; approximate control, not author original code'))
    save_json(args.output/'summary.json',summary); plots(args.output,[record],record)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
