"""Recheck all finished revised experiments with the independent PyTorch simulator."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from mqne import Block,Library,save_json


def main():
    root=Path(__file__).resolve().parent
    evidence=root/'fix_evidence_20261013'; evidence.mkdir(exist_ok=True)
    checks=[]
    for summary_file in sorted((root/'results').glob('*/summary.json')):
        result=summary_file.parent
        summary=json.loads(summary_file.read_text(encoding='utf-8'))
        if summary.get('protocol')!='fixed_v2':continue  # Historical runs are preserved separately.
        records=json.loads((result/'candidates.json').read_text(encoding='utf-8'))
        config=json.loads((result/'config.json').read_text(encoding='utf-8'))
        circuit=json.loads((result/'best_circuit.json').read_text(encoding='utf-8'))
        manifest=json.loads((result/'data_manifest.json').read_text(encoding='utf-8'))
        library=Library(7 if summary['task']=='cancer' else 9)
        paths=[[v-1 for v in r['path_1based']] for r in records]
        if len(records)!=summary['circuits_trained']:raise ValueError('Candidate count mismatch')
        if summary['run_type']=='MQNE_search' and config.get('fitness_threshold',1.01)>1:
            budget=config['population']*(1+(config['generations']-1)*config['survivors'])
            if len(records)!=budget or max(r['generation'] for r in records)!=config['generations']:
                raise ValueError('Incomplete configured search budget')
        for path,record in zip(paths,records):
            if summary['run_type']!='brickwork10' and not all(library.allowed(library.blocks[a],library.blocks[b]) for a,b in zip(path,path[1:])):
                raise ValueError('Illegal architecture')
            if any(library.stats(path)[k]!=record[k] for k in ('parameters','block_depth','R_gates','CRx_gates')):
                raise ValueError('Resource count mismatch')
            if record['final']['epoch']!=config['epochs']:raise ValueError('Wrong training budget')
        actual=[library.blocks.index(Block(tuple(tuple(p) for p in b['pairs']),tuple(b['rotations']))) for b in circuit['blocks']]
        if [v+1 for v in actual]!=summary['best']['path_1based']:raise ValueError('Best circuit path mismatch')
        if len(circuit['parameters'])!=library.stats(actual)['parameters']:raise ValueError('Parameter count mismatch')
        log=evidence/(result.name+'_restored.log')
        with log.open('w',encoding='utf-8') as stream:
            completed=subprocess.run([sys.executable,'-O',str(root/'predict.py'),'--result',str(result)],
                                     cwd=root,env=os.environ|{'PYTHONIOENCODING':'utf-8'},stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError('PyTorch restore failed: '+str(log))
        checks.append(dict(result=result.name,circuits=len(records),status='passed',optimized_python=True,
                           physical_restore='PyTorch float64/complex128',input_bit_order=manifest.get('input_bit_order','msb')))
    save_json(evidence/'finished_result_checks.json',checks)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    save_json(evidence/'final_source_sha256.json',hashes)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
