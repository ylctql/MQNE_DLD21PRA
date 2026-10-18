"""Reproduce gate-library scaling and audit numbered paths without assuming equivalence."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mqne import Library, PAPER_PATHS, save_json


def main():
    output=Path('results/library_audit')
    output.mkdir(parents=True,exist_ok=True)
    scaling=[]
    for n in range(2,11):
        library=Library(n)
        theoretical=sum(2**(n-i)*math.comb(n-i,i) for i in range(n//2+1))
        assert len(library.blocks)==theoretical-1
        degrees=np.array([len(library.successors(i)) for i in range(len(library.blocks))])
        row=dict(qubits=n,including_identity=theoretical,nonempty=len(library.blocks),
                 average_outdegree=float(degrees.mean()),maximum_outdegree=int(degrees.max()))
        scaling.append(row); print(row,flush=True)
        # Cache is intentionally local; discard it before the next library.
        del library
    save_json(output/'scaling.json',scaling)
    fig,ax=plt.subplots(figsize=(7,4))
    for key,label in [('nonempty','Gate-block library (excluding identity)'),
                      ('average_outdegree','Average outdegree'),('maximum_outdegree','Maximum outdegree')]:
        ax.semilogy([r['qubits'] for r in scaling],[r[key] for r in scaling],'o-',label=label)
    ax.set(xlabel='Qubits',ylabel='Count',xticks=range(2,11))
    ax.legend(); fig.tight_layout(); fig.savefig(output/'scaling.png',dpi=160)
    audits={}
    for task,path in PAPER_PATHS.items():
        library=Library(7 if task=='cancer' else 9)
        invalid=[(a,b) for a,b in zip(path,path[1:])
                 if not library.allowed(library.blocks[a-1],library.blocks[b-1])]
        audits[task]=dict(published_path_1based=path,invalid_transitions_under_this_numbering=invalid,
                          **library.stats([i-1 for i in path]))
    save_json(output/'published_path_numbering_audit.json',audits)


if __name__=='__main__':
    main()
