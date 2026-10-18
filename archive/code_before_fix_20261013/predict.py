"""Restore a saved explicit circuit and verify its metrics, without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mqne import Block, Circuit, Library, dataset, metrics, save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    config=json.loads((args.result/'config.json').read_text(encoding='utf-8'))
    saved=json.loads((args.result/'best_circuit.json').read_text(encoding='utf-8'))
    summary=json.loads((args.result/'summary.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    library=Library({'cancer':7,'spt':9,'mnist':9}[config['task']])
    path=[library.blocks.index(Block(tuple(tuple(p) for p in b['pairs']),tuple(b['rotations'])))
          for b in saved['blocks']]
    model=Circuit(library,path)
    with torch.no_grad():model.theta.copy_(torch.tensor(saved['parameters'],dtype=torch.float64))
    train,validation,test,meta=dataset(config,root)
    verification={}
    for name,data in [('train',train),('validation',validation),('test',test)]:
        if data is None:continue
        inputs,labels=[torch.tensor(v,dtype=torch.float64) for v in data]
        computed=metrics(model,inputs,labels)
        expected=summary['independent_test'] if name=='test' else saved['record']['final'][name]
        assert abs(computed['loss']-expected['loss'])<1e-8
        assert abs(computed['accuracy']-expected['accuracy'])<1e-12
        with torch.no_grad():probs=torch.cat([model(x) for x in inputs.split(256)]).numpy()
        predictions=(probs>=.5).astype(int)
        matrix=[[int(np.sum((np.array(data[1])==i)&(predictions==j))) for j in range(2)] for i in range(2)]
        verification[name]=dict(metrics=computed,confusion_matrix_true_rows=matrix,
                                probability_range=[float(probs.min()),float(probs.max())])
        np.savetxt(args.result/(name+'_predictions.csv'),
                   np.column_stack([data[1],probs,predictions]),delimiter=',',
                   header='true_label,p_class1,predicted_label',comments='')
    save_json(args.result/'restored_verification.json',verification)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
