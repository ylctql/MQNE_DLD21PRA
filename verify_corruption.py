"""End-to-end regression of the optimized-Python restoration failure case."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root=Path(__file__).resolve().parent; evidence=root/'fix_evidence_20261013'
    evidence.mkdir(exist_ok=True); checks=[]
    source=root/'results'/'cancer_fixed_paper_seed21'
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        target=Path(temporary)
        for name in ('config.json','summary.json','best_circuit.json','data_manifest.json'):
            shutil.copy2(source/name,target/name)
        corrupted=json.loads((target/'best_circuit.json').read_text(encoding='utf-8'))
        corrupted['record']['final']['validation']['accuracy']=0
        (target/'best_circuit.json').write_text(json.dumps(corrupted),encoding='utf-8')
        for optimized in (False,True):
            command=[sys.executable]+(['-O'] if optimized else [])+[str(root/'predict.py'),'--result',str(target)]
            completed=subprocess.run(command,cwd=root,env=os.environ|{'PYTHONIOENCODING':'utf-8'},
                                     capture_output=True,text=True,encoding='utf-8')
            log=evidence/('corrupted_record_optimized.log' if optimized else 'corrupted_record_normal.log')
            log.write_text(completed.stdout+completed.stderr,encoding='utf-8')
            generated=list(target.glob('*predictions.csv'))+list(target.glob('restored_verification.json'))
            if completed.returncode==0 or 'ValueError: Restored accuracy mismatch' not in completed.stderr or generated:
                raise RuntimeError('Corruption regression failed')
            checks.append(dict(optimized=optimized,exit_code=completed.returncode,
                               exception='ValueError',verification_artifacts_written=False,status='passed'))
    (evidence/'corruption_regression.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
