"""Run the charging-price study's reproducible analysis workflows."""
from pathlib import Path
import argparse
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workflow',choices=['core','raw','distribution','simulation'])
    p.add_argument('--output',type=Path,required=True,help='New output directory outside the repository')
    p.add_argument('--bootstrap',choices=['saved','regenerate'],default='saved')
    p.add_argument('--raw-dir',type=Path)
    p.add_argument('--phase',choices=['pilot','formal'],default='pilot')
    p.add_argument('--reps',type=int)
    a=p.parse_args();root=Path(__file__).resolve().parent;out=a.output.resolve()
    if sys.flags.optimize:p.error('Python -O disables required scientific assertions')
    if out==root or root in out.parents:p.error('--output must be outside the repository')
    if out.exists() and any(out.iterdir()):p.error('--output must be empty')
    subprocess.run([sys.executable,str(root/'scripts/verify_release.py')],check=True)
    if a.workflow in ['core','raw']:
        cmd=[root/'scripts/reproduce_prices.py','--mode','derived' if a.workflow=='core' else 'raw',
             '--package-root',root,'--output-dir',out,'--bootstrap',a.bootstrap]
        if a.workflow=='raw':
            if a.raw_dir is None:p.error('raw requires --raw-dir')
            cmd+=['--raw-dir',a.raw_dir.resolve()]
    elif a.workflow=='distribution':cmd=[root/'scripts/reproduce_distributions.py','--output',out]
    else:
        cmd=[root/'src/simulation/reconstruction.py','--phase',a.phase,'--output',out]
        if a.reps is not None:cmd+=['--reps',a.reps]
    subprocess.run([sys.executable,*[str(x) for x in cmd]],check=True)

if __name__=='__main__':main()
