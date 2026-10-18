"""Run distribution, population, threshold and input-use analyses."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','main','common','extensions','application'],default='all')
    args=parser.parse_args();out=args.output.resolve()
    if out==ROOT or ROOT in out.parents:parser.error('Output must be outside the repository')
    out.mkdir(parents=True,exist_ok=True)
    def run(script,*arguments):
        subprocess.run([sys.executable,str(ROOT/script),*[str(x) for x in arguments]],check=True)
    if args.stage in ['all','main']:
        run('src/distribution/portable_main_results.py','--package',ROOT,'--output',out/'main')
        run('src/distribution/render_external_direction.py','--input',out/'main/external_direction_figure_data.csv','--output',out/'figures')
    if args.stage in ['all','common']:
        run('src/distribution/build_task_comparisons.py','--input-dir',ROOT/'data/derived/common','--output-dir',out/'common')
    if args.stage in ['all','extensions']:
        run('src/extensions/analyze_extensions.py','--out',out/'extensions','--protocol',ROOT/'config/extensions.json',
            '--beijing-panel',ROOT/'data/derived/extensions/beijing_panel.parquet',
            '--beijing-menu',ROOT/'data/derived/extensions/beijing_menu.parquet',
            '--external-panel',ROOT/'data/derived/extensions/urbanev_panel.parquet')
        run('src/extensions/fix_e2_reporting.py','--draws-dir',ROOT/'data/reference/masking/draws',
            '--old-summary',ROOT/'data/reference/masking/scenario_summary.csv',
            '--out-dir',out/'extensions/analysis_results/e2_reporting',
            '--table-dir',out/'extensions/working_source/tables','--expected-draws',900)
        run('src/extensions/build_extension_assets.py','--out',out/'extensions')
    if args.stage in ['all','application']:
        run('src/distribution/analyze_input_application.py','--input',ROOT/'data/derived/common/unit_level_signed.csv',
            '--protocol',ROOT/'config/application.json','--output',out/'application')
        run('src/distribution/render_core_input_figure.py','--inputdir',out/'application','--figuredir',out/'figures',
            '--checks',out/'figures/core_input_figure_checks.json')
    (out/'replay_status.json').write_text(json.dumps({'status':'PASS','stage':args.stage,
        'scope':'Panel recalculation, saved-draw reaggregation and identified frozen-summary extraction; no historical model refit.'},indent=2)+'\n')

if __name__=='__main__':main()
