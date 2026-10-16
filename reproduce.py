"""Single entry point: check archived evidence, regenerate figures, or replay mass tests."""
from pathlib import Path
import argparse, subprocess, sys
ROOT = Path(__file__).resolve().parent

def run(script, *args):
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'figures', 'mass', 'road'])
    stage = parser.parse_args().stage
    if stage == 'check':
        for name in ['test_core.py', 'validate_results.py', 'analyze_mass_sensitivity.py',
                     'analyze_optimized_mass.py', 'analyze_revision_details.py', 'validate_road_halfcar.py', 'check_road_coverage.py', 'mass_regime_study.py']:
            run(name, *(['--check-only'] if name == 'analyze_optimized_mass.py' else ['--check'] if name=='validate_road_halfcar.py' else ['audit'] if name=='mass_regime_study.py' else []))
    elif stage == 'figures':
        run('analyze_results.py', '--figures-only')
        run('plot_supporting_figures.py')
        run('analyze_optimized_mass.py', '--figures-only')
        run('validate_road_halfcar.py', '--figures')
        run('mass_regime_study.py', 'figures')
    elif stage == 'road':
        run('validate_road_halfcar.py')
    else:
        run('evaluate_paired_mass.py', '--name', 'mass_heads_final', '--split', 'heldout', '--mpc')
        for name in ['mass_heads_old_estimator', 'original_reference']:
            run('evaluate_paired_mass.py', '--name', name, '--split', 'heldout')
        run('mass_regime_study.py', 'validation')
        run('mass_regime_study.py', 'mpc')
        run('mass_regime_study.py', 'audit')
        run('mass_regime_study.py', 'figures')

if __name__ == '__main__':
    main()
