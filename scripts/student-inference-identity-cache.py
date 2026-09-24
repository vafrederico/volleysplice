#!/usr/bin/env python3
"""Register or explicitly qualify disclosed student numerical-audit hash reuse."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_student_inference_identity_cache as body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for name in ('student-plan', 'qualification-root', 'output'): reg.add_argument('--'+name, type=Path, required=True)
    qualify = sub.add_parser('qualify')
    qualify.add_argument('--plan', type=Path, required=True)
    qualify.add_argument('--plan-sha256', required=True)
    qualify.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    body.require(Path('/mnt/freenas').is_mount() and args.output.resolve().is_relative_to(body.STUDY),
                 'New NAS output required')
    if args.action == 'register':
        plan = body.make_plan(body.identity(args.student_plan), args.qualification_root)
        body._write_new(args.output, plan)
        body.verify_plan(body.identity(args.output))
    else:
        reference = {'path': str(args.plan), 'sha256': args.plan_sha256}
        plan = body.verify_plan(reference); cache = body.StudentInferenceAuditCache(); stages = []
        control = plan['qualification']['controlAudit']; body.verified(control)
        for target in plan['qualification']['targets']:
            fit = body.identity(Path(target['fitDirectory'])/'fit-numerical-audit.json')
            command = body.command_for(target, fit['path'], plan['pythonExecutable'])
            stage = cache.execute(command, plan_ref=reference, task_ref=target['task'],
                panel_ref=target['panel'], fit_audit_ref=fit, receipt_path=target['companionPath'],
                log_path=Path(plan['qualification']['outputRoot'])/'qualification.log')
            body.require(body.identity(target['outputPath'])['sha256'] == control['sha256'],
                         'Cold/warm delegated audit differs from saved uncached control')
            stages.append(stage)
        body.verified(control)
        result = {'kind': body.QUALIFICATION_KIND, 'passed': True, 'plan': reference,
            'adapter': body.identity(body.__file__), 'source': body.identity(__file__),
            'controlAudit': control, 'executions': stages,
            'byteExactColdParity': True, 'byteExactWarmParity': True,
            'productionOutputsModified': False, 'calibrationOrOutcomeMetricsExecuted': False}
        body._write_new(args.output, result)
        body.verify_qualification(body.identity(args.output), reference)
    print(json.dumps(body.identity(args.output)), flush=True)


if __name__ == '__main__': main()
