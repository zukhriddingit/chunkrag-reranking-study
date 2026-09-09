"""Offline payload verification and CSV formatting only; no scientific estimation."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        parser.error('Use a new, nonexistent output directory outside the extracted package.')
    manifest = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text())
    for rel, expected in manifest['files'].items():
        path = root / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise SystemExit('Payload identity failure: ' + rel)
    sections = []
    counts = {}
    for path in sorted((root / 'aggregates').glob('*.csv')):
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise SystemExit('Malformed CSV: ' + path.name)
        counts[path.name] = len(rows) - 1
        header = '<tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in rows[0]) + '</tr>'
        body = ''.join('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in row) + '</tr>' for row in rows[1:])
        sections.append('<h2>' + html.escape(path.stem) + '</h2><div class="table"><table><thead>' + header + '</thead><tbody>' + body + '</tbody></table></div>')
    output.mkdir(parents=True)
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>ChunkRAG aggregate companion</title><style>body{font:16px system-ui;margin:32px;color:#17212b}h2{margin-top:36px;font-size:20px}.table{overflow-x:auto}table{border-collapse:collapse}td,th{border:1px solid #ccd3da;padding:7px;text-align:left;white-space:nowrap}th{background:#eef2f6}p{max-width:850px}</style><h1>ChunkRAG non-human aggregates</h1><p>Local release candidate; not published. CSV values displayed unchanged. Studies remain separate; neither primary decision rule passed. This page does not recompute statistics or establish full experiment reproducibility. Human records and provisional human tables are excluded.</p>' + ''.join(sections) + '</html>'
    (output / 'report.html').write_text(document)
    result = {'status': 'PASS', 'payload_hashes_verified': len(manifest['files']), 'tables': counts, 'scientific_recomputation': False}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
