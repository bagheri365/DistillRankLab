"""Create a validated Markdown and JSON summary from the saved full 7E report."""
import argparse
import json
from pathlib import Path

from distillranklab.research_summary import consolidate, markdown_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('datasets/scifact_multiseed.json'))
    parser.add_argument('--markdown', type=Path, default=Path('datasets/scifact_research_summary.md'))
    parser.add_argument('--json-output', type=Path, default=Path('datasets/scifact_research_summary.json'))
    args = parser.parse_args()
    if len({args.input.resolve(), args.markdown.resolve(), args.json_output.resolve()}) != 3:
        parser.error('input and output paths must be distinct')
    summary = consolidate(json.loads(args.input.read_text(encoding='utf-8')))
    for path in (args.markdown, args.json_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_summary(summary), encoding='utf-8')
    args.json_output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'markdown': str(args.markdown), 'json': str(args.json_output),
                      'architectures': list(summary['architectures']),
                      'test_used': summary['test_used']}, indent=2))


if __name__ == '__main__':
    main()
