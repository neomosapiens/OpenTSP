#!/usr/bin/env python3
"""Check documentation placement and project source naming consistently."""
from pathlib import Path
import re
import sys
import unicodedata

from source_files import inventory


SOURCE_CREDIT_READMES = {"README.md", "opentsp-arch-spec/README.md"}
SOURCE_CREDIT = "OpenTSP is inspired by " + "Gr" + "oq's original Tensor Streaming Processor papers."


def text_for_naming_check(name, text):
    if name in SOURCE_CREDIT_READMES:
        lines = text.splitlines()
        if SOURCE_CREDIT in lines:
            lines.remove(SOURCE_CREDIT)
        return "\n".join(lines)
    return text


def normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if unicodedata.category(c) != 'Cf')


def check(root):
    term = 'gr' + 'oq'
    pattern = re.compile(r'[\W_]{0,2}'.join(re.escape(c) for c in term))
    allowed = set()
    problems = []
    allow = root / 'tools/check_docs_allow.txt'
    if allow.exists():
        for number, line in enumerate(allow.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, _, reason = line.partition('|')
            if not reason.strip():
                problems.append(f'tools/check_docs_allow.txt:{number}: exemption needs a reason')
            allowed.add(name.strip())
    for name in inventory(root):
        path = root / name
        if path.suffix.lower() == '.md' and path.name not in ('README.md', 'AGENTS.md') and not name.startswith(('docs/', 'opentsp-arch-spec/docs/')):
            problems.append(name + ': Markdown belongs in docs/ or opentsp-arch-spec/docs/')
        if pattern.search(normalized(name)):
            problems.append(name + ': disallowed project naming in filename')
        data = path.read_bytes()
        binary = b'\0' in data
        if binary and name in allowed:
            continue
        text = data.decode('utf-8', 'ignore')
        forms = [text if binary else text_for_naming_check(name, text)]
        if binary:
            forms.extend([data.decode('utf-16-le', 'ignore'), data.decode('utf-16-be', 'ignore')])
        if any(pattern.search(normalized(form)) for form in forms):
            problems.append(name + ': disallowed project naming in content')
    return problems


def main():
    root = Path(__file__).resolve().parents[1]
    problems = check(root)
    if problems:
        print('HARD RULE: documentation/source naming violation\n' + '\n'.join(problems), file=sys.stderr)
        return 1
    print('documentation paths and source naming: clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
