"""Quick sanity check on main.tex labels, refs, and citations."""

import re
from pathlib import Path

text = Path("main.tex").read_text(encoding="utf-8")

labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
cites_raw = re.findall(r"\\cite\{([^}]+)\}", text)
cites = set()
for c in cites_raw:
    for k in c.split(","):
        cites.add(k.strip())
bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", text))

print(f"labels: {len(labels)}  refs: {len(refs)}  cites: {len(cites)}  bib: {len(bib_keys)}")
print(f"\ndangling \\ref (used but never defined): {refs - labels}")
print(f"dangling \\cite (cited but missing from bib): {cites - bib_keys}")
print(f"\nunused labels: {labels - refs}")
print(f"unused bib_keys: {bib_keys - cites}")

# Find lines with potential issues
issues = []
for i, line in enumerate(text.splitlines(), 1):
    # Unmatched single $ on a line — rough heuristic
    if line.count("$") % 2 != 0 and "\\$" not in line:
        issues.append((i, "odd $ count", line.strip()[:120]))
# Don't report if "$\sim$" usage is present (often used twice on a line)
for i, kind, line in issues[:30]:
    print(f"  line {i}: {kind} | {line}")
