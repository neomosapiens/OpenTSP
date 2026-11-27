"""Parse actual host instruction counters without accepting unavailable or duplicate events."""
import csv
import re


def count(system, text):
    if system == "Darwin":
        values = re.findall(r"^\s*(\d+)\s+instructions retired\s*$", text, re.M)
    elif system == "Linux":
        values = [row[0] for row in csv.reader(text.splitlines())
                  if len(row) > 2 and row[2].strip() == "instructions"]
    else:
        raise ValueError("unsupported instruction-counter platform: " + system)
    if len(values) != 1 or not re.fullmatch(r"[0-9]+", values[0]) or int(values[0]) <= 0:
        raise ValueError("missing, duplicated or unavailable retired-instruction counter")
    return int(values[0])
