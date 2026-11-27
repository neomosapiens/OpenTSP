"""Write config/schema.json as a C string literal (#define TSP_CONFIG_SCHEMA) for the
harnesses that check a configuration with tsp_config_load (c_emulator/tsp_config_check.c):
tsp_sim, isa_query, exhaustive32 and the unit-test runner.

usage: schema_header.py <schema.json> <out.h>
"""
import json, sys

text = open(sys.argv[1]).read()
json.loads(text)
body = "".join(f"\\{b:03o}" if b < 32 or b in (34, 92, 63) or b > 126 else chr(b) for b in text.encode())
open(sys.argv[2], "w").write(f'#define TSP_CONFIG_SCHEMA "{body}"\n')
