import json, re, glob, os
out = {}
for p in sorted(glob.glob('/mnt/user-data/uploads/m2020/zcam/*.IMG')):
    with open(p, 'rb') as f:
        head = f.read(200000).decode('latin-1')
    end = head.find('\nEND\r') if '\nEND\r' in head else head.find('\nEND\n')
    head = head[:end] if end > 0 else head
    def grab(key):
        m = re.search(r'\b' + key + r'\s*=\s*(\([^)]*\)|\S+)', head, re.S)
        return m.group(1) if m else None
    # INSTRUMENT_STATE_PARMS group
    g = head.find('GROUP                              = INSTRUMENT_STATE_PARMS')
    g = head.find('INSTRUMENT_STATE_PARMS')
    seg = head[g:g+4000]
    tv = re.search(r'INSTRUMENT_TEMPERATURE\s*=\s*\(([^)]*)\)', seg, re.S)
    tn = re.search(r'INSTRUMENT_TEMPERATURE_NAME\s*=\s*\(([^)]*)\)', seg, re.S)
    vals = [float(re.sub(r'<.*?>', '', v).strip()) for v in tv.group(1).split(',')] if tv else []
    names = [n.strip().strip('"\'') for n in tn.group(1).split(',')] if tn else []
    stem = os.path.basename(p)[:-4]
    rec = dict(zip(names, vals))
    rec['focus'] = grab('FOCUS_POSITION_COUNT')
    rec['sclk'] = grab('SPACECRAFT_CLOCK_START_COUNT')
    rec['lmst'] = grab('LOCAL_MEAN_SOLAR_TIME')
    out[stem] = rec
json.dump(out, open('ztemps.json', 'w'), indent=1)
for k, v in out.items():
    print(k[:40], {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
