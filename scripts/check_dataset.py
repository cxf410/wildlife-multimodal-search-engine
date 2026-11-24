#!/usr/bin/env python3
"""Report genuine local file presence and update CSV flags; no Internet required."""
import csv,hashlib
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
CSV=BASE/'metadata'/'dataset_manifest.csv'
with CSV.open(encoding='utf-8',newline='') as f:
    rd=csv.DictReader(f);fields=rd.fieldnames;data=list(rd)
count=Counter()
for row in data:
    p=BASE/row['local_path']
    valid=p.is_file() and p.stat().st_size>100
    if valid:
        row['download_status']='local_available'
        if row['modality'] in ('image','audio','video') and not row.get('sha256'):
            digest=hashlib.sha256()
            with p.open('rb') as fh:
                for buf in iter(lambda:fh.read(1048576),b''):digest.update(buf)
            row['sha256']=digest.hexdigest()
    else:
        row['download_status']='external_not_downloaded' if row['modality']!='text' else 'missing_local_text'
        row['sha256']=''
    count[(row['modality'],'present' if valid else 'missing')]+=1
    print(('OK     ' if valid else 'MISSING'),row['modality'].ljust(6),row['species_id'].ljust(23),row['local_path'])
with CSV.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
print('\nSUMMARY')
for mod in ['text','image','audio','video']:
    print(f'{mod.title():7}: {count[(mod,"present")]} files present, {count[(mod,"missing")]} missing')
print('Manifest flags updated to match the actual filesystem.')
