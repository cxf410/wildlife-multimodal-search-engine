#!/usr/bin/env python3
"""Download the actual Wikimedia Commons media referenced in the CSV manifest.
Run on a laptop/desktop with Internet. Non-destructive; reports failures clearly.
Uses only the public MediaWiki Commons API and requests.
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

try:
    import requests
except ImportError:
    print('Missing dependency: run `python -m pip install requests`')
    raise SystemExit(2)

BASE = Path(__file__).resolve().parents[1]
CSV_PATH = BASE / 'metadata' / 'dataset_manifest.csv'
API = 'https://commons.wikimedia.org/w/api.php'
UA = 'CWRU-CSDS305-WildlifeEducationalDemo/1.0 (student, Wikimedia Commons metadata retrieval; Python requests)'

def info(session, name, medium):
    prop = 'videoinfo' if medium == 'video' else 'imageinfo'
    params = {'action':'query', 'format':'json','formatversion':'2','prop':prop,'titles':'File:'+name}
    params['viprop' if medium == 'video' else 'iiprop'] = 'url|size|derivatives' if medium=='video' else 'url|size'
    resp=session.get(API,params=params,timeout=35)
    resp.raise_for_status()
    payload=resp.json()
    if payload.get('error'): raise RuntimeError(str(payload['error']))
    pages=payload.get('query',{}).get('pages',[])
    if not pages or pages[0].get('missing'): raise RuntimeError('Wikimedia page not found')
    versions=pages[0].get(prop,[])
    if not versions: raise RuntimeError('No media URL returned by Wikimedia API')
    return versions[0]

def pick_url(obj, medium, max_bytes):
    original=obj.get('url')
    if not original: raise RuntimeError('Missing original URL')
    if medium != 'video': return original, obj.get('size',0), 'original'
    derivatives=obj.get('derivatives',[])
    variants=[]
    for d in derivatives:
        u=d.get('src') or d.get('url')
        if not u: continue
        typ=(d.get('type') or '').lower()
        label=(d.get('shorttitle') or d.get('title') or '').lower()
        h=int(d.get('height') or 9999)
        size=int(d.get('size') or 0)
        # Prefer videos with 240-480p resolution; avoid audio-only derivatives.
        if ('video/' in typ or u.lower().split('?')[0].endswith(('.webm','.mp4','.ogv'))) and h<=480:
            rank=abs(h-360)+(0 if 'webm' in typ or '.webm' in u.lower() else 20)
            variants.append((rank,size,u,label))
    if variants:
        variants.sort(key=lambda x:x[0])
        _,size,u,label=variants[0]
        return u,size,label or 'low-res derivative'
    origsize=int(obj.get('size') or 0)
    if origsize and origsize>max_bytes:
        raise RuntimeError(f'Original is {origsize/1024/1024:.1f} MB and no small derivative is available; use the source page manually')
    return original,origsize,'original video'

def download(session,url,output,max_bytes):
    temp=output.with_suffix(output.suffix+'.part')
    output.parent.mkdir(parents=True,exist_ok=True)
    if temp.exists(): temp.unlink()
    try:
        with session.get(url,timeout=(15,90),stream=True) as response:
            response.raise_for_status()
            mime=response.headers.get('content-type','').lower()
            if 'text/html' in mime: raise RuntimeError('Server returned HTML rather than a media file')
            length=int(response.headers.get('content-length') or 0)
            if length and length>max_bytes: raise RuntimeError(f'File exceeds max size ({length/1024/1024:.1f} MB)')
            count=0
            digest=hashlib.sha256()
            with temp.open('wb') as f:
                for chunk in response.iter_content(chunk_size=262144):
                    if not chunk: continue
                    count += len(chunk)
                    if count>max_bytes: raise RuntimeError('Download exceeded maximum size limit')
                    f.write(chunk)
                    digest.update(chunk)
            if count < 100: raise RuntimeError('File too small to be valid media')
        temp.replace(output)
        return digest.hexdigest(),count
    finally:
        if temp.exists(): temp.unlink()

def run():
    pa=argparse.ArgumentParser(description='Download educational bird photo/audio/video pilot dataset from Wikimedia Commons')
    pa.add_argument('--media',default='image,audio,video',help='Comma-separated modalities: image,audio,video')
    pa.add_argument('--max-mb',type=int,default=45,help='Maximum megabytes per media file (default 45)')
    pa.add_argument('--limit',type=int,default=0,help='Optional limit for testing (e.g. 2)')
    pa.add_argument('--delay',type=int,default=12,help='Seconds between media items, to respect Wikimedia rate limits (default 12)')
    args=pa.parse_args()
    allowed={m.strip() for m in args.media.split(',')}
    if not allowed<= {'image','audio','video'}: pa.error('Choose image,audio,video')
    with CSV_PATH.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames; rows=list(reader)
    session=requests.Session();session.headers.update({'User-Agent':UA,'Accept':'application/json, audio/*, video/*, image/*, */*'})
    done=failed=0
    rate_limited=False
    for r in rows:
        if r['modality'] not in allowed: continue
        if args.limit and done+failed>=args.limit: break
        dst=BASE/r['local_path']
        if dst.exists() and dst.stat().st_size>100:
            r['download_status']='local_available'
            if not r.get('sha256'):
                sha=hashlib.sha256()
                with dst.open('rb') as f:
                    for chunk in iter(lambda:f.read(1048576),b''): sha.update(chunk)
                r['sha256']=sha.hexdigest()
            print('[EXISTS]',r['id'],dst.stat().st_size,'bytes')
            done+=1
            continue
        print('[GET]',r['id'])
        try:
            inf=info(session,r['commons_filename'],r['modality'])
            url,original_bytes,variant=pick_url(inf,r['modality'],args.max_mb*1024*1024)
            # Transcoded files can have a different extension from the media original.
            media_ext=Path(urlsplit(url).path).suffix.lower()
            if r['modality']=='video' and media_ext in ('.webm','.mp4','.ogv','.mov') and dst.suffix.lower()!=media_ext:
                dst=dst.with_suffix(media_ext)
                r['local_path']=dst.relative_to(BASE).as_posix()
            # Pause before the file download too (separate request from API metadata lookup).
            time.sleep(args.delay)
            sha,count=download(session,url,dst,args.max_mb*1024*1024)
            r['download_status']='local_available'
            r['sha256']=sha
            print('  OK:',count,'bytes;',variant,'=>',dst.relative_to(BASE))
            done+=1
        except (requests.RequestException,RuntimeError,OSError,ValueError,KeyError) as exc:
            r['download_status']='external_not_downloaded'
            r['sha256']=''
            print('  FAILED:',str(exc)[:250]);print('  Manual page:',r['source_page'])
            failed+=1
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                rate_limited=True
                print('  STOP: Wikimedia is temporarily rate limiting requests (HTTP 429).')
                print('  No more requests will be made in this run. Try again later, not repeatedly.')
        # Save after each row so interrupted progress is not lost.
        with CSV_PATH.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        if rate_limited:
            break
        time.sleep(args.delay)
    print('\nFinished. Verified existing/downloaded:',done,'not downloaded:',failed)
    if failed: print('Some files could not be obtained automatically. Try again later or use the provided source pages for manual download.')
    print('Now run `python scripts/check_dataset.py`.')

if __name__=='__main__':run()
