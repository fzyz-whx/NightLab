"""Fetch only an explicitly confirmed manifest; default is an offline plan."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from http.client import HTTPException
import re
import sys
from threading import Event, Lock
import time

from .cli import ROOT, load_manifest
from .sample import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--confirm-large-download',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',args.dataset):
        parser.error('invalid dataset id')
    if args.execute and not args.confirm_large_download:
        parser.error('user authorization and --confirm-large-download required')
    manifest = load_manifest(ROOT/f'data/manifests/{args.dataset}.json',ROOT)
    plans = [sample(ROOT,manifest,entry['path'],confirmed_large=True) for entry in manifest['files']]
    if not args.execute:
        print(json.dumps({'mode':'manifest dry-run','network':False,'writes':False,
                          'files':plans,'selected_total_bytes':sum(plan['exact_bytes'] for plan in plans)},indent=2))
        return 0
    stop, log_lock = Event(), Lock()
    def log(message):
        with log_lock:
            print(message,file=sys.stderr,flush=True)
    def fetch(plan):
        last_time, last_bytes = time.monotonic(), 0
        path = plan['path']
        def progress(received,total):
            nonlocal last_time,last_bytes
            now = time.monotonic()
            if received-last_bytes >= 64*1024*1024 or now-last_time >= 30 or received == total:
                log(f'{path}: {received}/{total} bytes')
                last_time,last_bytes = now,received
        try:
            result = sample(ROOT,manifest,path,execute=True,confirmed_large=True,
                            progress=progress,stop_requested=stop.is_set)
            log(f"{path}: {result['status']}")
            return {'path':path,'ok':True,'result':result}
        except (OSError,ValueError,HTTPException) as exc:
            log(f'{path}: failed: {exc}')
            return {'path':path,'ok':False,'error':str(exc)}
    pool = ThreadPoolExecutor(max_workers=args.workers)
    results = []
    interrupted = False
    try:
        futures = [pool.submit(fetch,plan) for plan in plans]
        for future in as_completed(futures):
            results.append(future.result())
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        log('cancel requested; waiting for bounded current reads, partials preserved')
    finally:
        pool.shutdown(wait=True,cancel_futures=interrupted)
    print(json.dumps({'mode':'manifest execute','interrupted':interrupted,'files':results},indent=2))
    return 130 if interrupted else (0 if all(result['ok'] for result in results) else 1)


if __name__ == '__main__':
    raise SystemExit(main())
