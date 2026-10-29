"""Publish the verified submission bundle to the existing GitHub repository.

Uses the configured Git credential helper without printing or persisting tokens.
Run only when ready to publish the complete, verified source and model bundle.
"""
import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO='faikbairamov/pascal-voc-object-detection'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-auth',action='store_true')
    args=parser.parse_args()
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                          text=True,capture_output=True,cwd=ROOT,
                          env={**__import__('os').environ,'GIT_TERMINAL_PROMPT':'0'})
    if result.returncode:
        raise SystemExit('GitHub credentials are not available through the configured Git helper.')
    credential=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    token=credential.get('password')
    if not token:raise SystemExit('The Git helper did not return a GitHub token.')

    def request(url,method='GET',data=None,content_type='application/json'):
        host=urllib.parse.urlparse(url).hostname
        if host not in {'api.github.com','uploads.github.com'}:
            raise RuntimeError('Refusing a non-GitHub endpoint.')
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                 'X-GitHub-Api-Version':'2022-11-28','User-Agent':'voc-submission-publisher',
                 'Content-Type':content_type}
        payload=json.dumps(data).encode() if isinstance(data,dict) else data
        req=urllib.request.Request(url,data=payload,headers=headers,method=method)
        with urllib.request.urlopen(req,timeout=180) as response:
            return json.load(response)

    info=request('https://api.github.com/repos/'+REPO)
    permissions=info.get('permissions',{})
    print(json.dumps({'repository':info['html_url'],'push_permission':permissions.get('push',False)}),flush=True)
    if args.check_auth:return
    if not permissions.get('push'):raise SystemExit('This GitHub identity lacks repository push permission.')
    bundle=ROOT/'dist/pascal-voc-submission.zip'
    checksum=bundle.with_suffix('.zip.sha256')
    if not bundle.exists() or not checksum.exists():raise SystemExit('Build and verify the submission bundle first.')
    dev=json.loads((ROOT/'reports/evaluation/dev/summary.json').read_text())
    val=json.loads((ROOT/'reports/evaluation/val/summary.json').read_text())
    model=json.loads((ROOT/'models/best.metadata.json').read_text())
    assert dev['checkpoint_sha256']==val['checkpoint_sha256']==model['sha256']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tag='submission-2026-10-29-'+commit[:7]
    body=(f"Complete Pascal VOC object-detection submission. Development mAP@0.5: {dev['map50']:.2%}; "
          f"held-out local validation mAP@0.5: {val['map50']:.2%}. Custom VOC all-points metric, IoU > 0.5.\n\n"
          'The ZIP includes source, the exact trained inference weights, report PDF/Markdown, results, '
          'and a SHA-256 file manifest. See README and SUBMISSION.md for local FastAPI, CLI and Docker instructions.\n\n'
          f"Model SHA-256: `{model['sha256']}`.\nArchive checksum: `{checksum.read_text().split()[0]}`.\n\n"
          'AI assistance and evaluation/reproducibility limitations are disclosed in the included reports.')
    release=request('https://api.github.com/repos/'+REPO+'/releases',method='POST',data={
        'tag_name':tag,'target_commitish':commit,'name':'Pascal VOC submission - 29 October 2026',
        'body':body,'draft':False,'prerelease':False})
    uploaded=[]
    endpoint=release['upload_url'].split('{',1)[0]
    for path,mime in [(bundle,'application/zip'),(checksum,'text/plain')]:
        asset=request(endpoint+'?name='+urllib.parse.quote(path.name),method='POST',
                      data=path.read_bytes(),content_type=mime)
        uploaded.append({'name':asset['name'],'url':asset['browser_download_url']})
    output={'release':release['html_url'],'assets':uploaded,'commit':commit}
    (ROOT/'reports/verification/published_release.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
