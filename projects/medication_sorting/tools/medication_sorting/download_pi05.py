"""官方 π0.5 基座断点下载与离线复验：长度及 manifest 的 MD5/CRC32C。

GCS 组合对象没有 MD5，必须检查其 CRC32C，不能只检查长度。
CRC32C 使用依赖层的 google-crc32c；安装就绪后通过 CPU wrapper 运行。
--verify-only 只读取本地文件并写校验报告，不下载、不加载模型或 GPU。
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
PREFIX='checkpoints/pi05_base/'
DEST=ROOT/'data/medication_sorting/pretrained/pi05_base'
DOWNLOAD_MANIFEST=ROOT/'data/medication_sorting/reports/pi05-download-manifest.json'


def verify_file(item, path):
    """逐文件实际核对所有可用的官方哈希，内存只保留一个读取块。"""
    expected_md5 = item.get('md5Hash')
    expected_crc = item.get('crc32c')
    if not expected_md5 and not expected_crc:
        raise ValueError(f'官方 manifest 缺少校验和：{item["name"]}')
    result = {'name': item['name'], 'path': str(path), 'expected_size': int(item['size']),
              'expected_md5': expected_md5, 'expected_crc32c': expected_crc,
              'actual_size': None, 'actual_md5': None, 'actual_crc32c': None, 'verified': False}
    if not path.is_file():
        result['error'] = '文件缺失'
        return result
    result['actual_size'] = path.stat().st_size
    if result['actual_size'] != result['expected_size']:
        result['error'] = '长度不匹配'
        return result
    md5 = hashlib.md5() if expected_md5 else None
    crc = None
    if expected_crc:
        try:
            import google_crc32c
        except ImportError as exc:
            raise RuntimeError('CRC32C 校验需要 google-crc32c；安装完成后请通过 pi05_python.sh --cpu 运行') from exc
        crc = google_crc32c.Checksum()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            if md5 is not None:
                md5.update(block)
            if crc is not None:
                crc.update(block)
    if md5 is not None:
        result['actual_md5'] = base64.b64encode(md5.digest()).decode()
    if crc is not None:
        result['actual_crc32c'] = base64.b64encode(crc.digest()).decode()
    result['verified'] = ((not expected_md5 or result['actual_md5'] == expected_md5)
                          and (not expected_crc or result['actual_crc32c'] == expected_crc))
    if not result['verified']:
        result['error'] = '校验和不匹配'
    return result


def verify_manifest(manifest_path, destination):
    """离线读取 manifest 中的每一个文件，不信任旧的完成标记。"""
    manifest_bytes = Path(manifest_path).read_bytes()
    items = json.loads(manifest_bytes)['items']
    if not items:
        raise ValueError('官方 manifest 为空，不能报告校验成功')
    checks = [verify_file(item, Path(destination) / item['name'].removeprefix(PREFIX)) for item in items]
    verified = sum(check['verified'] for check in checks)
    md5_count = sum(bool(check['expected_md5']) and check['actual_md5'] == check['expected_md5'] for check in checks)
    crc_count = sum(bool(check['expected_crc32c']) and check['actual_crc32c'] == check['expected_crc32c'] for check in checks)
    return {'source': 'gs://openpi-assets/checkpoints/pi05_base',
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'files': [check['path'] for check in checks], 'checks': checks,
            'total_files': len(checks), 'verified_files': verified,
            'expected_bytes': sum(check['expected_size'] for check in checks),
            'all_files_verified': verified == len(checks),
            'md5_verified_files': md5_count, 'crc32c_verified_files': crc_count,
            'md5_verified': md5_count == len(checks), 'crc32c_verified': crc_count == len(checks)}


def fetch(item):
    path=DEST/item['name'].removeprefix(PREFIX)
    size=int(item['size'])
    def valid(file):
        return verify_file(item, file)['verified']
    if valid(path):
        return str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    url='https://storage.googleapis.com/openpi-assets/'+item['name']
    for attempt in range(5):
        offset=temp.stat().st_size if temp.exists() else 0
        try:
            if offset<size:
                request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'})
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request,timeout=60) as response:
                    mode='ab' if response.status==206 else 'wb'
                    with temp.open(mode) as out:
                        while block:=response.read(4*1024**2):out.write(block)
            elif size==0:temp.touch()
            if valid(temp):
                temp.replace(path)
                print('已验证',item['name'],size,flush=True)
                return str(path)
            temp.unlink(missing_ok=True)
        except Exception as exc:
            print('下载重试',attempt+1,item['name'],str(exc),flush=True)
    raise RuntimeError(f'官方权重下载失败：{item["name"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='官方 π0.5 基座下载及 MD5/CRC32C 校验')
    parser.add_argument('--verify-only', action='store_true', help='仅读本地 manifest 全部文件，不下载')
    args = parser.parse_args(argv)
    try:
        if not args.verify_only:
            items = json.loads(DOWNLOAD_MANIFEST.read_text())['items']
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(fetch, items))
        report = verify_manifest(DOWNLOAD_MANIFEST, DEST)
        (DEST/'download-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report['all_files_verified'] else 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({'error': str(exc), 'all_files_verified': False}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
