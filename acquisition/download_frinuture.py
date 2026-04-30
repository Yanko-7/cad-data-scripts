import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from tqdm import tqdm

ACCESS_KEY = "on_o6YWYdyVf9JjkFxUe0O8P"
SECRET_KEY = "9Tv0JZHzITstNa8pLZexKvu8JQ8UFeoiDrCPq0qczVNL1QiS"
BASE_URL = "https://cad.onshape.com"
AUTH = (ACCESS_KEY, SECRET_KEY)

MAX_WORKERS = 16
POLL_INTERVAL = 8
RETRY_BACKOFF = [10, 30, 60]
MAX_CALLS_PER_MINUTE = 60

_rate_lock = Lock()
_rate_timestamps: list = []
DATASET_DIR = Path(__file__).parent.parent / "furniture-dataset"
SPLITS = ["train", "val", "test"]


def _throttle():
    while True:
        with _rate_lock:
            now = time.time()
            _rate_timestamps[:] = [t for t in _rate_timestamps if now - t < 60]
            if len(_rate_timestamps) < MAX_CALLS_PER_MINUTE:
                _rate_timestamps.append(time.time())
                return
            wait = 60 - (now - _rate_timestamps[0]) + 0.5
        time.sleep(wait)


def _api(method, url, **kwargs):
    _throttle()
    for wait in RETRY_BACKOFF:
        resp = getattr(requests, method)(url, auth=AUTH, **kwargs)
        if resp.status_code != 402:
            return resp
        time.sleep(wait)
        _throttle()
    return getattr(requests, method)(url, auth=AUTH, **kwargs)


def _get_default_wid(did):
    resp = _api("get", f"{BASE_URL}/api/documents/d/{did}",
                headers={"Accept": "application/json;charset=UTF-8"})
    resp.raise_for_status()
    return resp.json()["defaultWorkspace"]["id"]


def _post_translation(did, wvm, wvmid, eid, ep, payload, headers):
    url = f"{BASE_URL}/api/{ep}/d/{did}/{wvm}/{wvmid}/e/{eid}/translations"
    return _api("post", url, json=payload, headers=headers)


def _translate(item):
    did, eid = item["did"], item["eid"]
    wvm = "v" if "vid" in item else "w"
    wvmid = item.get("vid") or item["wid"]
    doc_type = item.get("type", "partstudio")

    primary_ep = "assemblies" if doc_type == "assembly" else "partstudios"
    fallback_ep = "partstudios" if primary_ep == "assemblies" else "assemblies"
    headers = {"Accept": "application/json;charset=UTF-8"}
    payload = {"formatName": "STEP", "storeInDocument": False, "translate": True}

    # 1. try primary → fallback endpoint with stored wid
    resp = None
    for ep in (primary_ep, fallback_ep):
        resp = _post_translation(did, wvm, wvmid, eid, ep, payload, headers)
        if resp.status_code != 404:
            break

    # 2. if still 404 and wid was used, fetch the document's current default workspace and retry
    if resp.status_code == 404 and wvm == "w":
        wvmid = _get_default_wid(did)
        for ep in (primary_ep, fallback_ep):
            resp = _post_translation(did, wvm, wvmid, eid, ep, payload, headers)
            if resp.status_code != 404:
                break

    resp.raise_for_status()
    tid = resp.json()["id"]

    status_url = f"{BASE_URL}/api/translations/{tid}"
    while True:
        sr = _api("get", status_url, headers=headers)
        sr.raise_for_status()
        data = sr.json()
        state = data["requestState"]
        if state == "DONE":
            return data["resultExternalDataIds"][0]
        elif state == "FAILED":
            raise RuntimeError(data.get("failureReason", "unknown"))
        time.sleep(POLL_INTERVAL)


def _download(did, external_id, filepath):
    url = f"{BASE_URL}/api/documents/d/{did}/externaldata/{external_id}"
    resp = _api("get", url, stream=True)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    return os.path.getsize(filepath)


def process(item, split_dir):
    filename = f"{item['category']}_{item['data_id']}.step"
    filepath = Path(split_dir) / filename

    if filepath.exists() and filepath.stat().st_size > 0:
        return f"[跳过] {filename}"
    if "vid" not in item and "wid" not in item:
        return f"[错误] {filename}: 缺少 wid/vid"

    try:
        external_id = _translate(item)
        size = _download(item["did"], external_id, filepath)
        if size == 0:
            return f"[警告] {filename}: 下载成功但文件为空"
        return f"[成功] {filename} ({size / 1024:.1f} KB)"
    except requests.exceptions.RequestException as e:
        detail = e.response.text if e.response is not None else str(e)
        return f"[失败] {filename}: {detail}"
    except Exception as e:
        return f"[失败] {filename}: {e}"


def main():
    all_tasks = []
    for split in SPLITS:
        json_path = DATASET_DIR / f"{split}.json"
        if not json_path.exists():
            print(f"[跳过] {json_path} 不存在")
            continue
        split_dir = DATASET_DIR / split
        split_dir.mkdir(exist_ok=True)
        items = json.loads(json_path.read_text())
        all_tasks.extend((item, split_dir, split) for item in items)

    total = len(all_tasks)
    print(f"共 {total} 个模型 (train/val/test 并行)，并发进程数: {MAX_WORKERS}\n" + "="*50)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, item, split_dir): (item, split)
                   for item, split_dir, split in all_tasks}
        with tqdm(total=total, unit="file") as bar:
            for future in as_completed(futures):
                item, split = futures[future]
                try:
                    msg = future.result()
                except Exception as e:
                    msg = f"[异常] {item.get('data_id')}: {e}"
                bar.set_postfix_str(f"[{split}] {msg}")
                bar.update(1)


if __name__ == "__main__":
    main()