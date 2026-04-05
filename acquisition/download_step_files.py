import os
import time
import requests
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

# --- 配置 ---
INPUT_FILE = str(Path(__file__).parent / "step_files_urls.txt")
MAX_WORKERS = 5  # 这是一个很好的平衡点
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 这是一个位置队列，存放 1, 2, 3, 4, 5
# 线程启动时取走一个数字（作为进度条在屏幕上的行号），下载完后还回去
POSITION_QUEUE = queue.Queue()
for i in range(1, MAX_WORKERS + 1):
    POSITION_QUEUE.put(i)


def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)


def download_task(line):
    parts = line.strip().split()
    if not parts:
        return

    url = parts[0]
    filename = get_filename_from_url(url)
    temp_filename = filename + ".tmp"

    # 获取一个屏幕位置 (1-5)
    # block=True 表示如果5个位置都满了，就等待，直到有位置空出来
    current_pos = POSITION_QUEUE.get(block=True)

    try:
        if os.path.exists(filename):
            # 使用 tqdm.write 可以在不打断下方进度条的情况下，在上方打印信息
            tqdm.write(f"✅ [已存在] {filename}")
            return

        resume_byte_pos = 0
        if os.path.exists(temp_filename):
            resume_byte_pos = os.path.getsize(temp_filename)

        # 重试循环
        for attempt in range(1, 4):
            try:
                current_headers = HEADERS.copy()
                if resume_byte_pos > 0:
                    current_headers["Range"] = f"bytes={resume_byte_pos}-"

                with requests.get(
                    url, headers=current_headers, stream=True, timeout=30
                ) as response:
                    if response.status_code not in [200, 206]:
                        # 如果不是200(OK)或206(Partial)，抛错重试
                        if response.status_code == 416:  # 范围错误，通常意味着下完了
                            os.rename(temp_filename, filename)
                            tqdm.write(f"✅ [已补全] {filename}")
                            return
                        raise Exception(f"Status {response.status_code}")

                    total_size = int(response.headers.get("content-length", 0))
                    mode = "ab" if response.status_code == 206 else "wb"
                    if mode == "ab":
                        total_size += resume_byte_pos

                    # --- 进度条核心配置 ---
                    # position=current_pos: 强制固定在第几行
                    # leave=False: 下载完后清空这一行，让给下一个文件
                    desc_text = filename  # 截断文件名防止太长
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=desc_text,
                        initial=resume_byte_pos,
                        position=current_pos,
                        leave=False,
                        ncols=100,  # 限制宽度，防止自动换行导致乱码
                    ) as bar:
                        with open(temp_filename, mode) as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    bar.update(len(chunk))

                    # 循环正常结束，重命名
                    os.rename(temp_filename, filename)
                    tqdm.write(f"✅ [完成] {filename}")
                    return

            except Exception as e:
                # 出错时，在上方打印，不要干扰下方的进度条
                tqdm.write(f"⚠️ [重试{attempt}] {filename}: {e}")
                time.sleep(2)

        tqdm.write(f"❌ [失败] {filename}")

    finally:
        # 无论成功失败，最后必须把位置还回去，供下一个任务使用
        POSITION_QUEUE.put(current_pos)


def main():
    if not os.path.exists(INPUT_FILE):
        print("找不到 urls.txt")
        return

    with open(INPUT_FILE, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"🚀 开始下载 {len(lines)} 个文件 (并发: {MAX_WORKERS})")
    print("-" * 60)

    # 额外的一个总进度条，放在第0行
    # 这里我们只用来占位显示总状态，具体工作在线程里做
    with tqdm(
        total=len(lines),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} Files",
        position=0,
    ) as total_bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            futures = []
            for line in lines:
                fut = executor.submit(download_task, line)
                futures.append(fut)

            # 这里的逻辑是：每当一个任务完成，总进度条+1
            # 但要注意，download_task 是阻塞的吗？不，它是并行的。
            # 为了让 total_bar 动起来，我们需要监控 futures
            import concurrent.futures

            for _ in concurrent.futures.as_completed(futures):
                total_bar.update(1)

    print("\n🎉 全部结束")


if __name__ == "__main__":
    main()

