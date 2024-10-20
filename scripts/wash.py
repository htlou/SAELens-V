import json
import os
import hashlib
import asyncio
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import random
from collections import Counter, defaultdict

## obelic100k:37827

# 输入和输出文件路径
input_path = "/home/saev/hantao/data/obelics_obelics_100k/obelics_100k.json"
output_path = "/home/saev/changye/data/obelics_100k_washed.json"

# 图片保存目录
images_dir = "/home/saev/changye/data/images"
os.makedirs(images_dir, exist_ok=True)

# 读取输入数据
with open(input_path, "r") as f:
    data = json.load(f)

print(f"Total items: {len(data)}")

# 定义信号量，限制并发请求数量（建议减少到50）
semaphore = asyncio.Semaphore(50)

# 定义重试策略
retry_options = ExponentialRetry(
    attempts=1,  # 总共重试10次
    start_timeout=1,  # 初始退避时间为1秒
    factor=1,  # 退避因子为1，表示线性退避
    statuses={429, 500, 502, 503, 504},  # 移除了403
    exceptions={aiohttp.ClientError, asyncio.TimeoutError},  # 针对这些异常进行重试
)

# 定义多个 User-Agent，随机选择一个
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/92.0.4515.159 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) "
    "Gecko/20100101 Firefox/90.0",
    # 您可以添加更多常见的 User-Agent
]

# 获取随机的请求头
def get_random_headers():
    return {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,"
                  "application/xml;q=0.9,image/avif,image/webp,"
                  "image/apng,*/*;q=0.8,application/signed-exchange;"
                  "v=b3;q=0.9",
        "Referer": "https://www.google.com/",
    }

# 设置在模拟请求时的重试次数
SIMULATED_REQUEST_RETRIES = 2

# 初始化统计数据
error_counter = Counter()
error_details = defaultdict(list)
attempts_counter = Counter()
retry_success_counter = Counter()

async def load_image(session, url, path):
    async with semaphore:  # 确保遵守并发限制
        image_name_hash = hashlib.sha256(url.encode()).hexdigest()
        image_path = f"{path}/{image_name_hash}.png"
        if os.path.exists(image_path):
            print(f"Image {url} already exists")
            return image_path
        # else:
        #     return None
        attempt = 1  # 初始化尝试次数
        attempts_counter[url] = attempt

        try:
            # 第一次尝试：正常请求
            print(f"Attempting to load image {url} (Attempt {attempt})")
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                content = await response.read()
            # 在退出 async with 块后，响应对象将被自动关闭
            image = Image.open(BytesIO(content))
            image.save(image_path)
            image.close()  # 关闭 Image 对象
            print(f"Successfully loaded image {url} on first attempt")
            return image_path
        except aiohttp.ClientResponseError as e:
            # 如果发生403错误，尝试使用模拟浏览器的请求头重新请求
            error_type = type(e).__name__
            error_counter[error_type] += 1
            error_details[error_type].append(f"{url}: {e}")
            if e.status == 403:
                print(f"403 Forbidden for {url}, retrying with browser headers...")
                # 使用模拟请求的重试机制
                for attempt in range(1, SIMULATED_REQUEST_RETRIES + 1):
                    attempts_counter[url] = attempt
                    try:
                        headers = get_random_headers()
                        await asyncio.sleep(random.uniform(1, 3))  # 随机等待1到3秒
                        print(f"Attempting to load image {url} with browser headers (Attempt {attempt})")
                        async with session.get(url, timeout=10, headers=headers) as response:
                            response.raise_for_status()
                            content = await response.read()
                        image = Image.open(BytesIO(content))
                        image.save(image_path)
                        image.close()
                        retry_success_counter[url] = attempt
                        print(f"Successfully loaded image {url} on attempt {attempt} with browser headers")
                        return image_path
                    except Exception as e_inner:
                        error_type_inner = type(e_inner).__name__
                        error_counter[error_type_inner] += 1
                        error_details[error_type_inner].append(f"{url}: {e_inner}")
                        print(f"Attempt {attempt} failed for {url} with browser headers: {e_inner}")
                        if attempt >= SIMULATED_REQUEST_RETRIES:
                            print(f"All attempts failed for {url} with browser headers")
                            return None
                        else:
                            continue  # 继续重试
            else:
                print(f"Error loading image {url}: {e}")
                return None
        except Exception as e:
            # 记录错误类型
            error_type = type(e).__name__
            error_counter[error_type] += 1
            error_details[error_type].append(f"{url}: {e}")
            print(f"Error loading image {url}: {e}")
            return None

async def process_piece(session, piece):
    new_item = piece.copy()
    flag = True
    tasks = []
    for i in range(len(piece["images"])):
        if piece['images'][i] is None and piece['texts'][i] is not None:
            continue
        elif piece['images'][i] is not None and piece['texts'][i] is None:
            tasks.append((i, load_image(session, piece['images'][i], images_dir)))
    if tasks:
        results = await asyncio.gather(*(t[1] for t in tasks))
        for idx, result in zip((t[0] for t in tasks), results):
            if result:
                new_item["images"][idx] = result
            else:
                flag = False
                break
    return new_item if flag else None

async def main():
    # 使用 TCPConnector 限制并发连接数
    connector = aiohttp.TCPConnector(limit=50)
    async with RetryClient(retry_options=retry_options, connector=connector) as retry_client:
        tasks = [process_piece(retry_client, piece) for piece in data]
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            result = await future
            if result is not None:
                results.append(result)
        print(f"Processed items: {len(results)}")
        with open(output_path, "w") as f_out:
            json.dump(results, f_out)
    # 输出统计信息
    print("\nError Types and Counts:")
    for error_type, count in error_counter.items():
        print(f"{error_type}: {count}")
    print("\nDetailed Error Information:")
    for error_type, details in error_details.items():
        print(f"\n{error_type}:")
        for detail in details:
            print(detail)
    print("\nRequest Attempts Distribution:")
    attempts_distribution = Counter(attempts_counter.values())
    for attempts, count in attempts_distribution.items():
        print(f"{attempts} attempts: {count} images")
    if retry_success_counter:
        total_retried = len(retry_success_counter)
        successful_retries = sum(1 for a in retry_success_counter.values() if a <= SIMULATED_REQUEST_RETRIES)
        success_rate = successful_retries / total_retried * 100
        print(f"\nRetry Success Rate: {success_rate:.2f}% ({successful_retries}/{total_retried})")
    else:
        print("\nNo retries were successful.")

# 运行主程序
if __name__ == "__main__":
    asyncio.run(main())
