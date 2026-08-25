import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# === 基础配置 ===
API_KEY = os.getenv("SILICONFLOW_KEY", "")  # 密钥从环境变量读取，勿硬编码
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# === 高级参数 ===
TEMPERATURE = 0.5
TOP_P = 0.95
MAX_TOKENS = 1024
MAX_WORKERS = 3
MAX_RETRIES = 3
RETRY_DELAY = 10  # 秒
BATCH_SIZE = 2000  # 每批最多处理文件数
MAX_FILES = 7500  # 总处理文件数上限，防止误操作

# === 路径配置 ===
PROMPT_NAME = 'prompt6'
PROMPT_FILE = fr"Prompt_Text_New\{PROMPT_NAME}.txt"
BATCH_NAME = "report_2nd_non-postoperative"  # 👈 指定要处理的批次
INPUT_DIR = fr"E:\Data\Reports_Collected\Train\{BATCH_NAME}"
OUTPUT_DIR = fr"E:\Data\Reports_Collected\AI_Conclusion\{PROMPT_NAME}\{BATCH_NAME}"

LOG_FILE = "run_log.txt"
FAIL_FILE = "failed_list.txt"

# === 指定模型禁用 enable_thinking ===
MODELS_DISABLE_THINKING = {
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "tencent/Hunyuan-A13B-Instruct",
    "zai-org/GLM-4.5V",
    "deepseek-ai/DeepSeek-V3.1",
    "Pro/deepseek-ai/DeepSeek-V3.1",
    "Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
}

# === 工具函数 ===
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return {"role": "system", "content": f.read().strip()}

def load_user_prompt(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return {"role": "user", "content": f.read().strip()}

def save_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(f"[{ts}] {msg}\n")

def gather_input_files(folder):
    """递归收集所有 .txt 文件"""
    files = []
    for base, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(".txt"):
                abs_path = os.path.join(base, n)
                rel_path = os.path.relpath(abs_path, folder)
                files.append(rel_path)
    files.sort()
    return files

# === 主处理函数 ===
def generate_report(rel_path, system_message):
    input_path = os.path.join(INPUT_DIR, rel_path)
    output_path = os.path.join(OUTPUT_DIR, rel_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"[⏩] {rel_path} 已存在，跳过")
        log(f"[⏩] {rel_path} 已存在，跳过")
        return True

    user_message = load_user_prompt(input_path)
    messages = [system_message, user_message]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
            }

            if MODEL_NAME in MODELS_DISABLE_THINKING:
                payload["extra_body"] = {"enable_thinking": False}

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)

            if response.status_code != 200:
                print(f"[{rel_path}] ⚠️ HTTP错误 ({response.status_code}) - 第 {attempt} 次尝试")
                log(f"[{rel_path}] ⚠️ HTTP错误 ({response.status_code}) - 第 {attempt} 次尝试; resp={response.text[:200]}")
                time.sleep(RETRY_DELAY)
                continue

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            if not content or not content.strip():
                print(f"[{rel_path}] ⚠️ 内容为空 - 第 {attempt} 次尝试")
                log(f"[{rel_path}] ⚠️ 内容为空 - 第 {attempt} 次尝试")
                time.sleep(RETRY_DELAY)
                continue

            save_text(output_path, content)
            print(f"[✓] {rel_path} 生成成功")
            log(f"[✓] {rel_path} 生成成功")
            return True

        except Exception as e:
            print(f"[{rel_path}] ⚠️ 异常 - 第 {attempt} 次尝试: {e}")
            log(f"[{rel_path}] ⚠️ 异常 - 第 {attempt} 次尝试: {e}")
            time.sleep(RETRY_DELAY)

    print(f"[✗] {rel_path} 所有尝试失败")
    log(f"[✗] {rel_path} 所有尝试失败")
    return False


def process_batch(batch_files, system_message):
    failed_files = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_report, rel, system_message): rel for rel in batch_files}
        for future in tqdm(as_completed(futures), total=len(batch_files), desc="处理进度"):
            rel = futures[future]
            try:
                success = future.result()
                if not success:
                    failed_files.append(rel)
            except Exception as e:
                print(f"[✗] {rel} 执行错误: {e}")
                log(f"[✗] {rel} 执行错误: {e}")
                failed_files.append(rel)
    return failed_files


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        raise RuntimeError(f"输入目录不存在：{INPUT_DIR}")

    system_message = load_system_prompt(PROMPT_FILE)
    all_files = gather_input_files(INPUT_DIR)[4500:MAX_FILES]

    print(f"📂 当前批次：{BATCH_NAME}")
    print(f"共 {len(all_files)} 个输入文件，模型：{MODEL_NAME}")
    log(f"启动任务，共 {len(all_files)} 个文件。模型={MODEL_NAME} 批次={BATCH_NAME}")

    round_no = 1
    while True:
        remaining = [
            rel for rel in all_files
            if not (os.path.exists(os.path.join(OUTPUT_DIR, rel)) and os.path.getsize(os.path.join(OUTPUT_DIR, rel)) > 0)
        ]

        if not remaining:
            print("\n🎉 所有文件均已处理完毕！")
            log(f"批次 {BATCH_NAME} 所有文件均已处理完毕！")
            break

        batch = remaining[:BATCH_SIZE]
        print(f"\n🚀 第 {round_no} 批次：共 {len(batch)} 个文件（剩余 {len(remaining)}）")
        log(f"开始第 {round_no} 批次，{len(batch)} 个文件待处理。")

        failed = process_batch(batch, system_message)

        if failed:
            print(f"❗第 {round_no} 批次有 {len(failed)} 个失败，已记录。")
            # save_text(FAIL_FILE, "\n".join(failed))
        else:
            print(f"✅ 第 {round_no} 批次全部成功。")

        round_no += 1
