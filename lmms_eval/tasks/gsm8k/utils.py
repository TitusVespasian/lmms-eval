from pathlib import Path
import json
import re
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# load config file for additional metadata (e.g., quick_extract flag)
with open(Path(__file__).parent / "gsm8k-format.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        # strip function definitions if any (yaml cannot parse them)
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

# 独立实现 GSM8K 评估逻辑，避免复用 MathVistaEvaluator

import os
import time
from openai import AzureOpenAI, OpenAI


class Gsm8kEvaluator:
    """Evaluator responsible for
    1. 从模型输出中抽取最终数值答案（可选 GPT 辅助）
    2. 归一化为整数/浮点字符串
    3. 判断预测与参考答案是否一致
    """

    API_TYPE = os.getenv("API_TYPE", "openai")

    if API_TYPE == "openai":
        API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
        # openai-python v1 需要 base_url 去掉 /chat/completions 尾巴
        _base = re.sub(r"/chat/completions$", "", API_URL)
        client = OpenAI(base_url=_base, api_key=API_KEY)
        gpt_model = config.get("metadata", {}).get("gpt_eval_model_name", "gpt-4o-mini")
    elif API_TYPE == "azure":
        API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com")
        API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
        API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
        client = AzureOpenAI(azure_endpoint=API_URL, api_version=API_VERSION, api_key=API_KEY)
        gpt_model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

    def __init__(self, quick_extract: bool = False):
        self.quick_extract = quick_extract

    # ---------------------------------------------------------------------
    # GPT wrapper
    # ---------------------------------------------------------------------
    def get_chat_response(self, prompt: str, temperature: float = 0.0, max_tokens: int = 64, patience: int = 5):
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.gpt_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        while patience > 0:
            patience -= 1
            try:
                resp = self.client.chat.completions.create(**payload)
                content = resp.choices[0].message.content.strip()
                if content:
                    return content
            except Exception as e:
                eval_logger.error(e)
                time.sleep(1)
        return ""

    # ---------------------------------------------------------------------
    # 核心抽取逻辑
    # ---------------------------------------------------------------------
    @staticmethod
    def _regex_extract_number(text: str):
        """返回 text 中最后出现的数字字符串"""
        nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
        return nums[-1] if nums else ""

    def extract_answer(self, response: str, quick_extract: bool = False):
        """从模型回复中抽取数字答案"""
        if not response:
            return ""

        # 直接数字
        direct = self._regex_extract_number(response)
        if direct:
            return direct

        # quick extract: The answer is 42.
        if quick_extract or self.quick_extract:
            m = re.search(r"[Tt]he answer is\s+(-?\d+\.?\d*)", response)
            if m:
                return m.group(1)

        # GPT 辅助抽取
        demo_prompt = (
            "请从下列模型回复中抽取最终数值答案，仅输出数字。例如：\n"
            "模型回复：The answer is 14.\n"
            "抽取结果：14\n"
            "----\n"
        )
        prompt = f"{demo_prompt}\n模型回复：{response}\n抽取结果："
        extraction = self.get_chat_response(prompt)
        return self._regex_extract_number(extraction)

    # ---------------------------------------------------------------------
    @staticmethod
    def normalize(extraction: str):
        """统一返回值格式，去掉多余前导零"""
        if extraction == "":
            return None
        try:
            if "." in extraction:
                # 保留原小数位
                return str(float(extraction))
            else:
                return str(int(float(extraction)))
        except Exception:
            return None

    @staticmethod
    def safe_equal(pred, refer):
        try:
            return abs(float(pred) - float(refer)) < 1e-6
        except Exception:
            return False


# 初始化 evaluator 实例
gsm8k_evaluator = Gsm8kEvaluator()


def _extract_gt_answer(answer_text: str):
    """Extract the final numerical answer from the GSM8K `answer` field.

    In GSM8K the ground-truth answer string usually looks like:
    "... #### 42" or "... #### 4.5".
    We take the token after the last "####" marker. Fallback to the last
    number appearing in the string.
    """
    if answer_text is None:
        return None

    answer_text = answer_text.strip()

    # 1. split by the official delimiter
    if "####" in answer_text:
        candidate = answer_text.split("####")[-1].strip()
    else:
        candidate = answer_text

    # 2. take the last number present
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", candidate)
    if numbers:
        return numbers[-1].lstrip("0") if len(numbers[-1]) > 1 else numbers[-1]

    # 3. fallback: return the whole stripped string
    return candidate


def gsm8k_process_results(doc, results):
    """Post-process a single GSM8K sample following MathVista logic.

    Args:
        doc (dict): raw sample from the dataset. Must contain at least
            `question` and `answer` keys.
        results (list[str]): model outputs (length 1).

    Returns:
        dict: a dict with keys `gpt_eval_score` and `submission`, each
        holding identical result information (for compatibility with
        the lmm-eval evaluation framework).
    """
    # model prediction (raw)
    prediction_raw = results[0].strip() if results else ""

    # ground truth answer extraction
    gt_answer = _extract_gt_answer(doc.get("answer", ""))

    # decide answer type & precision heuristically
    if gt_answer is None or gt_answer == "":
        answer_type = "text"
        precision = 0
    elif "." in gt_answer:
        answer_type = "float"
        precision = len(gt_answer.split(".")[-1])
    else:
        answer_type = "integer"
        precision = 0

    problem = {
        "question_type": "free_form",  # GSM8K is open-ended numerical QA
        "answer_type": answer_type,
        "query": doc.get("question", ""),
        "choices": [],
        "answer": gt_answer,
        "precision": precision,
    }

    # extract answer from model response using GPT (or heuristic if quick_extract)
    extraction = gsm8k_evaluator.extract_answer(
        prediction_raw, False
    )

    # normalise extracted answer
    prediction_norm = gsm8k_evaluator.normalize(extraction)

    # evaluate correctness (if test set GT available)
    true_false = (
        gsm8k_evaluator.safe_equal(prediction_norm, problem["answer"])
        if problem["answer"] is not None and problem["answer"] != ""
        else False
    )

    result = {
        "question": doc.get("question", ""),
        "answer": problem["answer"],
        "extraction": extraction,
        "prediction": prediction_norm,
        "true_false": true_false,
    }

    return {
        "gpt_eval_score": result,
        "submission": result,
    }


def gsm8k_aggregate_results(results, args):
    """Aggregate per-sample results into overall accuracy and save json."""
    total = len(results)
    correct = sum(1 for r in results if r["true_false"])
    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0

    # save full predictions for submission / inspection
    path = generate_submission_file("gsm8k_scores.json", args)
    with open(path, "w") as f:
        json.dump({i: r for i, r in enumerate(results)}, f, indent=4)
    eval_logger.info(f"Saved GSM8K results to {path}")

    return accuracy

