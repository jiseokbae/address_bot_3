import asyncio
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI


SYSTEM_PROMPT = """당신은 주소 표기 관계를 판별하는 전문가입니다.

주어진 대화에는 정확히 2개의 [ADDRESS]가 포함되어 있습니다.
당신의 임무는 두 [ADDRESS]의 관계를 아래 셋 중 하나로 분류하는 것입니다.

분류 라벨:
- same:
  두 [ADDRESS]가 같은 주소를 가리킨다.
  예: 같은 주소를 반복 확인, 같은 장소를 다시 말함, 같은 referent를 다른 표현으로 재진술.

- split:
  두 [ADDRESS]가 서로 다른 주소가 아니라, 하나의 주소가 둘로 나뉘어 표현된 것이다.
  예: 앞 주소 조각 + 뒤 상세주소, 같은 완전한 주소를 구성하는 두 부분.

- different:
  두 [ADDRESS]가 서로 다른 주소이다.

중요 규칙:
1. 반드시 same / split / different 중 하나만 선택하세요.
2. 출력은 반드시 JSON 객체 하나만 출력하세요.
3. JSON 형식:
{"label":"same|split|different","reason":"짧은 근거"}
4. reason은 한 문장으로 짧게 작성하세요.
5. JSON 외 다른 텍스트는 출력하지 마세요.
"""


def build_user_prompt(sample: dict) -> str:
    lines = []
    lines.append("다음 대화에서 정확히 2개의 [ADDRESS] 관계를 판별하세요.")
    lines.append("")
    lines.append(f"target_turn_idx: {sample.get('target_turn_idx')}")
    lines.append("dialogue:")

    for turn in sample["dialogue"]:
        lines.append(f"[{turn['turn_idx']}] {turn['speaker']}: {turn['tagged_text']}")

    return "\n".join(lines)


def parse_model_json(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)

    label = data.get("label")
    reason = data.get("reason", "")

    if label not in {"same", "split", "different"}:
        raise ValueError(f"Invalid label: {label}")

    return {
        "label": label,
        "reason": reason,
    }


def relabel_sample_dialogue(sample: dict, label: str) -> dict:
    if label == "same":
        return sample

    replacement_tags = {
        "split": ["[ADDRESS_1]", "[ADDRESS_2]"],
        "different": ["[ADDRESS_A]", "[ADDRESS_B]"],
    }

    if label not in replacement_tags:
        return sample

    tags = replacement_tags[label]
    replaced_count = 0

    new_dialogue = []
    for turn in sample["dialogue"]:
        text = turn["tagged_text"]

        def _replace_one(match):
            nonlocal replaced_count
            if replaced_count < 2:
                tag = tags[replaced_count]
                replaced_count += 1
                return tag
            return match.group(0)

        new_text = re.sub(r"\[ADDRESS\]", _replace_one, text)

        new_turn = dict(turn)
        new_turn["tagged_text"] = new_text
        new_dialogue.append(new_turn)

    new_sample = dict(sample)
    new_sample["dialogue"] = new_dialogue
    return new_sample


def count_sample_address(sample: dict) -> int:
    return sum(turn["tagged_text"].count("[ADDRESS]") for turn in sample["dialogue"])


async def classify_sample_async(
    client: AsyncOpenAI,
    model: str,
    sample: dict,
    temperature: float = 0.0,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample)},
    ]

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
    )

    content = resp.choices[0].message.content
    parsed = parse_model_json(content)

    return {
        "raw_response": content,
        "label": parsed["label"],
        "reason": parsed["reason"],
    }


async def process_one(
    idx: int,
    sample: dict,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    async with semaphore:
        addr_count = count_sample_address(sample)
        if addr_count != 2:
            return {
                "index": idx,
                "ok": False,
                "error": f"Sample address count is not 2: {addr_count}",
                "sample": sample,
            }

        last_error = None

        for attempt in range(max_retries):
            try:
                cls = await classify_sample_async(client, model, sample, temperature=0.0)
                relabeled_sample = relabel_sample_dialogue(sample, cls["label"])

                output_item = dict(relabeled_sample)
                output_item["relation_label"] = cls["label"]
                output_item["relation_reason"] = cls["reason"]
                output_item["raw_model_response"] = cls["raw_response"]

                return {
                    "index": idx,
                    "ok": True,
                    "result": output_item,
                }

            except Exception as e:
                last_error = str(e)
                await asyncio.sleep(0.5 * (attempt + 1))

        return {
            "index": idx,
            "ok": False,
            "error": last_error,
            "sample": sample,
        }


async def save_json(path: str, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def run_pipeline(
    input_path: str,
    output_path: str,
    error_path: str,
    api_key: str,
    api_base: str,
    concurrency: int = 64,
    save_every: int = 500,
):
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
    )

    models = await client.models.list()
    model = models.data[0].id
    print("model:", model)

    dataset = json.loads(Path(input_path).read_text(encoding="utf-8"))
    print("num_samples:", len(dataset))

    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        asyncio.create_task(process_one(i, sample, client, model, semaphore))
        for i, sample in enumerate(dataset)
    ]

    results = []
    errors = []
    done_count = 0
    started_at = time.time()

    for fut in asyncio.as_completed(tasks):
        item = await fut
        done_count += 1

        if item["ok"]:
            results.append((item["index"], item["result"]))
        else:
            errors.append({
                "index": item["index"],
                "error": item["error"],
                "sample": item["sample"],
            })

        if done_count % 100 == 0:
            elapsed = time.time() - started_at
            print(f"[{done_count}/{len(dataset)}] elapsed={elapsed:.1f}s")

        if done_count % save_every == 0:
            sorted_results = [x[1] for x in sorted(results, key=lambda x: x[0])]
            sorted_errors = sorted(errors, key=lambda x: x["index"])

            await save_json(output_path, sorted_results)
            await save_json(error_path, sorted_errors)
            print(f"partial saved at {done_count}")

    sorted_results = [x[1] for x in sorted(results, key=lambda x: x[0])]
    sorted_errors = sorted(errors, key=lambda x: x["index"])

    await save_json(output_path, sorted_results)
    await save_json(error_path, sorted_errors)

    print("saved results:", len(sorted_results))
    print("saved errors :", len(sorted_errors))


if __name__ == "__main__":
    input_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr2.json"
    output_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr2_relabeled.json"
    error_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr2_relabeled_errors.json"

    # input_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_2.json"
    # output_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_2_relabeled.json"
    # error_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_2_relabeled_errors.json"



    api_key = "EMPTY"
    api_base = "http://10.250.128.255:30059/v1"

    asyncio.run(
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            error_path=error_path,
            api_key=api_key,
            api_base=api_base,
            concurrency=256,
            save_every=1000,
        )
    )