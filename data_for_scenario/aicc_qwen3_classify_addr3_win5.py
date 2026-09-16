import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import AsyncOpenAI


SYSTEM_PROMPT = """당신은 주소 표기 관계를 판별하는 전문가입니다.

주어진 대화에는 정확히 3개의 [ADDRESS]가 포함되어 있습니다.
당신의 임무는 각 [ADDRESS] occurrence가
1) 어떤 실제 주소 그룹(group)에 속하는지
2) 어떤 표현 형태(form)인지
를 판별하는 것입니다.

반드시 아래 규칙을 따르세요.

[판별 대상]
등장 순서대로 3개의 [ADDRESS]를 각각 occurrence 1, 2, 3이라고 부릅니다.

[group 정의]
- group은 실제 referent 기준입니다.
- 같은 실제 주소를 가리키면 같은 group을 부여하세요.
- 서로 다른 실제 주소이면 다른 group을 부여하세요.
- 사용할 수 있는 값은 "A", "B", "C" 뿐입니다.

[form 정의]
각 occurrence의 form은 아래 넷 중 하나입니다.

- "single":
  완결된 주소 표현이다.
  독립적으로 하나의 주소로 볼 수 있다.

- "shallow":
  축약되거나 일부 정보만 있는 주소 표현이다.
  독립 referent는 맞지만 표현이 얕다.

- "split_1":
  하나의 주소가 둘로 나뉘어 표현될 때 앞부분이다.

- "split_2":
  하나의 주소가 둘로 나뉘어 표현될 때 뒷부분이다.

[중요 규칙]
1. 출력은 반드시 JSON 객체 하나만 출력하세요.
2. JSON 외의 텍스트는 절대 출력하지 마세요.
3. occurrence는 반드시 3개여야 합니다.
4. occurrence_idx는 반드시 1, 2, 3 순서여야 합니다.
5. split_1 과 split_2 는 반드시 같은 group 안에서만 사용하세요.
6. 어떤 group에서 split_1이 있으면 같은 group에 split_2도 반드시 있어야 합니다.
7. 어떤 group에서 split_2가 있으면 같은 group에 split_1도 반드시 있어야 합니다.
8. 같은 referent면 같은 group을 사용하세요.
9. 서로 다른 referent면 다른 group을 사용하세요.
10. form은 반드시 single / shallow / split_1 / split_2 중 하나만 사용하세요.
11. reason은 한 문장으로 짧게 작성하세요.

[출력 형식]
{
  "occurrences": [
    {"occurrence_idx": 1, "group": "A", "form": "single"},
    {"occurrence_idx": 2, "group": "B", "form": "split_1"},
    {"occurrence_idx": 3, "group": "B", "form": "split_2"}
  ],
  "reason": "짧은 근거 한 문장"
}
"""


def build_user_prompt(sample: dict) -> str:
    lines = []
    lines.append("다음 대화에서 정확히 3개의 [ADDRESS] occurrence를 판별하세요.")
    lines.append("")
    lines.append(f"target_turn_idx: {sample.get('target_turn_idx')}")
    lines.append("dialogue:")

    for turn in sample["dialogue"]:
        lines.append(f"[{turn['turn_idx']}] {turn['speaker']}: {turn['tagged_text']}")

    lines.append("")
    lines.append("등장 순서대로 [ADDRESS] occurrence 1, 2, 3을 판단하세요.")
    return "\n".join(lines)


def count_sample_address(sample: dict) -> int:
    return sum(turn["tagged_text"].count("[ADDRESS]") for turn in sample["dialogue"])


def parse_model_json(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)

    occurrences = data.get("occurrences")
    reason = data.get("reason", "")

    if not isinstance(occurrences, list):
        raise ValueError("occurrences must be a list")
    if len(occurrences) != 3:
        raise ValueError(f"occurrences length must be 3, got {len(occurrences)}")

    allowed_groups = {"A", "B", "C"}
    allowed_forms = {"single", "shallow", "split_1", "split_2"}

    parsed_occurrences = []
    for i, item in enumerate(occurrences, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"occurrence {i} must be a dict")

        occurrence_idx = item.get("occurrence_idx")
        group = item.get("group")
        form = item.get("form")

        if occurrence_idx != i:
            raise ValueError(
                f"occurrence_idx must be ordered 1,2,3. got {occurrence_idx} at position {i}"
            )
        if group not in allowed_groups:
            raise ValueError(f"Invalid group: {group}")
        if form not in allowed_forms:
            raise ValueError(f"Invalid form: {form}")

        parsed_occurrences.append(
            {
                "occurrence_idx": occurrence_idx,
                "group": group,
                "form": form,
            }
        )

    validate_occurrence_plan(parsed_occurrences)

    return {
        "occurrences": parsed_occurrences,
        "reason": reason,
    }


def validate_occurrence_plan(occurrences: List[dict]) -> None:
    """
    split 규칙 검증:
    - split_1 / split_2 는 같은 group 안에서만 쌍으로 존재해야 함
    - 한 group에 split_1, split_2 각각 최대 1개
    """
    group_to_forms: Dict[str, List[str]] = {}

    for occ in occurrences:
        group_to_forms.setdefault(occ["group"], []).append(occ["form"])

    for group, forms in group_to_forms.items():
        split_1_count = forms.count("split_1")
        split_2_count = forms.count("split_2")

        if split_1_count != split_2_count:
            raise ValueError(
                f"group {group}: split_1 and split_2 counts do not match "
                f"({split_1_count} vs {split_2_count})"
            )

        if split_1_count > 1 or split_2_count > 1:
            raise ValueError(
                f"group {group}: split_1/split_2 may appear at most once each"
            )


def extract_occurrence_positions(sample: dict) -> List[dict]:
    """
    대화 전체에서 [ADDRESS] occurrence의 위치를 등장 순서대로 추출.
    각 occurrence마다:
    - occurrence_idx
    - turn_idx
    - speaker
    - nth_in_turn
    """
    positions = []
    occurrence_idx = 0

    for turn in sample["dialogue"]:
        text = turn["tagged_text"]
        matches = list(re.finditer(r"\[ADDRESS\]", text))

        for nth_in_turn, _ in enumerate(matches, start=1):
            occurrence_idx += 1
            positions.append(
                {
                    "occurrence_idx": occurrence_idx,
                    "turn_idx": turn["turn_idx"],
                    "speaker": turn["speaker"],
                    "nth_in_turn": nth_in_turn,
                }
            )

    return positions


def build_address_plan(
    sample: dict,
    occurrences: List[dict],
) -> List[dict]:
    """
    occurrence별 메타정보 + 나중 삽입용 plan 생성.
    placeholder는 아래 규칙:
    - split_1 -> [ADDRESS_1]
    - split_2 -> [ADDRESS_2]
    - single/shallow -> group 기준 [ADDRESS_A]/[ADDRESS_B]/[ADDRESS_C]
    """
    positions = extract_occurrence_positions(sample)

    if len(positions) != 3:
        raise ValueError(f"Expected 3 positions, got {len(positions)}")

    plan = []
    for pos, occ in zip(positions, occurrences):
        group = occ["group"]
        form = occ["form"]

        if form == "split_1":
            placeholder = "[ADDRESS_1]"
        elif form == "split_2":
            placeholder = "[ADDRESS_2]"
        else:
            placeholder = f"[ADDRESS_{group}]"

        plan.append(
            {
                "occurrence_idx": occ["occurrence_idx"],
                "turn_idx": pos["turn_idx"],
                "speaker": pos["speaker"],
                "nth_in_turn": pos["nth_in_turn"],
                "group": group,
                "form": form,
                "placeholder": placeholder,
            }
        )

    return plan


def relabel_sample_dialogue(sample: dict, address_plan: List[dict]) -> dict:
    """
    [ADDRESS] 3개를 등장 순서대로 address_plan의 placeholder로 치환.
    """
    placeholder_by_occ_idx = {
        item["occurrence_idx"]: item["placeholder"] for item in address_plan
    }

    global_occ_idx = 0
    new_dialogue = []

    for turn in sample["dialogue"]:
        text = turn["tagged_text"]

        def _replace_one(match):
            nonlocal global_occ_idx
            global_occ_idx += 1
            return placeholder_by_occ_idx[global_occ_idx]

        new_text = re.sub(r"\[ADDRESS\]", _replace_one, text)

        new_turn = dict(turn)
        new_turn["tagged_text"] = new_text
        new_dialogue.append(new_turn)

    new_sample = dict(sample)
    new_sample["dialogue"] = new_dialogue
    return new_sample


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
        "occurrences": parsed["occurrences"],
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
        if addr_count != 3:
            return {
                "index": idx,
                "ok": False,
                "error": f"Sample address count is not 3: {addr_count}",
                "sample": sample,
            }

        last_error = None

        for attempt in range(max_retries):
            try:
                cls = await classify_sample_async(client, model, sample, temperature=0.0)
                address_plan = build_address_plan(sample, cls["occurrences"])
                relabeled_sample = relabel_sample_dialogue(sample, address_plan)

                output_item = dict(relabeled_sample)
                output_item["address_plan"] = address_plan
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
            errors.append(
                {
                    "index": item["index"],
                    "error": item["error"],
                    "sample": item["sample"],
                }
            )

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
    input_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr3.json"
    output_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr3_relabeled.json"
    error_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr3_relabeled_errors.json"

    # input_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_3.json"
    # output_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_3_relabeled.json"
    # error_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_3_relabeled_errors.json"


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