import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI


# =========================================================
# CONFIG
# =========================================================

EXPECTED_ADDRESS_COUNT = 4
ALLOWED_GROUPS = {"A", "B", "C", "D"}
ALLOWED_FORMS = {"single", "shallow", "split_1", "split_2"}

# split group이 1개일 때는 기존처럼 [ADDRESS_1], [ADDRESS_2] 사용
# split group이 2개 이상이면 [ADDRESS_A1], [ADDRESS_A2]처럼 group별 split placeholder 사용
USE_LEGACY_SPLIT_PLACEHOLDER_WHEN_SINGLE_SPLIT_GROUP = True


SYSTEM_PROMPT = """당신은 주소 표기 관계를 판별하는 전문가입니다.

주어진 대화에는 정확히 4개의 [ADDRESS]가 포함되어 있습니다.
당신의 임무는 각 [ADDRESS] occurrence가
1) 어떤 실제 주소 그룹(group)에 속하는지
2) 어떤 표현 형태(form)인지
를 판별하는 것입니다.

반드시 아래 규칙을 따르세요.

[판별 대상]
등장 순서대로 4개의 [ADDRESS]를 각각 occurrence 1, 2, 3, 4라고 부릅니다.

[group 정의]
- group은 실제 referent 기준입니다.
- 같은 실제 주소를 가리키면 같은 group을 부여하세요.
- 서로 다른 실제 주소이면 다른 group을 부여하세요.
- 사용할 수 있는 값은 "A", "B", "C", "D" 뿐입니다.
- group 이름은 실제 주소의 종류를 뜻하는 것이 아니라, 같은 주소인지 다른 주소인지 구분하기 위한 임시 기호입니다.

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
3. occurrence는 반드시 4개여야 합니다.
4. occurrence_idx는 반드시 1, 2, 3, 4 순서여야 합니다.
5. split_1 과 split_2 는 반드시 같은 group 안에서만 사용하세요.
6. 어떤 group에서 split_1이 있으면 같은 group에 split_2도 반드시 있어야 합니다.
7. 어떤 group에서 split_2가 있으면 같은 group에 split_1도 반드시 있어야 합니다.
8. 한 group 안에서 split_1은 최대 1개, split_2도 최대 1개만 사용할 수 있습니다.
9. 같은 referent면 같은 group을 사용하세요.
10. 서로 다른 referent면 다른 group을 사용하세요.
11. form은 반드시 single / shallow / split_1 / split_2 중 하나만 사용하세요.
12. 서로 다른 두 주소가 각각 split되어 있으면 서로 다른 group을 사용하세요.
13. reason은 한 문장으로 짧게 작성하세요.

[출력 형식 예시 1: 모두 다른 완결 주소]
{
  "occurrences": [
    {"occurrence_idx": 1, "group": "A", "form": "single"},
    {"occurrence_idx": 2, "group": "B", "form": "single"},
    {"occurrence_idx": 3, "group": "C", "form": "single"},
    {"occurrence_idx": 4, "group": "D", "form": "single"}
  ],
  "reason": "네 occurrence가 서로 다른 완결 주소를 가리킵니다."
}

[출력 형식 예시 2: 하나의 split 주소 + 나머지 주소]
{
  "occurrences": [
    {"occurrence_idx": 1, "group": "A", "form": "single"},
    {"occurrence_idx": 2, "group": "B", "form": "split_1"},
    {"occurrence_idx": 3, "group": "B", "form": "split_2"},
    {"occurrence_idx": 4, "group": "C", "form": "shallow"}
  ],
  "reason": "2번과 3번은 하나의 주소가 나뉜 표현이고 나머지는 별도 주소입니다."
}

[출력 형식 예시 3: 두 개의 split 주소]
{
  "occurrences": [
    {"occurrence_idx": 1, "group": "A", "form": "split_1"},
    {"occurrence_idx": 2, "group": "A", "form": "split_2"},
    {"occurrence_idx": 3, "group": "B", "form": "split_1"},
    {"occurrence_idx": 4, "group": "B", "form": "split_2"}
  ],
  "reason": "1번과 2번, 3번과 4번이 각각 하나의 주소를 이루는 split 표현입니다."
}
"""


def build_user_prompt(sample: dict) -> str:
    lines = []
    lines.append(f"다음 대화에서 정확히 {EXPECTED_ADDRESS_COUNT}개의 [ADDRESS] occurrence를 판별하세요.")
    lines.append("")
    lines.append(f"target_turn_idx: {sample.get('target_turn_idx')}")
    lines.append("dialogue:")

    for turn in sample["dialogue"]:
        lines.append(f"[{turn['turn_idx']}] {turn['speaker']}: {turn['tagged_text']}")

    lines.append("")
    lines.append(
        f"등장 순서대로 [ADDRESS] occurrence 1, 2, 3, 4를 판단하세요."
    )
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

    if len(occurrences) != EXPECTED_ADDRESS_COUNT:
        raise ValueError(
            f"occurrences length must be {EXPECTED_ADDRESS_COUNT}, got {len(occurrences)}"
        )

    parsed_occurrences = []

    for i, item in enumerate(occurrences, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"occurrence {i} must be a dict")

        occurrence_idx = item.get("occurrence_idx")
        group = item.get("group")
        form = item.get("form")

        if occurrence_idx != i:
            raise ValueError(
                f"occurrence_idx must be ordered 1,2,3,4. "
                f"got {occurrence_idx} at position {i}"
            )

        if group not in ALLOWED_GROUPS:
            raise ValueError(f"Invalid group: {group}")

        if form not in ALLOWED_FORMS:
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
    - 같은 group 안에서 split_1이 split_2보다 먼저 나와야 함
    """
    group_to_items: Dict[str, List[dict]] = {}

    for occ in occurrences:
        group_to_items.setdefault(occ["group"], []).append(occ)

    for group, items in group_to_items.items():
        forms = [item["form"] for item in items]

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

        if split_1_count == 1 and split_2_count == 1:
            split_1_idx = next(
                item["occurrence_idx"] for item in items if item["form"] == "split_1"
            )
            split_2_idx = next(
                item["occurrence_idx"] for item in items if item["form"] == "split_2"
            )

            if split_1_idx > split_2_idx:
                raise ValueError(
                    f"group {group}: split_1 must appear before split_2 "
                    f"({split_1_idx} > {split_2_idx})"
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


def get_split_groups(occurrences: List[dict]) -> List[str]:
    """
    split_1/split_2가 존재하는 group 목록.
    예:
    [
      {"group": "A", "form": "split_1"},
      {"group": "A", "form": "split_2"},
      {"group": "B", "form": "split_1"},
      {"group": "B", "form": "split_2"},
    ]
    -> ["A", "B"]
    """
    split_groups = sorted(
        {
            occ["group"]
            for occ in occurrences
            if occ["form"] in {"split_1", "split_2"}
        }
    )
    return split_groups


def make_placeholder(
    group: str,
    form: str,
    split_group_count: int,
) -> str:
    """
    placeholder 생성 규칙.

    single/shallow:
      group 기준 [ADDRESS_A], [ADDRESS_B], ...

    split_1/split_2:
      - split group이 1개뿐이면 기존 호환을 위해 [ADDRESS_1], [ADDRESS_2]
      - split group이 2개 이상이면 충돌 방지를 위해 [ADDRESS_A1], [ADDRESS_A2] 사용

    주의:
    [ADDRESS_A1] 형태는 기존 PLACEHOLDER_PATTERN이
    r"\\[(ADDRESS(?:_[A-Z0-9]+)?)\\]"
    이런 식이면 정상 매칭 가능함.
    """
    if form == "single" or form == "shallow":
        return f"[ADDRESS_{group}]"

    if form == "split_1":
        if (
            USE_LEGACY_SPLIT_PLACEHOLDER_WHEN_SINGLE_SPLIT_GROUP
            and split_group_count == 1
        ):
            return "[ADDRESS_1]"
        return f"[ADDRESS_{group}1]"

    if form == "split_2":
        if (
            USE_LEGACY_SPLIT_PLACEHOLDER_WHEN_SINGLE_SPLIT_GROUP
            and split_group_count == 1
        ):
            return "[ADDRESS_2]"
        return f"[ADDRESS_{group}2]"

    raise ValueError(f"Invalid form: {form}")


def build_address_plan(
    sample: dict,
    occurrences: List[dict],
) -> List[dict]:
    """
    occurrence별 메타정보 + 나중 삽입용 plan 생성.

    placeholder 규칙:
    - single/shallow -> [ADDRESS_A]/[ADDRESS_B]/[ADDRESS_C]/[ADDRESS_D]
    - split_1/split_2
        - split group이 1개면 기존처럼 [ADDRESS_1]/[ADDRESS_2]
        - split group이 2개 이상이면 [ADDRESS_A1]/[ADDRESS_A2]처럼 group별 생성
    """
    positions = extract_occurrence_positions(sample)

    if len(positions) != EXPECTED_ADDRESS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ADDRESS_COUNT} positions, got {len(positions)}"
        )

    split_groups = get_split_groups(occurrences)
    split_group_count = len(split_groups)

    plan = []

    for pos, occ in zip(positions, occurrences):
        group = occ["group"]
        form = occ["form"]

        placeholder = make_placeholder(
            group=group,
            form=form,
            split_group_count=split_group_count,
        )

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
    [ADDRESS] 4개를 등장 순서대로 address_plan의 placeholder로 치환.
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

    if global_occ_idx != EXPECTED_ADDRESS_COUNT:
        raise ValueError(
            f"Relabeled occurrence count mismatch: {global_occ_idx}"
        )

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

        if addr_count != EXPECTED_ADDRESS_COUNT:
            return {
                "index": idx,
                "ok": False,
                "error": (
                    f"Sample address count is not {EXPECTED_ADDRESS_COUNT}: "
                    f"{addr_count}"
                ),
                "sample": sample,
            }

        last_error = None

        for attempt in range(max_retries):
            try:
                cls = await classify_sample_async(
                    client=client,
                    model=model,
                    sample=sample,
                    temperature=0.0,
                )

                address_plan = build_address_plan(
                    sample=sample,
                    occurrences=cls["occurrences"],
                )

                relabeled_sample = relabel_sample_dialogue(
                    sample=sample,
                    address_plan=address_plan,
                )

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    print("expected_address_count:", EXPECTED_ADDRESS_COUNT)
    print("concurrency:", concurrency)

    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        asyncio.create_task(
            process_one(
                idx=i,
                sample=sample,
                client=client,
                model=model,
                semaphore=semaphore,
            )
        )
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
    # input_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr4.json"
    # output_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr4_relabeled.json"
    # error_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr4_relabeled_errors.json"

    input_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_4.json"
    output_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_4_relabeled.json"
    error_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_4_relabeled_errors.json"


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