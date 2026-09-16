import csv
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    SECOND_POC_GT_CSV,
    second_poc_output_dir,
    train_output_dir,
)

# =========================================================
# Config
# =========================================================
TEXT_COLUMN = "ITN_NORMALIZED_GT"
WINDOW_SIZE = 6
MAX_LENGTH = 512

# 디버깅할 거면 True, 최종 결과만 작게 저장하려면 False
KEEP_TOKEN_PREDICTIONS = True


# =========================================================
# IO
# =========================================================
def parse_csv(file_path: str, required_column: str = TEXT_COLUMN) -> List[Dict[str, Any]]:
    encodings = ["utf-8-sig", "utf-8", "cp949", "ms949", "euc-kr", "utf-16"]

    last_error = None

    for enc in encodings:
        try:
            with open(file_path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows and required_column not in rows[0]:
                last_error = ValueError(
                    f"encoding={enc} 로 읽었지만 required_column={required_column} 없음. "
                    f"columns={list(rows[0].keys())}"
                )
                continue

            print(f"[CSV OK] encoding={enc}, rows={len(rows)}")
            return rows

        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 읽기 실패: {file_path}, last_error={last_error}")


def save_json(data: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# Dialogue parsing
# =========================================================
def split_speaker_line(line: str, row_idx: int, raw_turn_idx: int) -> Optional[Dict[str, str]]:
    """
    입력:
        상담사: ...
        고객: ...

    출력:
        {
          "speaker": "상담사",
          "text": "..."
        }

    빈 줄은 제거.
    ':' 없는 줄은 깨진 케이스로 보고 에러 발생.
    """
    line = line.strip()
    if not line:
        return None

    if ":" in line:
        speaker, text = line.split(":", 1)
    elif "：" in line:
        speaker, text = line.split("：", 1)
    else:
        raise ValueError(
            f"[row_idx={row_idx}, raw_turn_idx={raw_turn_idx}] ':' 없는 줄 발견: {line}"
        )

    speaker = speaker.strip()
    text = text.strip()

    # "고객:"처럼 내용이 비어 있으면 제거
    if not text:
        return None

    return {
        "speaker": speaker,
        "text": text,
        "raw_line": line,
    }


def dialogue_to_turns(row_idx: int, dialogue: str) -> List[Dict[str, Any]]:
    turns = []

    for raw_turn_idx, line in enumerate(str(dialogue).splitlines()):
        parsed = split_speaker_line(line, row_idx=row_idx, raw_turn_idx=raw_turn_idx)
        if parsed is None:
            continue

        parsed["turn_idx"] = len(turns)
        parsed["raw_turn_idx"] = raw_turn_idx
        turns.append(parsed)

    return turns


def turns_to_text(turns: List[Dict[str, Any]]) -> str:
    """
    상담사:/고객: 제거된 전체 text.
    top-level text 저장용.
    """
    return "\n".join(turn["text"] for turn in turns)


# =========================================================
# Windowing
# =========================================================
def build_window(turns: List[Dict[str, Any]], start_turn_idx: int, window_size: int = WINDOW_SIZE):
    """
    window text는 speaker 제거된 turn text를 \n으로 연결.
    각 turn의 window 내부 start/end offset도 같이 계산.
    """
    window_turns = turns[start_turn_idx:start_turn_idx + window_size]

    parts = []
    turn_spans = {}

    cursor = 0
    for local_idx, turn in enumerate(window_turns):
        if local_idx > 0:
            cursor += 1  # '\n'

        start_idx = cursor
        text = turn["text"]
        cursor += len(text)
        end_idx = cursor

        turn_spans[turn["turn_idx"]] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

        parts.append(text)

    window_text = "\n".join(parts)

    return {
        "text": window_text,
        "start_turn_idx": window_turns[0]["turn_idx"],
        "end_turn_idx": window_turns[-1]["turn_idx"],
        "turn_spans": turn_spans,
    }


def make_windows(turns: List[Dict[str, Any]], window_size: int = WINDOW_SIZE):
    """
    확정 규칙:
    - 6턴 이하면 한 번만 inference
    - 6턴 초과면 0~5, 1~6, 2~7, ... sliding window
    - 첫 window는 0~5턴 전부 채택
    - 이후 window는 마지막 turn만 채택
    """
    n_turns = len(turns)

    if n_turns == 0:
        return []

    if n_turns <= window_size:
        window = build_window(turns, 0, window_size=window_size)
        window["target_turn_indices"] = [turn["turn_idx"] for turn in turns]
        return [window]

    windows = []
    for start in range(0, n_turns - window_size + 1):
        window = build_window(turns, start, window_size=window_size)

        if start == 0:
            # 첫 window: 0~5턴 전부 사용
            window["target_turn_indices"] = list(range(0, window_size))
        else:
            # 이후 window: 마지막 turn만 사용
            window["target_turn_indices"] = [start + window_size - 1]

        windows.append(window)

    return windows


# =========================================================
# Inference
# =========================================================
def get_label(id2label, pred_id: int) -> str:
    if pred_id in id2label:
        return id2label[pred_id]
    if str(pred_id) in id2label:
        return id2label[str(pred_id)]
    raise KeyError(f"pred_id={pred_id} not found in id2label")


def predict_window(
    text: str,
    tokenizer,
    model,
    id2label,
    max_length: int = MAX_LENGTH,
) -> List[Dict[str, Any]]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    sequence_ids = enc.sequence_ids()

    tensor_enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )

    tensor_inputs = {k: v.to(model.device) for k, v in tensor_enc.items()}

    with torch.no_grad():
        outputs = model(**tensor_inputs)
        pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    token_predictions = []
    for tok, pred_id, offset, seq_id in zip(
        tokens,
        pred_ids,
        enc["offset_mapping"],
        sequence_ids,
    ):
        if seq_id != 0:
            continue
        if offset[0] == offset[1]:
            continue

        start_idx, end_idx = offset

        token_predictions.append({
            "token": tok,
            "label": get_label(id2label, pred_id),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text": text[start_idx:end_idx],
        })

    return token_predictions


def map_token_to_turn(
    pred: Dict[str, Any],
    window: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    window 기준 token offset을 turn 기준 offset으로 변환.
    반환되는 start_idx/end_idx는 해당 turn text 내부 기준.
    """
    token_start = pred["start_idx"]
    token_end = pred["end_idx"]

    for turn_idx, span in window["turn_spans"].items():
        turn_start = span["start_idx"]
        turn_end = span["end_idx"]

        if token_start >= turn_start and token_end <= turn_end:
            local_start = token_start - turn_start
            local_end = token_end - turn_start

            return {
                "turn_idx": turn_idx,
                "token": pred["token"],
                "label": pred["label"],
                "start_idx": local_start,
                "end_idx": local_end,
            }

    return None


def predict_turns(
    turns: List[Dict[str, Any]],
    tokenizer,
    model,
    id2label,
    max_length: int = MAX_LENGTH,
) -> List[Dict[str, Any]]:
    """
    최종 결과는 turn 단위.
    각 turn 안에 token_predictions를 유지한다.

    token_predictions의 start_idx/end_idx는 해당 turn["text"] 기준이다.
    """

    turn_outputs = [
        {
            "turn_idx": turn["turn_idx"],
            "raw_turn_idx": turn["raw_turn_idx"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "raw_line": turn["raw_line"],
            "source_window": None,
            "token_predictions": [],
        }
        for turn in turns
    ]

    windows = make_windows(turns, window_size=WINDOW_SIZE)

    for window in windows:
        target_turn_set = set(window["target_turn_indices"])

        # 이 window가 담당하는 turn들에 source_window 먼저 기록
        for target_turn_idx in target_turn_set:
            turn_outputs[target_turn_idx]["source_window"] = {
                "start_turn_idx": window["start_turn_idx"],
                "end_turn_idx": window["end_turn_idx"],
            }

        window_token_predictions = predict_window(
            text=window["text"],
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            max_length=max_length,
        )

        for pred in window_token_predictions:
            mapped = map_token_to_turn(pred, window)

            if mapped is None:
                continue

            turn_idx = mapped["turn_idx"]

            # 중복 방지:
            # 첫 window는 0~5 전부 사용
            # 이후 window는 마지막 turn만 사용
            if turn_idx not in target_turn_set:
                continue

            turn_text = turn_outputs[turn_idx]["text"]

            turn_outputs[turn_idx]["token_predictions"].append({
                "token": mapped["token"],
                "label": mapped["label"],
                "start_idx": mapped["start_idx"],
                "end_idx": mapped["end_idx"],
                "text": turn_text[mapped["start_idx"]:mapped["end_idx"]],

                # 디버깅용. 필요 없으면 빼도 됨.
                "window_start_idx": pred["start_idx"],
                "window_end_idx": pred["end_idx"],
            })

    for turn in turn_outputs:
        turn["token_predictions"] = sorted(
            turn["token_predictions"],
            key=lambda x: (x["start_idx"], x["end_idx"]),
        )

    return turn_outputs

# =========================================================
# Decode BIO to entities
# =========================================================
def parse_bio_label(label: str) -> Tuple[str, Optional[str]]:
    if label == "O":
        return "O", None
    if "-" not in label:
        return "O", None

    prefix, slot = label.split("-", 1)
    if prefix not in {"B", "I"}:
        return "O", None

    return prefix, slot


def flush_entity(current_entity: Optional[Dict[str, Any]], entities: List[Dict[str, Any]]):
    if current_entity is not None:
        entities.append(current_entity)


def decode_token_predictions_to_entities(
    token_predictions: List[Dict[str, Any]],
    text: str,
) -> List[Dict[str, Any]]:
    entities = []
    current_entity = None

    for pred in token_predictions:
        label = pred["label"]
        start_idx = pred["start_idx"]
        end_idx = pred["end_idx"]

        prefix, slot = parse_bio_label(label)

        if prefix == "O":
            flush_entity(current_entity, entities)
            current_entity = None
            continue

        if prefix == "B":
            flush_entity(current_entity, entities)
            current_entity = {
                "slot": slot,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
            continue

        # prefix == "I"
        if current_entity is not None and current_entity["slot"] == slot:
            current_entity["end_idx"] = end_idx
        else:
            # I로 시작해도 새 entity로 살림
            flush_entity(current_entity, entities)
            current_entity = {
                "slot": slot,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }

    flush_entity(current_entity, entities)

    for ent in entities:
        ent["text"] = text[ent["start_idx"]:ent["end_idx"]]

    return entities


# =========================================================
# Entity normalization
# =========================================================
HYPHEN_CONNECTOR_PATTERN = r"(?:다시|대시|데시|dash|에|의)"
CODE_TOKEN_PATTERN = r"(?:[A-Za-z]{1,3}\d{0,4}|\d{1,4}[A-Za-z]{0,3})"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_unit_suffix(text: str) -> str:
    """
    101동 -> 101
    3층 -> 3
    7호 -> 7
    B-7동 -> B-7
    C-19호 -> C-19
    """
    return re.sub(r"(동|층|호)$", "", text)


def restore_hyphen_for_dong_ho(text: str) -> str:
    text = normalize_space(text)

    # C 대시 7호 -> C-7호
    p1 = re.compile(
        rf"({CODE_TOKEN_PATTERN})\s*{HYPHEN_CONNECTOR_PATTERN}\s*(\d{{1,4}})(동|호)(?=$|\s)"
    )
    text = p1.sub(r"\1-\2\3", text)

    # 7 대시 C호 -> 7-C호
    p2 = re.compile(
        rf"(\d{{1,4}})\s*{HYPHEN_CONNECTOR_PATTERN}\s*([A-Za-z]{{1,3}})(동|호)(?=$|\s)"
    )
    text = p2.sub(r"\1-\2\3", text)

    return text


def restore_hyphen_for_building(text: str) -> str:
    text = normalize_space(text)

    pattern = re.compile(
        rf"({CODE_TOKEN_PATTERN})\s*{HYPHEN_CONNECTOR_PATTERN}\s*({CODE_TOKEN_PATTERN})"
    )
    return pattern.sub(r"\1-\2", text)


def normalize_entity_text(slot: str, text: str) -> str:
    text = normalize_space(text)

    if slot in {"동명칭", "호명칭"}:
        text = restore_hyphen_for_dong_ho(text)
    elif slot == "건물명":
        text = restore_hyphen_for_building(text)

    if slot in {"동명칭", "층명칭", "호명칭"}:
        text = strip_unit_suffix(text)

    return text

def decode_turn_outputs(turn_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    turn 단위 결과를 최종 output 형식으로 변환한다.

    최종 output에는 token_predictions를 남기지 않는다.
    token_predictions는 entities 생성용으로만 사용한다.
    """
    decoded_turns = []

    for turn in turn_outputs:
        text = turn["text"]
        token_predictions = turn.get("token_predictions", [])

        entities = decode_token_predictions_to_entities(
            token_predictions=token_predictions,
            text=text,
        )

        for ent in entities:
            ent["normalized_text"] = normalize_entity_text(ent["slot"], ent["text"])

        new_turn = {
            "turn_idx": turn["turn_idx"],
            "raw_turn_idx": turn["raw_turn_idx"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "raw_line": turn["raw_line"],
            "source_window": turn.get("source_window"),
            "entities": entities,
        }

        decoded_turns.append(new_turn)

    return decoded_turns
# =========================================================
# POC data conversion
# =========================================================
def poc_GT_to_inputs(data_path: str) -> List[Dict[str, Any]]:
    rows = parse_csv(data_path, required_column=TEXT_COLUMN)
    tc_inputs = []

    for row_idx, row in enumerate(rows):
        dialogue = row[TEXT_COLUMN]
        turns = dialogue_to_turns(row_idx=row_idx, dialogue=dialogue)

        meta = {k: v for k, v in row.items() if k != TEXT_COLUMN}

        tc_inputs.append({
            **meta,
            "row_idx": row_idx,
            "raw_text": dialogue,
            "text": turns_to_text(turns),
            "turns": turns,
        })

    return tc_inputs

def infer_one_tc(
    sample: Dict[str, Any],
    tokenizer,
    model,
    id2label,
) -> Dict[str, Any]:
    # 1. 내부적으로 turn별 token_predictions 생성
    turn_outputs = predict_turns(
        turns=sample["turns"],
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        max_length=MAX_LENGTH,
    )

    # 2. token_predictions -> entities 변환
    # 3. 최종 output에서는 token_predictions 제거
    turn_outputs = decode_turn_outputs(turn_outputs)

    result = {}

    for k, v in sample.items():
        if k == "turns":
            continue
        result[k] = v

    result["turns"] = turn_outputs
    return result

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    DIR_IDX = "2"


    poc_path = SECOND_POC_GT_CSV
    out_dir = second_poc_output_dir(DIR_IDX)
    out_dir.mkdir(parents=True, exist_ok=True)

    tc_inputs = poc_GT_to_inputs(poc_path)
    
    for MODEL_TYPE in ["klue"]:
        for ONE in [1000, 2000]:
            for TRI in [3, 4]:
                for QUAD in [3, 4]:
                    for IDX in [1, 2, 3]:
                        for CKPT in [13932, 16344]:
                            MODEL_DIR = (
                                train_output_dir(ONE, TRI, QUAD, IDX)
                                / f"checkpoint-{CKPT}"
                            )

                            if not os.path.isdir(MODEL_DIR):
                                print(f"[SKIP] model dir not found: {MODEL_DIR}")
                                continue

                            print(f"[LOAD] {MODEL_DIR}")

                            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
                            model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
                            model.eval()

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model.to(device)

                            results = []

                            for sample in tqdm(tc_inputs, desc=f"{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}"):
                                result = infer_one_tc(
                                    sample=sample,
                                    tokenizer=tokenizer,
                                    model=model,
                                    id2label=model.config.id2label,
                                )
                                results.append(result)

                            out_path = out_dir / (
                                f"poc_inference_results_turn_decoded_"
                                f"{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                            )

                            save_json(results, str(out_path))
                            print(f"[SAVED] {out_path}")

                            del model
                            del tokenizer

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()