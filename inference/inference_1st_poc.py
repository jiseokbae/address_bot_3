import csv
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
from typing import Dict, List, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    FIRST_POC_INPUT_CSV,
    FIRST_POC_OUTPUT_DIR,
    train_output_dir,
)

def parse_csv(file_path):
    results = []
    
    with open(file_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    return results

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(file_path):
    """
    JSONL 파일을 읽어서 딕셔너리 리스트로 반환합니다.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 줄바꿈이나 공백을 제거하고 내용이 있는 경우만 파싱
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"에러: '{file_path}' 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError as e:
        print(f"에러: JSON 파싱 중 오류가 발생했습니다: {e}")
        
    return data

def dialogue_to_input(
    dialogue: List[Dict[str, Any]],
) -> Dict[str, Any]:

    rendered_turns = [
        turn['text']
        for turn in dialogue
    ]

    text = "\n".join(rendered_turns)

    return {
        "text": text,
        "dialogue": dialogue,
    }

def poc_GT_to_inputs(data_path):
    data = parse_csv(data_path)
    tc_inputs = []

    for sample in data :
        dialogue = [{"text" : sample['ITN_NORMALIZED_GT']}]
        tc_input = dialogue_to_input(dialogue)
        tc_inputs.append({
                "NO": int(sample["NO"]),
                "id": sample["TC ID"],
                "음원파일": sample['음원파일'],
                "raw_text": sample["ITN_NORMALIZED_GT"],
                "tc_input": tc_input,
            })
    return tc_inputs


def poc_itn_to_inputs(data_path):
    def read_itn(file_path):
        """
        JSONL 파일을 읽어서 딕셔너리 리스트로 반환합니다.
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    # 줄바꿈이나 공백을 제거하고 내용이 있는 경우만 파싱
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"에러: '{file_path}' 파일을 찾을 수 없습니다.")
        except json.JSONDecodeError as e:
            print(f"에러: JSON 파싱 중 오류가 발생했습니다: {e}")
            
        return data

    data = read_itn(data_path)

    tc_inputs = []
    for sample in data :
        dialogue = [{"text" : sample['address_itn_hyp']}]
        tc_input = dialogue_to_input(dialogue)
        tc_dict = {}
        for k,v in sample.items():
            tc_dict[k] = v
        tc_dict["tc_input"] = tc_input
        tc_inputs.append(tc_dict)
    return tc_inputs


def encode_inference_sample_with_meta(
    sample,
    tokenizer,
    max_length=512,
):
    enc = tokenizer(
        sample["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    sequence_ids = enc.sequence_ids()

    tensor_enc = tokenizer(
        sample["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offset_mapping": enc["offset_mapping"],
        "sequence_ids": sequence_ids,
        "tensor_inputs": tensor_enc,
    }

def get_label(id2label, pred_id: int) -> str:
    if pred_id in id2label:
        return id2label[pred_id]
    if str(pred_id) in id2label:
        return id2label[str(pred_id)]
    raise KeyError(f"pred_id={pred_id} not found in id2label")

def predict_one(
    sample,
    tokenizer,
    model,
    id2label,
    max_length=512,
):
    enc = encode_inference_sample_with_meta(
        sample=sample,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    tensor_inputs = {k: v.to(model.device) for k, v in enc["tensor_inputs"].items()}

    with torch.no_grad():
        outputs = model(**tensor_inputs)
        pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    token_predictions = []
    for tok, pred_id, offset, seq_id in zip(
        tokens,
        pred_ids,
        enc["offset_mapping"],
        enc["sequence_ids"],
    ):
        if seq_id != 0:
            continue
        if offset[0] == offset[1]:
            continue

        token_predictions.append({
            "token": tok,
            # "label": id2label[pred_id],
            "label": get_label(id2label, pred_id),
            "start_idx": offset[0],
            "end_idx": offset[1],
            "text": sample["text"][offset[0]:offset[1]],
        })

    return {
        "text": sample["text"],
        "token_predictions": token_predictions,
    }

FIRST_POC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
from tqdm import tqdm
if __name__ == "__main__":
    for MODEL_TYPE in ["klue"]:#, "uplus"]:
        for ONE in [1000,2000]: 
            for TRI in [3,4]:  
                for QUAD in [3,4]: 
                    for IDX in [1,2,3]: 
                        MODEL_DIR = (
                            train_output_dir(ONE, TRI, QUAD, IDX)
                            / "best_model"
                        )
                        if not os.path.isdir(MODEL_DIR):
                            continue
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
                        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
                        model.eval()

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)

                        poc_path = FIRST_POC_INPUT_CSV
                        out_path = (
                            FIRST_POC_OUTPUT_DIR
                            / f"poc_inference_results_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                        )
                        tc_inputs = poc_GT_to_inputs(poc_path)

                        results = []
                        for sample in tqdm(tc_inputs):
                            result = predict_one(sample['tc_input'], tokenizer, model, model.config.id2label)
                            result_dict = {}
                            for k,v in sample.items():
                                if k != "tc_input":
                                    result_dict[k] = v
                            
                            result_dict["text"] = sample['tc_input']["text"]
                            result_dict["token_predictions"] = result["token_predictions"]
                            results.append(result_dict)

                        results = sorted(results, key=lambda x: x.get("id", x.get("idx", 0)))    
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        del model
                        del tokenizer
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()