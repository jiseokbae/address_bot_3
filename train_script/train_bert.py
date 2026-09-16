import os
import random
import pickle
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
import torch.nn as nn

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import resolve_project_path

cfg = load_yaml(args.config)
cfg = resolve_yaml_templates(cfg)

cfg["data"]["dir"] = str(
    resolve_project_path(cfg["data"]["dir"])
)

cfg["data"]["negative_path"] = str(
    resolve_project_path(cfg["data"]["negative_path"])
)

cfg["output"]["output_dir"] = str(
    resolve_project_path(cfg["output"]["output_dir"])
)

# =========================================================
# utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class SafeFormatDict(dict):
    def __missing__(self, key):
        raise KeyError(f"Missing format variable in yaml: {key}")


def build_format_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    yaml의 vars 섹션을 format 변수로 사용.

    예:
      vars:
        oneline: 1000
        triple: 3

      data:
        dir: ".../oneline{oneline}_triple{triple}"
    """
    variables = {}

    if "vars" in cfg:
        if not isinstance(cfg["vars"], dict):
            raise TypeError("cfg['vars'] must be dict")
        variables.update(cfg["vars"])

    return variables


def resolve_yaml_templates(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    YAML 문자열 안의 {변수명}을 cfg["vars"] 값으로 치환한다.
    """

    variables = SafeFormatDict(build_format_vars(cfg))

    def resolve_obj(obj):
        if isinstance(obj, dict):
            return {
                key: resolve_obj(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                resolve_obj(value)
                for value in obj
            ]

        if isinstance(obj, str):
            return obj.format_map(variables)

        return obj

    return resolve_obj(cfg)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def check_label_maps_same(*datasets):
    if not datasets:
        raise ValueError("datasets must not be empty")

    base = datasets[0]
    base_label_list = base["label_list"]
    base_label2id = base["label2id"]
    base_id2label = base["id2label"]

    for i, ds in enumerate(datasets[1:], start=1):
        if ds["label_list"] != base_label_list:
            raise ValueError(f"label_list mismatch at dataset index {i}")

        if ds["label2id"] != base_label2id:
            raise ValueError(f"label2id mismatch at dataset index {i}")

        if ds["id2label"] != base_id2label:
            raise ValueError(f"id2label mismatch at dataset index {i}")

    return base_label_list, base_label2id, base_id2label


def sample_negative_to_match_positive(
    positive_rows: List[Dict[str, Any]],
    negative_rows: List[Dict[str, Any]],
    seed: int = 7,
) -> List[Dict[str, Any]]:
    pos_n = len(positive_rows)
    neg_n = len(negative_rows)

    if neg_n == 0:
        raise ValueError("negative rows are empty")

    # negative가 positive보다 많거나 같으면 positive 수에 맞게 downsampling
    if neg_n >= pos_n:
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(neg_n), pos_n)
        sampled_indices.sort()
        sampled_negative = [negative_rows[i] for i in sampled_indices]
        return sampled_negative

    # negative가 더 적으면 전체 사용
    return negative_rows


def split_train_valid(
    rows: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 7,
):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    n_val = int(len(rows) * val_ratio)
    val_indices = set(indices[:n_val])

    train_rows = []
    valid_rows = []

    for i, row in enumerate(rows):
        if i in val_indices:
            valid_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, valid_rows


def print_dataset_stats(
    name: str,
    rows: List[Dict[str, Any]],
):
    print(f"\n===== {name} stats =====")
    print(f"num_rows: {len(rows)}")

    add_type_counter = {}
    refer_counter = {}

    for row in rows:
        add_type = row.get("add_type", "unknown")
        refer = row.get("refer", "unknown")

        add_type_counter[add_type] = add_type_counter.get(add_type, 0) + 1
        refer_counter[refer] = refer_counter.get(refer, 0) + 1

    print("add_type_counts:", add_type_counter)
    print("refer_counts:", refer_counter)


# =========================================================
# dataset
# =========================================================

class TokenClsDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        item = {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }

        if "token_type_ids" in row:
            item["token_type_ids"] = row["token_type_ids"]

        return item


# =========================================================
# weighted loss
# =========================================================

def build_label_weights(
    label_list: List[str],
    o_weight: float = 0.9,
    other_weight: float = 1.0,
) -> torch.Tensor:
    weights = []

    for label in label_list:
        if label == "O":
            weights.append(o_weight)
        else:
            weights.append(other_weight)

    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            class_weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=-100,
            )

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# =========================================================
# metric
# =========================================================

def compute_token_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    valid_mask = labels != -100
    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_labels) == 0:
        return {
            "token_acc": 0.0,
        }

    token_acc = (valid_preds == valid_labels).mean().item()

    return {
        "token_acc": float(token_acc),
    }


# =========================================================
# arg / config
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to yaml config file",
    )
    return parser.parse_args()


def normalize_id2label(id2label):
    if all(isinstance(k, str) for k in id2label.keys()):
        return {int(k): v for k, v in id2label.items()}
    return id2label


def require_config_key(cfg: Dict[str, Any], path: str):
    """
    path 예:
      data.positive_path
      output.output_dir
    """
    cur = cfg
    parts = path.split(".")

    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: {path}")
        cur = cur[part]

    return cur

def find_one_pkl_by_keyword(data_dir: str, keyword: str) -> str:
    """
    data_dir 안에서 keyword가 파일명에 들어간 pkl을 정확히 1개 찾는다.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"data.dir does not exist: {data_dir}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"data.dir is not directory: {data_dir}")

    matches = sorted([
        str(path)
        for path in data_path.glob("*.pkl")
        if keyword in path.name
    ])

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No pkl found for keyword='{keyword}' in dir={data_dir}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple pkls found for keyword='{keyword}' in dir={data_dir}: {matches}"
        )

    return matches[0]


def build_data_paths_from_dir(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    positive/triple/quadra/oneline은 data.dir에서 자동 탐색.
    negative만 data.negative_path를 직접 사용.
    """
    data_dir = require_config_key(cfg, "data.dir")
    negative_path = require_config_key(cfg, "data.negative_path")

    paths = {
        "positive": find_one_pkl_by_keyword(data_dir, "positive"),
        "negative": negative_path,
        "triple": find_one_pkl_by_keyword(data_dir, "triple"),
        "quadra": find_one_pkl_by_keyword(data_dir, "quadra"),
        "oneline_single": find_one_pkl_by_keyword(data_dir, "oneline_single"),
        "oneline_split": find_one_pkl_by_keyword(data_dir, "oneline_split"),
    }

    return paths
# =========================================================
# main
# =========================================================

def main():

    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = resolve_yaml_templates(cfg)

    # =========================================================
    # config
    # =========================================================

    bert_path = require_config_key(cfg, "bert_path")

    data_paths = build_data_paths_from_dir(cfg)

    positive_path = data_paths["positive"]
    negative_path = data_paths["negative"]
    triple_path = data_paths["triple"]
    quadra_path = data_paths["quadra"]
    oneline_single_path = data_paths["oneline_single"]
    oneline_split_path = data_paths["oneline_split"]

    print("\n===== resolved data paths =====")
    for name, path in data_paths.items():
        print(f"{name:16s}: {path}")

    output_dir = require_config_key(cfg, "output.output_dir")

    seed = require_config_key(cfg, "train.seed")
    val_ratio = require_config_key(cfg, "train.val_ratio")

    train_batch_size = require_config_key(cfg, "train.train_batch_size")
    eval_batch_size = require_config_key(cfg, "train.eval_batch_size")
    num_epochs = require_config_key(cfg, "train.num_epochs")
    learning_rate = require_config_key(cfg, "train.learning_rate")
    lr_scheduler_type = require_config_key(cfg, "train.lr_scheduler_type")
    weight_decay = require_config_key(cfg, "train.weight_decay")
    warmup_ratio = require_config_key(cfg, "train.warmup_ratio")
    logging_steps = require_config_key(cfg, "train.logging_steps")
    save_total_limit = require_config_key(cfg, "train.save_total_limit")

    o_weight = require_config_key(cfg, "loss.o_weight")
    other_weight = require_config_key(cfg, "loss.other_weight")

    ensure_dir(output_dir)
    set_seed(seed)

    print("\n===== loaded config =====")
    print(yaml.dump(cfg, allow_unicode=True, sort_keys=False))

    # =========================================================
    # 1) load tokenized pkl
    # =========================================================

    positive_obj = read_pkl(positive_path)
    negative_obj = read_pkl(negative_path)
    triple_obj = read_pkl(triple_path)
    quadra_obj = read_pkl(quadra_path)
    oneline_single_obj = read_pkl(oneline_single_path)
    oneline_split_obj = read_pkl(oneline_split_path)

    label_list, label2id, id2label = check_label_maps_same(
        positive_obj,
        negative_obj,
        triple_obj,
        quadra_obj,
        oneline_single_obj,
        oneline_split_obj,
    )

    positive_rows = positive_obj["data"]
    negative_rows = negative_obj["data"]
    triple_rows = triple_obj["data"]
    quadra_rows = quadra_obj["data"]
    oneline_single_rows = oneline_single_obj["data"]
    oneline_split_rows = oneline_split_obj["data"]

    print_dataset_stats("positive(full)", positive_rows)
    print_dataset_stats("negative(full)", negative_rows)
    print_dataset_stats("triple(full)", triple_rows)
    print_dataset_stats("quadra(full)", quadra_rows)
    print_dataset_stats("oneline_single(full)", oneline_single_rows)
    print_dataset_stats("oneline_split(full)", oneline_split_rows)

    # =========================================================
    # 2) sample negative to match positive
    # =========================================================

    sampled_negative_rows = sample_negative_to_match_positive(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        seed=seed,
    )

    print_dataset_stats("negative(sampled)", sampled_negative_rows)

    # =========================================================
    # 3) merge
    # =========================================================

    all_rows = (
        positive_rows
        + sampled_negative_rows
        + triple_rows
        + quadra_rows
        + oneline_single_rows
        + oneline_split_rows
    )

    random.Random(seed).shuffle(all_rows)

    print_dataset_stats("all(merged)", all_rows)

    # =========================================================
    # 4) split
    # =========================================================

    train_rows, valid_rows = split_train_valid(
        all_rows,
        val_ratio=val_ratio,
        seed=seed,
    )

    print_dataset_stats("train", train_rows)
    print_dataset_stats("valid", valid_rows)

    # =========================================================
    # 5) tokenizer/model
    # =========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        bert_path,
        use_fast=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        bert_path,
        num_labels=len(label_list),
        id2label=normalize_id2label(id2label),
        label2id=label2id,
    )

    class_weights = build_label_weights(
        label_list=label_list,
        o_weight=o_weight,
        other_weight=other_weight,
    )

    print("\n===== class weights =====")
    for label, w in zip(label_list, class_weights.tolist()):
        print(f"{label:20s}: {w}")

    # =========================================================
    # 6) dataset/collator
    # =========================================================

    train_dataset = TokenClsDataset(train_rows)
    valid_dataset = TokenClsDataset(valid_rows)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # =========================================================
    # 7) training args
    # =========================================================

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        report_to="none",
    )

    # =========================================================
    # 8) trainer
    # =========================================================

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_token_metrics,
        class_weights=class_weights,
    )

    # =========================================================
    # 9) train
    # =========================================================

    trainer.train()

    # =========================================================
    # 10) save
    # =========================================================

    best_model_dir = os.path.join(output_dir, "best_model")
    ensure_dir(best_model_dir)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    save_pkl(
        {
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "label_list": label_list,
            "label2id": label2id,
            "id2label": id2label,
            "bert_path": bert_path,
            "data_paths": data_paths,
            "data_dir": require_config_key(cfg, "data.dir"),   
            "sampled_negative_size": len(sampled_negative_rows),
            "positive_size": len(positive_rows),
            "negative_size": len(negative_rows),
            "triple_size": len(triple_rows),
            "quadra_size": len(quadra_rows),
            "oneline_single_size": len(oneline_single_rows),
            "oneline_split_size": len(oneline_split_rows),
            "train_size": len(train_rows),
            "valid_size": len(valid_rows),
            "config": cfg,
        },
        os.path.join(output_dir, "train_valid_split.pkl"),
    )

    print("\nTraining finished.")
    print(f"best model saved to: {best_model_dir}")
    print(f"train_valid_split saved to: {os.path.join(output_dir, 'train_valid_split.pkl')}")


if __name__ == "__main__":
    main()