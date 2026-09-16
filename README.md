# Address Bot NER

주소 NER의 **학습 데이터 생성 → 모델 학습 → PoC 추론/평가** 코드입니다.

시나리오 생성(GPT / AICC / Qwen)은 본 인수인계 범위에서 제외합니다.  
`text_template*.json` 형태의 최종 시나리오가 준비되어 있다고 가정하고 주소 삽입 단계부터 시작합니다.

## 1. 구조

```text
address_bot_3/
├── paths.py
├── processed/                         # jibun_*.json, road_*.json
├── negative_source/                   # addr0.json, addr5plus_0.json
│
├── make_address_insertion/
│   ├── make_address_text3.py          # 주소 DB 전처리
│   ├── make_insertion_address.py      # 삽입용 주소 표현 생성
│   ├── insert_address_multi.py        # single / double / split 삽입
│   ├── insert_address_by_plan.py      # triple / quadra 삽입
│   └── data/
│       ├── template/                  # 최종 scenario template
│       └── oneline*_triple*_quadra*_dataidx*/
│
├── make_positive_data.py
├── make_negative_data.py
├── process_negative_data.py
├── make_bert_token_cls_data3.py
│
├── train_data/                        # token classification PKL
│
├── train_script/
│   ├── train_bert.py
│   ├── run_train.sh
│   └── config/
│
└── inference/
    ├── inference_1st_poc.py
    ├── decode_inference_output.py
    ├── poc_calcualate_metric.py
    ├── inference_2nd_poc.py
    ├── evaluate_turn_2nd_poc.py
    └── parse_2nd_poc_rseult.py
```

`paths.py`에서 프로젝트 내부 경로를 관리합니다.

## 2. 학습 데이터 생성

### Positive

```text
processed/
   +
template/
   ↓
make_positive_data.py
   ↓
positive / triple / quadra / oneline dataset
```

`make_positive_data.py` 내부에서 다음 모듈을 사용합니다.

- `make_address_text3.py`: 지번/도로명 DB 로드 및 주소 row 구성
- `make_insertion_address.py`: full / shallow / split 주소 표현 생성
- `insert_address_multi.py`: single / double / split 주소 삽입
- `insert_address_by_plan.py`: triple / quadra 주소 삽입

실행:

```bash
python make_positive_data.py
```

### Negative

```text
negative_source/
   ↓
make_negative_data.py
   ↓
process_negative_data.py
   ↓
final_negative_dataset_addr0_dedup_thinned.json
```

실행:

```bash
python make_negative_data.py
python process_negative_data.py
```

### Token Classification 데이터

Positive / Negative 결과를 BERT/RoBERTa 학습용 PKL로 변환합니다.

```bash
python make_bert_token_cls_data3.py
```

결과는 `train_data/`에 저장됩니다.

## 3. 모델 학습

설정 파일:

```text
train_script/config/train_klue_1000_3_3_2.yaml
```

학습 실행:

```bash
bash train_script/run_train.sh
```

학습 결과는 `train_script/outputs_token_cls_klue/` 아래에 생성되며, 추론 시 각 실험의 `best_model/`을 사용합니다.

## 4. 1차 PoC

실행 순서:

```bash
python inference/inference_1st_poc.py
python inference/decode_inference_output.py
python inference/poc_calcualate_metric.py
```

흐름:

```text
입력 CSV
   ↓
token 단위 inference
   ↓
entity / span decoding
   ↓
Precision / Recall / F1 평가
```

## 5. 2차 PoC

실행 순서:

```bash
python inference/inference_2nd_poc.py
python inference/evaluate_turn_2nd_poc.py
python inference/parse_2nd_poc_rseult.py
```

흐름:

```text
multi-turn 입력
   ↓
turn / window 단위 inference + decoding
   ↓
turn 단위 평가
   ↓
결과 CSV 변환
```

2차 PoC는 별도의 decode script 없이 `inference_2nd_poc.py` 내부에서 decoding까지 수행합니다.

## 6. 전체 실행 순서

```text
[학습]

1. make_positive_data.py
2. make_negative_data.py
3. process_negative_data.py
4. make_bert_token_cls_data3.py
5. train_script/run_train.sh


[1차 PoC]

1. inference_1st_poc.py
2. decode_inference_output.py
3. poc_calcualate_metric.py


[2차 PoC]

1. inference_2nd_poc.py
2. evaluate_turn_2nd_poc.py
3. parse_2nd_poc_rseult.py
```