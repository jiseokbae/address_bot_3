from pathlib import Path


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Address DB / training data generation
# =========================================================

# jibun_*.json, road_*.json
PROCESSED_DIR = PROJECT_ROOT / "processed"

ADDRESS_INSERTION_DIR = PROJECT_ROOT / "make_address_insertion"
ADDRESS_DATA_DIR = ADDRESS_INSERTION_DIR / "data"

# 최종 scenario template
TEMPLATE_DIR = ADDRESS_DATA_DIR / "template"


# =========================================================
# Negative data
# =========================================================
#
# negative를 다시 생성할 경우 필요한 원천 데이터.
# 시나리오 생성 파이프라인은 인수인계 범위에서 제외했으므로
# 필요한 addr0 데이터만 별도 보관하는 구조로 둠.
#
NEGATIVE_SOURCE_DIR = PROJECT_ROOT / "negative_source"

NEGATIVE_ADDR0_PATH = NEGATIVE_SOURCE_DIR / "addr0.json"
NEGATIVE_ADDR5PLUS0_PATH = NEGATIVE_SOURCE_DIR / "addr5plus_0.json"


# =========================================================
# Token classification data / training
# =========================================================

# make_bert_token_cls_data3.py 실행 시 생성
TRAIN_DATA_DIR = PROJECT_ROOT / "train_data"

TRAIN_SCRIPT_DIR = PROJECT_ROOT / "train_script"

# train_bert.py 결과
TRAIN_OUTPUT_DIR = TRAIN_SCRIPT_DIR / "outputs_token_cls_klue"


# =========================================================
# Model
# =========================================================

# Hugging Face model ID
KLUE_MODEL_ID = "klue/roberta-large"

# Local model
IXI_MODEL_DIR = PROJECT_ROOT / "models" / "ixi-RoBERTa-base_250203"

# =========================================================
# Inference
# =========================================================

INFERENCE_DIR = PROJECT_ROOT / "inference"

FIRST_POC_DIR = INFERENCE_DIR / "first_poc"
SECOND_POC_DIR = INFERENCE_DIR / "second_poc"

# 1차 PoC 입력 / GT
FIRST_POC_INPUT_CSV = FIRST_POC_DIR / "ADDRESS_BOT_TC.csv"
FIRST_POC_GT_CSV = FIRST_POC_DIR / "260327_주소봇_1차PoC_NER_GT.csv"

# 2차 PoC GT
SECOND_POC_GT_CSV = (
    SECOND_POC_DIR
    / "(최종)2차PoC_신규TC_주소NER_정답지_260522.csv"
)

# 1차 PoC 결과
FIRST_POC_OUTPUT_DIR = INFERENCE_DIR / "output"
FIRST_POC_RESULT_DIR = INFERENCE_DIR / "result"


# =========================================================
# Helper functions
# =========================================================

def experiment_name(oneline, triple, quadra, dataidx):
    return (
        f"oneline{oneline}_"
        f"triple{triple}_"
        f"quadra{quadra}_"
        f"dataidx{dataidx}"
    )


def generated_data_dir(oneline, triple, quadra, dataidx):
    """make_positive_data.py 결과 저장 위치"""
    return ADDRESS_DATA_DIR / experiment_name(
        oneline, triple, quadra, dataidx
    )


def train_data_dir(oneline, triple, quadra, dataidx):
    """make_bert_token_cls_data3.py 결과 저장 위치"""
    return TRAIN_DATA_DIR / experiment_name(
        oneline, triple, quadra, dataidx
    )


def train_output_dir(oneline, triple, quadra, dataidx):
    """train_bert.py 결과 저장 위치"""
    return TRAIN_OUTPUT_DIR / experiment_name(
        oneline, triple, quadra, dataidx
    )


def second_poc_output_dir(dir_idx):
    return INFERENCE_DIR / f"output_{dir_idx}"


def second_poc_result_dir(dir_idx):
    return INFERENCE_DIR / f"result_{dir_idx}"


def resolve_project_path(path):
    """
    config에 상대경로가 들어오면 PROJECT_ROOT 기준으로 변환.
    이미 절대경로면 그대로 사용.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path