import json
from typing import Dict, List, Any, Tuple, Optional

def reverse_speaker(speaker: str) -> str:
    speaker = speaker.strip()
    if speaker == "상담사":
        return "고객"
    if speaker == "고객":
        return "상담사"
    raise ValueError(f"unexpected speaker: {speaker}")

def find_first_known_speaker(dialogue: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    for i, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            continue
        if "speaker" not in turn:
            continue
        speaker = turn["speaker"]
        if isinstance(speaker, str) and speaker.strip() in {"상담사", "고객"}:
            return i, speaker.strip()
    return None, None

def find_first_placeholder_turn(dialogue: List[Dict[str, Any]]) -> Optional[int]:
    for i, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            continue
        if "text" not in turn:
            continue
        text = turn["text"]
        if isinstance(text, str) and has_placeholder(text):
            return i
    return None


def fill_from_anchor(
    dialogue: List[Dict[str, Any]],
    anchor_idx: int,
    anchor_speaker: str,
    sample_idx: int,
    anchor_reason: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    anchor_idx의 speaker를 anchor_speaker로 두고,
    앞/뒤를 번갈아 채운다.
    """
    fixed_dialogue = deepcopy(dialogue)
    fix_logs: List[Dict[str, Any]] = []

    # anchor speaker 강제 세팅
    if fixed_dialogue[anchor_idx]["speaker"] != anchor_speaker:
        fixed_dialogue[anchor_idx]["speaker"] = anchor_speaker
        fix_logs.append({
            "sample_idx": sample_idx,
            "turn_pos": anchor_idx,
            "action": "set_anchor_speaker",
            "anchor_reason": anchor_reason,
            "new_speaker": anchor_speaker,
        })

    # 앞쪽: anchor에서 거꾸로 alternating
    current = anchor_speaker
    for i in range(anchor_idx - 1, -1, -1):
        turn = fixed_dialogue[i]

        if not isinstance(turn, dict) or "speaker" not in turn:
            fix_logs.append({
                "sample_idx": sample_idx,
                "turn_pos": i,
                "action": "left_as_is",
                "reason": "invalid_turn_or_missing_speaker_key",
            })
            continue

        if turn["speaker"] is None:
            filled = reverse_speaker(current)
            turn["speaker"] = filled
            fix_logs.append({
                "sample_idx": sample_idx,
                "turn_pos": i,
                "action": "filled_backward",
                "anchor_idx": anchor_idx,
                "anchor_reason": anchor_reason,
                "new_speaker": filled,
            })
            current = filled
        elif isinstance(turn["speaker"], str) and turn["speaker"].strip() in {"상담사", "고객"}:
            current = turn["speaker"].strip()
        else:
            fix_logs.append({
                "sample_idx": sample_idx,
                "turn_pos": i,
                "action": "left_as_is",
                "reason": f"invalid_existing_speaker:{turn['speaker']}",
            })

    # 뒤쪽: anchor에서 앞으로 alternating
    current = anchor_speaker
    for i in range(anchor_idx + 1, len(fixed_dialogue)):
        turn = fixed_dialogue[i]

        if not isinstance(turn, dict) or "speaker" not in turn:
            fix_logs.append({
                "sample_idx": sample_idx,
                "turn_pos": i,
                "action": "left_as_is",
                "reason": "invalid_turn_or_missing_speaker_key",
            })
            continue

        if turn["speaker"] is None:
            filled = reverse_speaker(current)
            turn["speaker"] = filled
            fix_logs.append({
                "sample_idx": sample_idx,
                "turn_pos": i,
                "action": "filled_forward",
                "anchor_idx": anchor_idx,
                "anchor_reason": anchor_reason,
                "new_speaker": filled,
            })
            current = filled
        elif isinstance(turn["speaker"], str) and turn["speaker"].strip() in {"상담사", "고객"}:
            current = turn["speaker"].strip()
        else:
            fix_logs.append({
                "sample_idx": sample_idx,
                "turn_pos": i,
                "action": "left_as_is",
                "reason": f"invalid_existing_speaker:{turn['speaker']}",
            })

    return fixed_dialogue, fix_logs


def fill_speakers_final(
    dialogue: List[Dict[str, Any]],
    sample_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    반환:
        fixed_dialogue, fix_logs, is_invalid_by_rule

    규칙:
    1) known speaker가 있으면 그걸 anchor로 사용
    2) 모두 None이면 placeholder가 처음 나오는 turn을 고객으로 anchor
    3) 모두 None + placeholder 없음 => invalid
    """
    if not isinstance(dialogue, list):
        raise TypeError(f"sample_idx={sample_idx}: dialogue must be list, got {type(dialogue)}")

    first_known_idx, first_known_speaker = find_first_known_speaker(dialogue)
    if first_known_idx is not None:
        fixed_dialogue, fix_logs = fill_from_anchor(
            dialogue=dialogue,
            anchor_idx=first_known_idx,
            anchor_speaker=first_known_speaker,
            sample_idx=sample_idx,
            anchor_reason="first_known_speaker",
        )
        return fixed_dialogue, fix_logs, False

    placeholder_idx = find_first_placeholder_turn(dialogue)
    if placeholder_idx is not None:
        fixed_dialogue, fix_logs = fill_from_anchor(
            dialogue=dialogue,
            anchor_idx=placeholder_idx,
            anchor_speaker="고객",
            sample_idx=sample_idx,
            anchor_reason="first_placeholder_turn_assumed_customer",
        )
        return fixed_dialogue, fix_logs, False

    return deepcopy(dialogue), [{
        "sample_idx": sample_idx,
        "turn_pos": None,
        "action": "invalid_left_all_none",
        "reason": "all_speakers_none_and_no_placeholder",
    }], True



from copy import deepcopy
if __name__ == "__main__":


    qwen_data = read_json(qwen_path)
    other_data = read_json(other_path)
    
    new_qwen_data = {}
    invalid_samples = []
    speaker_fix_logs = []

    for key in KEYS:
        new_qwen_data[key] = []
        for j in range(len(qwen_data[key])):
            dialogue = qwen_data[key][j]['dialogue']
            fixed_dialogue, fix_logs, is_invalid_by_rule = fill_speakers_final(
                        dialogue=dialogue,
                        sample_idx=j,                        
                    )
            for turn in dialogue:
                if turn['speaker'] not in ['고객', '상담사']:
                    print(j)
                    break

            for log in fix_logs:
                log["source"] = "qwen3"
                log["scenario_key"] = key
            speaker_fix_logs.extend(fix_logs)

            if is_invalid_by_rule:
                invalid_samples.append({
                            "source": source,
                            "scenario_key": scenario_key,
                            "sample_idx": sample_idx,
                            "error_type": "SpeakerFillRuleError",
                            "error_message": "all speakers are None and no placeholder exists",
                            "sample": sample,
                        })
                continue
            new_sample = deepcopy(qwen_data[key][j])
            new_sample['dialogue'] = fixed_dialogue
            new_qwen_data[key].append(deepcopy(new_sample))
    
    print(speaker_fix_logs)
    for key in KEYS:
        print(len(new_qwen_data[key]))
        print(len(qwen_data[key]))
    new_data = build_new_data(new_qwen_data, other_data)