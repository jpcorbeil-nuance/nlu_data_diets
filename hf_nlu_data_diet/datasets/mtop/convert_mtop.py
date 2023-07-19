import json

import pandas as pd


def clean_intent(intent: str) -> str:
    return intent.split(":")[1]


def parse_slots(slots: str) -> list:
    return [s.split(":") for s in slots.split(",")]


def reformat_slots(slots: str, tokenization: str):
    token_dict = json.loads(tokenization)
    slot_lst = parse_slots(slots) if isinstance(slots, str) else []
    current_slot = slot_lst.pop(0) if len(slot_lst) > 0 else None
    inside = False
    output = []
    for token in token_dict["tokenSpans"]:
        if current_slot is not None:
            start_pos = int(token["start"])
            end_pos = start_pos + int(token["length"])
            if start_pos == int(current_slot[0]):
                output.append(f"B-{current_slot[-1]}")
                if end_pos == int(current_slot[1]):
                    continue
                inside = True
            elif inside and (end_pos != int(current_slot[1])):
                output.append(f"I-{current_slot[-1]}")
            elif inside and (end_pos == int(current_slot[1])):
                output.append(f"I-{current_slot[-1]}")
                current_slot = slot_lst.pop(0) if len(slot_lst) > 0 else None
                inside = False
            else:
                output.append("O")
        else:
            output.append("O")
    return " ".join(output)


def format_text(tokenization: str):
    token_dict = json.loads(tokenization)
    return " ".join(token_dict["tokens"])


def load_dataset(filename: str = "train"):
    df = pd.read_csv(f"raw/{filename}.txt", sep="\t")
    df.columns = ["id", "intent", "slots", "ori_text", "scenario", "lang", "repr", "tokenization"]
    df["intent"] = df["intent"].apply(clean_intent)
    for i, d in df.iterrows():
        df.loc[i, "slots"] = reformat_slots(d["slots"], d["tokenization"])
    df["text"] = df["tokenization"].apply(format_text)
    df.drop(["lang", "scenario", "repr", "tokenization", "ori_text"], axis=1, inplace=True)
    return df


for f in ["train", "dev", "test"]:
    df = load_dataset(filename=f)
    df.to_parquet(f"nlu_data/{f}.parquet", engine="pyarrow")
