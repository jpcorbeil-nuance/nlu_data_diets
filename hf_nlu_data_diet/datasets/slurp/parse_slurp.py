import re
import json

import pandas as pd

def mention_format(mention: str, string: str) -> list:
    lst = [mention] * len(string.split())
    return [("B-" if i==0 else "I-") + l for i, l in enumerate(lst)]

def extract_mention(token: str):
    mention_split = token.split(" : ")
    return mention_format(*mention_split) if len(mention_split) == 2 else ["O"]*len(token.split())

def parse_mentions(string: str) -> str:
    tokens = re.split(r"\[|\]", string)
    tokens = [t for t in tokens if t != ""]
    groups = [extract_mention(t) for t in tokens]
    return " ".join([t for g in groups for t in g])


with open("raw/massive-en-US.jsonl", "r") as fp:
    data = [json.loads(i.strip()) for i in fp.readlines()]

df = pd.DataFrame(data).set_index("id")
df["slots"] = df["annot_utt"].apply(parse_mentions)
df = df.drop(["locale", "worker_id", "scenario", "annot_utt"], axis=1)
df = df.rename({"utt": "text"}, axis=1)
partition = df.pop("partition")

for p in partition.unique():
    temp_df = df[partition==p]
    temp_df["id"] = list(range(len(temp_df)))
    temp_df.to_parquet(f"nlu_data/{p}.parquet", engine="pyarrow")
