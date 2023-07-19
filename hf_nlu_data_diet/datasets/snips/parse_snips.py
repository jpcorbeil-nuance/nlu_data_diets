import pandas as pd
for f in ["train", "dev", "test"]:
    with open(f"raw/{f}/label", "r") as fp:
        intent = [i.replace("\n", "") for i in fp.readlines()]
    with open(f"raw/{f}/seq.in", "r") as fp:
        text = [i.replace("\n", "").strip() for i in fp.readlines()]
    with open(f"raw/{f}/seq.out", "r") as fp:
        slots = [i.replace("\n", "").strip() for i in fp.readlines()]
    df = pd.DataFrame({"intent": intent, "text": text, "slots": slots})
    df["id"] = list(range(len(df)))
    print(f)
    print(df["text"].apply(lambda x: len(x.split())))
    print(df["slots"].apply(lambda x: len(x.split())))
    df.to_parquet(f"nlu_data/{f}.parquet", engine="pyarrow")
