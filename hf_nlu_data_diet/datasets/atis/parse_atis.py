import pickle

import pandas as pd

KEYS = ["text", "slots", "intent"]
VOCAB_KEYS = ['token_ids', 'slot_ids', 'intent_ids']
DATA_KEYS = ['query', 'slot_labels', 'intent_labels']

for f in ["train", "test"]:
    with open(f"raw/atis.{f}.pkl", "rb") as fp:
        data = pickle.load(fp)
    full_data, vocabs = data

    final = {}
    for l, vocab, d in zip(KEYS, VOCAB_KEYS, DATA_KEYS):
        V = {v: k for k, v in vocabs[vocab].items()}
        final[l] = []
        for i in full_data[d]:
            mapped_i = list(map(lambda x: V.get(int(x)), i))
            if l in KEYS[:-1]:
                mapped_i = mapped_i[1:-1]
            else:
                mapped_i = mapped_i[0]
            final[l].append(mapped_i)

    final["text"] = [" ".join(t) for t in final["text"]]
    final["slots"] = [" ".join(t) for t in final["slots"]]

    df = pd.DataFrame(final)
    df["id"] = list(range(len(df)))

    print(f.upper())
    print(df)

    df.to_parquet(f"nlu_data/{f}.parquet", engine="pyarrow")
