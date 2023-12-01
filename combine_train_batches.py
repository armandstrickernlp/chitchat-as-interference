import json

train_gen = {}

for i in range(3):
    with open(f"outputs/TRAIN_batch{i}/gen_reactions.json") as f:
        gen = json.load(f)
        train_gen.update(gen)

with open("TRAIN/gen_reactions.json", "w") as f:
    json.dump(train_gen, f, indent=2)
