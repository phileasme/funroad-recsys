import glob
import os

from collections import defaultdict
queries = defaultdict(list)
for x in glob.glob("./gumroad_data/*"):
    folder = x.replace("gumroad_data/search_", "")
    folder_splits = folder.replace("/","").replace(".","").split("_")
    if len(folder_splits) > 1:
        queries[len(folder_splits)].append(folder_splits)
    

f = []
for d in [[" ".join(y) for y in x[0][1]] for x in sorted(list(zip(queries.items())), key=lambda x: -x[0][0])]:
    f.extend(d)

print(f)