import pandas as pd
import pickle

dataframe = pd.read_parquet("../complete.parquet", columns=["case:concept:name", "concept:name", "state", "stock"])
dataframe["concept:name"] = dataframe["concept:name"].apply(lambda x: x.split("e_custom_")[-1])
dataframe = dataframe[~dataframe["concept:name"].str.startswith("START")]
dataframe = dataframe[~dataframe["concept:name"].str.startswith("END")]
dataframe = dataframe[~dataframe["concept:name"].str.startswith("ST")]
gdf = dataframe.groupby("case:concept:name")
activities = gdf["concept:name"].agg(list).to_dict()
states = gdf["state"].agg(list).to_dict()
stock = gdf["stock"].agg(list).to_dict()
all_activities = sorted(list(dataframe["concept:name"].unique()))

vectors = []

for k in activities:
    vectors.append([])
    for i in range(len(activities[k])):
        act_subvec = [0] * (len(all_activities) + 2)
        act_subvec[all_activities.index(activities[k][i])] = 1
        act_subvec[len(all_activities)] = stock[k][i]
        act_subvec[len(all_activities)+1] = states[k][i]
        vectors[-1].append(act_subvec)

print(vectors)
pickle.dump(vectors, open("vectors.dump", "wb"))
