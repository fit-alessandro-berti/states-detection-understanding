import pickle
from collections import Counter
import numpy as np
import json

def compute(traces_all):
    ret = []
    for i in range(len(traces_all)):
        traces = traces_all[i]
        lst = [y for x in traces for y in x]
        act_orr = Counter(lst)
        dfg_relations = Counter([(x[i], x[i+1]) for x in traces for i in range(len(x)-1)])
        start_act = Counter([x[0] for x in traces])
        end_act = Counter([x[-1] for x in traces])
        ret.append({"act": act_orr, "dfg": dfg_relations, "start_act": start_act, "end_act": end_act})
    return ret


meta = pickle.load(open("som_global_meta.dump", "rb"))

listt = pickle.load(open("som_global_assignments.dump", "rb"))

listt2 = pickle.load(open("vectors.dump", "rb"))

all_activities = pickle.load(open("all_activities.dump", "rb"))

# keeps track of the activities executed prior to entering the current state
activities_pre = [list() for i in range(9)]
# keeps track of the activities executed before exiting the current state
activities_post = [list() for i in range(9)]
# keeps track of the activities executed in the current state
activities = [list() for i in range(9)]

for i in range(len(listt)):
    act_curr = []
    for j in range(len(listt[i])):
        vec = listt2[i][j+7]
        idxx = [i for i in range(len(vec)) if vec[i] == 1][0]
        act_curr.append(all_activities[idxx])
    j = 0
    while j < len(listt[i]):
        z = j
        while z < len(listt[i])-1:
            if listt[i][z+1] != listt[i][j]:
                break
            z = z + 1
        pre = act_curr[max(j-8, 0):j]
        if pre:
            activities_pre[listt[i][z]].append(pre)
        curr = act_curr[j:z+1]
        if curr:
            activities[listt[i][z]].append(curr)
        post = act_curr[max(j, z+1-8):z+1]
        if post:
            activities_post[listt[i][z]].append(post)
        j = z + 1

activities_pre1 = compute(activities_pre)
activities1 = compute(activities)
activities_post1 = compute(activities_post)

for i in range(len(activities_pre1)):
    for z in [activities_pre1, activities_post1, activities1]:
        z[i]["dfg"] = {x[0]+"->"+x[1]: y for x, y in z[i]["dfg"].items()}

#print(meta)

res = [{"coord": meta["coord_map"][i], "pre": activities_pre1[i], "post": activities_post1[i], "inside_state": activities1[i]} for i in range(len(activities_pre1))]
res = json.dumps(res, indent=2)
#print(res)

F = open("states.txt", "w", encoding="utf-8")
F.write(res)
F.close()
