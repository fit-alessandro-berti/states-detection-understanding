import pm4py
import json
from random import randrange


def split_on_capital_and_end(text: str) -> str:
    """
    Inserts a newline before each capital letter (except the first)
    and ensures the text ends with a newline.
    """
    result = ""
    for i, char in enumerate(text):
        if i > 0 and char.isupper():
            result += "\n"  # new line before capital
        result += char

    if not result.endswith("\n"):
        result += "\n"  # ensure newline at end

    return result


def save_vis(state, label):
    state0 = state[label]
    dfg, sa, ea = state0["dfg"], state0["start_act"], state0["end_act"]
    dfg = {(split_on_capital_and_end(x.split("->")[0]), split_on_capital_and_end(x.split("->")[1])): y for x, y in dfg.items()}
    sa = {split_on_capital_and_end(x): y for x, y in sa.items()}
    ea = {split_on_capital_and_end(x): y for x, y in ea.items()}
    pm4py.save_vis_dfg(dfg, sa, ea, label+".pdf", rankdir="TB")

state = json.load(open("states.txt", "r"))[-2]
save_vis(state, "inside_state")
save_vis(state, "pre")
save_vis(state, "post")
