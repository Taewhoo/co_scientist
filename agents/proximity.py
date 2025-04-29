import re
import random

system_prompt = "You are an expert tasked with comparing scientific hypotheses based on their relevance and similarity to a given research goal."
proximity_prompt = """You are given a set of hypotheses related to the following research goal. Your task is to assess the conceptual similarity between each pair of hypotheses, based on how closely they address the same mechanisms, scientific reasoning, or biological pathways relevant to the research goal.

Research Goal:
{research_goal}

Hypotheses:
{indexed_hypotheses}

First, provide a brief rationale explaining how you evaluated similarity across hypotheses.

Then, group the hypotheses into pairs that are most similar in concept or approach. Each hypothesis should appear in exactly one pair. If there is an odd number of hypotheses, leave one out unpaired. 

Present the result as a proximity graph, using the format [index]-[index] (e.g., [2]-[5]).

Your output should consist of the following two sections:

- Rationale: Your reasoning process and criteria for pairing similar hypotheses.
- Proximity Graph: A list of the most similar hypothesis pairs, one pair per line, in the format [x]-[y].

Do not include any content outside these two sections."""
same_hyp_prompt = """You are given a set of hypotheses related to the following research goal. Your task is to identify and remove hypotheses that are essentially the same, even if they are written differently.

Research Goal:
{research_goal}

Hypotheses:
{indexed_hypotheses}

First, analyze the meaning of each hypothesis. Consider two hypotheses to be the same if they express the same core idea or explain the same phenomenon in a similar way, even if they use different words.

Then, return only the distinct hypotheses by listing their indices, separated by commas.

Your output should consist of the following two sections:

- Rationale: Explain your reasoning process and the criteria you used to identify similar hypotheses.
- Final Hypotheses: A list of indices (e.g., [1], [3], [5]) separated by commas that correspond to distinct hypotheses.

Do not include any content outside these two sections."""

def parse_same_hyps_with_ids(llm_result, tmp_idx_dict):

    match = re.search(r"Final Hypotheses\s*:\s*(.+)", llm_result, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        indices = re.findall(r"\[?(\d+)\]?", text)
        indices = [f"[{i}]" for i in indices]

        ids = [tmp_idx_dict[idx] for idx in indices if idx in tmp_idx_dict]

    else: # parsing error : return all without filtering
        
        ids = list(tmp_idx_dict.items())
    
    return ids


def parse_proximity_graph_with_ids(llm_result, tmp_idx_dict):
    all_keys = list(tmp_idx_dict.keys())
    used_keys = set()
    id_pairs = []

    # Step 1: Try to extract the proximity graph section
    match = re.search(r"Proximity Graph\s*:\s*(.+)", llm_result, re.IGNORECASE | re.DOTALL)
    if match:
        graph_text = match.group(1).strip()

        # Step 2: Parse pairs like [1]-[4]
        pairs = re.findall(r"\[\s*(\d+)\s*\]\s*-\s*\[\s*(\d+)\s*\]", graph_text)

        for a, b in pairs:
            key_a = f"[{a}]"
            key_b = f"[{b}]"
            if key_a in tmp_idx_dict and key_b in tmp_idx_dict:
                id_pairs.append((tmp_idx_dict[key_a], tmp_idx_dict[key_b]))
                used_keys.update([key_a, key_b])
    
    ### do not force full pairing (there might be odd number of hypotheses)

    # # Step 3: Find any missing keys
    # unused_keys = list(set(all_keys) - used_keys)
    # random.shuffle(unused_keys)

    # # Step 4: Pair up remaining keys randomly
    # for i in range(0, len(unused_keys) - 1, 2):
    #     id_pairs.append((
    #         tmp_idx_dict[unused_keys[i]],
    #         tmp_idx_dict[unused_keys[i + 1]]
    #     ))

    # # Step 5: If everything failed and no pairs were parsed, do full random pairing
    # if not id_pairs:
    #     shuffled_keys = all_keys[:]
    #     random.shuffle(shuffled_keys)
    #     for i in range(0, len(shuffled_keys) - 1, 2):
    #         id_pairs.append((
    #             tmp_idx_dict[shuffled_keys[i]],
    #             tmp_idx_dict[shuffled_keys[i + 1]]
    #         ))

    return id_pairs

def calculate_proximity(llm, research_goal, hypotheses):

    tmp_idx_dict = dict()
    indexed_hypotheses = []
    tmp_idx = 1
    
    for hyp_dict in hypotheses:
        
        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]

        indexed_hypotheses.append(f"[{tmp_idx}]\n{hyp_full}")
        tmp_idx_dict[f"[{tmp_idx}]"] = id_
        tmp_idx += 1

    proximity_review_input = proximity_prompt.format(research_goal=research_goal, indexed_hypotheses="\n\n".join(indexed_hypotheses))
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": proximity_review_input
        }
    ]
    llm_result = llm.chat(input_messages) 

    real_id_pairs = parse_proximity_graph_with_ids(llm_result, tmp_idx_dict) # e.g., [('3456', '1234'), ('2345', '6789'), ('4567', '5678')]
    id_to_dict = {h["id_"]: h for h in hypotheses}
    final_list = [(id_to_dict[a], id_to_dict[b]) for a, b in real_id_pairs]
    
    print(f"proximity result: {final_list}\n##########################################")
    
    return final_list

def exclude_same_hyp(llm, research_goal, hypotheses):

    tmp_idx_dict = dict()
    indexed_hypotheses = []
    tmp_idx = 1
    
    for hyp_dict in hypotheses:
        
        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]

        indexed_hypotheses.append(f"[{tmp_idx}]\n{hyp_full}")
        tmp_idx_dict[f"[{tmp_idx}]"] = id_
        tmp_idx += 1
    
    same_hyp_input = same_hyp_prompt.format(research_goal=research_goal, indexed_hypotheses="\n\n".join(indexed_hypotheses))
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": same_hyp_input
        }
    ]
    llm_result = llm.chat(input_messages) 

    real_id_pairs = parse_same_hyps_with_ids(llm_result, tmp_idx_dict)
    id_to_dict = {h["id_"]: h for h in hypotheses}
    final_list = [id_to_dict[id_] for id_ in real_id_pairs]
    
    print(f"same hypotheses filtering result: {final_list}\n##########################################")
    
    return final_list