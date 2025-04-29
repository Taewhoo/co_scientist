import re
import math
from collections import defaultdict
import random

system_prompt_hyp_comparison = "You are an expert evaluator tasked with comparing two hypotheses."
system_prompt_sci_debate = "You are an expert in comparative analysis, simulating a panel of domain experts engaged in a structured discussion to evaluate two competing hypotheses."

hyp_comparison_prompt = """\
Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one is superior based on the specified attributes: {idea_attributes}.
Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".

Goal: {goal}

Evaluation criteria:
{preferences}

Each hypothesis includes an independent review. These reviews may contain numerical scores.
Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}
 
Review of hypothesis 1:
{review_1}

Review of hypothesis 2:
{review_2}

Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):\
"""

hyp_comparison_scientific_debate_prompt = """\
The objective is to rigorously determine which hypothesis is superior based on a predefined set of attributes and criteria.
The experts possess no pre-existing biases toward either hypothesis and are solely focused on identifying the optimal choice, given that only one can be implemented.

Goal: {goal}

Criteria for hypothesis superiority:
{preferences}

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Initial review of hypothesis 1:
{review_1}

Initial review of hypothesis 2:
{review_2}

Debate procedure:

The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.

Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.

Subsequent turns:

   * Pose clarifying questions to address any ambiguities or uncertainties.
   * Critically evaluate each hypothesis in relation to the stated Goal and Criteria. This evaluation should consider aspects such as:
      - Potential for correctness/validity.
      - Utility and practical applicability.
      - Sufficiency of detail and specificity.
      - Novelty and originality.
      - Desirability for implementation.
   * Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Termination and judgment:

Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns) and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment.
This judgment should succinctly state the rationale for the selection.
Then, indicate the superior hypothesis by writing the phrase "better idea: ", followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).\
"""

def default_match(llm, research_goal, research_plan_config, hyp_dict_a, hyp_dict_b):

   hypothesis_1 = hyp_dict_a["hyp_full"]
   hypothesis_2 = hyp_dict_b["hyp_full"]

   review_dict_1 = {
      "full_review": hyp_dict_a["full_review"],
      "deep_review": hyp_dict_a["deep_review"],
      "observation_review": hyp_dict_a["observation_review"],
      "simulation_review": hyp_dict_a["simulation_review"],
   }
   review_dict_2 = {
      "full_review": hyp_dict_b["full_review"],
      "deep_review": hyp_dict_b["deep_review"],
      "observation_review": hyp_dict_b["observation_review"],
      "simulation_review": hyp_dict_b["simulation_review"],
   }

   review_1 = ""
   for review_type, review in review_dict_1.items():
      review_1 += f"[{review_type}]\n{review}\n\n"
   review_2 = ""
   for review_type, review in review_dict_2.items():
      review_2 += f"[{review_type}]\n{review}\n\n"

   hyp_comparison_input = hyp_comparison_prompt.format(idea_attributes=research_plan_config["Attributes"], goal=research_goal, preferences=research_plan_config["Preferences"], hypothesis_1=hypothesis_1, hypothesis_2=hypothesis_2, review_1=review_1.strip(), review_2=review_2.strip())
   input_messages = [
      {"role": "system", "content": system_prompt_hyp_comparison},
      {"role": "user", "content": hyp_comparison_input},
   ]

   # ensure that there is always a winner (temperature=0.2)
   parsed = False
   while not parsed:
      llm_result = llm.chat(input_messages)
      match = re.search(r"better hypothesis\s*:\s*<?\s*([12])\s*>?", llm_result.strip(), re.IGNORECASE)
      if match:
         parsed = True

   return llm_result, int(match.group(1))

def debate_match(llm, research_goal, research_plan_config, hyp_dict_a, hyp_dict_b):
   
   hypothesis_1 = hyp_dict_a["hyp_full"]
   hypothesis_2 = hyp_dict_b["hyp_full"]

   review_dict_1 = {
      "full_review": hyp_dict_a["full_review"],
      "deep_review": hyp_dict_a["deep_review"],
      "observation_review": hyp_dict_a["observation_review"],
      "simulation_review": hyp_dict_a["simulation_review"],
   }
   review_dict_2 = {
      "full_review": hyp_dict_b["full_review"],
      "deep_review": hyp_dict_b["deep_review"],
      "observation_review": hyp_dict_b["observation_review"],
      "simulation_review": hyp_dict_b["simulation_review"],
   }

   review_1 = ""
   for review_type, review in review_dict_1.items():
      review_1 += f"[{review_type}]\n{review}\n\n"
   review_2 = ""
   for review_type, review in review_dict_2.items():
      review_2 += f"[{review_type}]\n{review}\n\n"

   sci_debate_comparison_input = hyp_comparison_scientific_debate_prompt.format(goal=research_goal, preferences=research_plan_config["Preferences"], hypothesis_1=hypothesis_1, hypothesis_2=hypothesis_2, review_1=review_1.strip(), review_2=review_2.strip())
   input_messages = [
      {"role": "system", "content": system_prompt_sci_debate},
      {"role": "user", "content": sci_debate_comparison_input},
   ]

   # ensure that there is always a winner (temperature=0.2)
   parsed = False
   while not parsed:
      llm_result = llm.chat(input_messages)
      match = re.search(r"better idea\s*:\s*['\"]?([12])['\"]?", llm_result.strip(), re.IGNORECASE)
      if match:
         parsed = True

   return llm_result, int(match.group(1))


def elo_tournament(llm, research_goal, research_plan_config, paired_hypotheses):

   INITIAL_ELO = 1200
   K_FACTOR = 32
   id_score_dict = defaultdict(lambda: INITIAL_ELO)
   id_hyp_dict = {hyp["id_"]:hyp for pair in paired_hypotheses for hyp in pair}
   id_result_win_dict = defaultdict(list)
   id_result_lose_dict = defaultdict(list)

   def update_elo(winner_id, loser_id): # updates id_score_dict
      Ra = id_score_dict[winner_id]
      Rb = id_score_dict[loser_id]

      # Expected scores
      Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
      Eb = 1 / (1 + 10 ** ((Ra - Rb) / 400))

      # Update ratings
      id_score_dict[winner_id] = Ra + K_FACTOR * (1 - Ea)
      id_score_dict[loser_id] = Rb + K_FACTOR * (0 - Eb)

   ## 1. provide initial Elo rating of 1200, and do pairwise comparison
   for hyp_dict_a, hyp_dict_b in paired_hypotheses:

      match_result, winner_int = default_match(llm, research_goal, research_plan_config, hyp_dict_a, hyp_dict_b)

      if winner_int == 1:
         update_elo(hyp_dict_a["id_"], hyp_dict_b["id_"])
         id_result_win_dict[hyp_dict_a["id_"]].append(match_result)
         id_result_lose_dict[hyp_dict_b["id_"]].append(match_result)
      else:
         update_elo(hyp_dict_b["id_"], hyp_dict_a["id_"])
         id_result_win_dict[hyp_dict_b["id_"]].append(match_result)
         id_result_lose_dict[hyp_dict_a["id_"]].append(match_result)

   ## 2. sort by score, split to winner-loser group(4-4), and conduct separate match (x4 for now)
   
   # needs improvement : add number of iterations as argument 
   for _ in range(4):

      sorted_ids = sorted(id_score_dict.items(), key=lambda x: x[1], reverse=True)
      sorted_hyp_dicts = [id_hyp_dict[id_] for id_, _ in sorted_ids]

      mid = len(sorted_hyp_dicts) // 2
      winners = sorted_hyp_dicts[:mid]
      losers = sorted_hyp_dicts[mid:]

      # needs improvement : pair using proximity agent?
      random.shuffle(winners)
      random.shuffle(losers)

      winner_pairs = [(winners[i], winners[i+1]) for i in range(0, len(winners)-1, 2)]
      loser_pairs = [(losers[i], losers[i+1]) for i in range(0, len(losers)-1, 2)]
      
      # debate match for top pairs
      for hyp_dict_a, hyp_dict_b in winner_pairs:
         
         match_result, winner_int = debate_match(llm, research_goal, research_plan_config, hyp_dict_a, hyp_dict_b)

         if winner_int == 1:
            update_elo(hyp_dict_a["id_"], hyp_dict_b["id_"])
            id_result_win_dict[hyp_dict_a["id_"]].append(match_result)
            id_result_lose_dict[hyp_dict_b["id_"]].append(match_result)
         else:
            update_elo(hyp_dict_b["id_"], hyp_dict_a["id_"])
            id_result_win_dict[hyp_dict_b["id_"]].append(match_result)
            id_result_lose_dict[hyp_dict_a["id_"]].append(match_result)

      # default match for low pairs
      for hyp_dict_a, hyp_dict_b in loser_pairs:
      
         match_result, winner_int = default_match(llm, research_goal, research_plan_config, hyp_dict_a, hyp_dict_b)

         if winner_int == 1:
            update_elo(hyp_dict_a["id_"], hyp_dict_b["id_"])
            id_result_win_dict[hyp_dict_a["id_"]].append(match_result)
            id_result_lose_dict[hyp_dict_b["id_"]].append(match_result)
         else:
            update_elo(hyp_dict_b["id_"], hyp_dict_a["id_"])
            id_result_win_dict[hyp_dict_b["id_"]].append(match_result)
            id_result_lose_dict[hyp_dict_a["id_"]].append(match_result)
   
   ## 3. return 
   hyp_after_ranking = []
   elo_scores = [score for score in id_score_dict.values()]
   for id_, score in id_score_dict.items():
      hyp_dict = id_hyp_dict[id_]
      
      hyp_dict["elo_score"] = score
      hyp_dict["ranking_win_results"] = id_result_win_dict[id_]
      hyp_dict["ranking_lose_results"] = id_result_lose_dict[id_]

      hyp_after_ranking.append(hyp_dict)

   print(f"elo scores: {elo_scores}\n##########################################") # needs improvement in logging (e.g. save in cache dir)

   ## return top scoring (half) hypotheses
   hyp_after_ranking.sort(key=lambda x: x["elo_score"], reverse=True)
   top_half = hyp_after_ranking[:len(hyp_after_ranking) // 2]

   return top_half