import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from openai import OpenAI
from models import StructuredLLM
from agents.generation import retrieve_and_reasoner, retrieve_from_db, explorator, debate_simulator, assumption_identifier, research_expander 
from agents.reflection import initial_reviewer, full_reviewer, deep_reviewer, observation_reviewer, simulation_reviewer, tournament_reviewer
from agents.proximity import calculate_proximity, exclude_same_hyp
from agents.ranking import elo_tournament
from agents.evolution import evolve_hypotheses
from agents.meta_review import metareview_generator

from api_config import API_CONFIG
for key, api_list in API_CONFIG.items():
    os.environ[key] = api_list[0]
import json
import time

def main(args):

    llm = StructuredLLM(llm_name=args.llm, temperature=0.2)
    llm_explorator = StructuredLLM(llm_name=args.llm, temperature=1.0)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    with open(args.input_path) as rf:
        research_goal = rf.read() # Develop a novel hypothesis for the key factor or process which causes ALS ...
    
    # research_plan_config = research_plan_generator(research_goal) # skip for now
    research_plan_config = {
        "Preferences" : "Focus on providing a novel hypothesis, with detailed explanation of the mechanism of action.",
        "Attributes" : "novel, feasible",
        "Constraints" : "should be correct, should be novel."
    }
    
    visited_hyp = []
    hyp_after_meta_review = []
    FINISH = False
    while FINISH==False:
        # 1. generation agent
        generation_start = time.time()
        print(f"Generation agent ...\n##########################################")

        if len(hyp_after_meta_review) == 0:
            articles_with_reasoning = retrieve_from_db(research_goal, 10) # retrieve top k documents from DB (need update : add reasoning)
            source_hypothesis = ""
            generated_hypotheses = explorator(llm_explorator, research_goal, research_plan_config["Preferences"], source_hypothesis, "\n\n".join(articles_with_reasoning), args.num_init_hyp) # keys : "id_", "hyp_full", "hyp_main"
        else:
            FINISH=True
            generated_hypotheses = debate_simulator(llm, research_plan_config["Attributes"], research_goal, research_plan_config["Preferences"], hyp_after_meta_review, 10) # max turns = 10 / return new hyp_dict (+ "prev_id")

            generated_hypotheses += hyp_after_evolution
        ### exclude assumptions, unexplored_areas for now
        # assumptions = assumption_identifier(llm, research_goal) # maybe return "assumptions" and their "sub-assumptions" dict?
        # unexplored_areas = research_expander(llm, visited_hyp_list, reviews_overview)
        
        generation_end = time.time()
        print(f"Generation agent time : {generation_end - generation_start} seconds")
        print(f"generated hypothesis (num={len(generated_hypotheses)}): {generated_hypotheses}\n##########################################")

        ## filtering duplicate hypotheses
        filtering_start = time.time()
        print(f"Filtering duplicates ...\n##########################################")
        generated_hypotheses = exclude_same_hyp(llm, research_goal, generated_hypotheses)
        filtering_end = time.time()
        print(f"Filtering time (num={len(generated_hypotheses)}): {filtering_end - filtering_start} seconds")

        ## 2. reflection agent
        reflection_start = time.time()
        print(f"Reflection agent ...\n##########################################")
        hyp_after_init_review = initial_reviewer(llm, generated_hypotheses)
        hyp_after_init_review = [hyp_dict for hyp_dict in hyp_after_init_review if "INAPPROPRIATE" not in hyp_dict["initial_review"]] # keys : "id_", "hyp_full", "hyp_main", "initial_review"
        visited_hyp += hyp_after_init_review

        hyp_after_full_review = full_reviewer(llm, hyp_after_init_review) # keys : "id_", "hyp_full", "hyp_main", "initial_review", "full_review", "related_articles_text"

        hyp_after_deep_review = deep_reviewer(llm, hyp_after_full_review) # keys : "id_", "hyp_full", "hyp_main", "initial_review", "full_review", "related_articles_text", "deep_review"

        hyp_after_observation_review = observation_reviewer(llm, hyp_after_deep_review) # keys : "id_", "hyp_full", "hyp_main", "initial_review", "full_review", "related_articles_text", "deep_review", "observation_review"

        hyp_after_simulation_review = simulation_reviewer(llm, hyp_after_observation_review) # keys : "id_", "hyp_full", "hyp_main", "initial_review", "full_review", "related_articles_text", "deep_review", "observation_review", "simulation_review"

        # tournament_review = tournament_reviewer(llm, generated_hypothesis, tournament_results)
        
        reflection_end = time.time()
        print(f"Reflection agent time : {reflection_end - reflection_start} seconds")
        
        hyp_with_reviews = hyp_after_simulation_review

        if len(hyp_after_meta_review) == 0:
            # save results (중간)
            with open(os.path.join(args.save_path, "results_iter1_gen_reflect.jsonl"), "w", encoding="utf-8") as wf:
                for line in hyp_after_simulation_review:
                    wf.write(json.dumps(line) + "\n")
        else:
            with open(os.path.join(args.save_path, "results_iter2_gen_reflect.jsonl"), "w", encoding="utf-8") as wf:
                for line in hyp_after_simulation_review:
                    wf.write(json.dumps(line) + "\n")

        # ### tmp ###
        # hyp_with_reviews = []
        # with open("/home/taewhoo/aigen/hypothesis/results_skp2/results_iter1_gen_reflect.jsonl") as rf:
        #     for line in rf:
        #         hyp_with_reviews.append(json.loads(line))

        ## 3. proximity agent
        proximity_start = time.time()
        print(f"Proximity agent ...\n##########################################")
        paired_hypotheses = calculate_proximity(llm, research_goal, hyp_with_reviews) # e.g., [(hyp_dict_a, hyp_dict_b), ...]
        
        proximity_end = time.time()
        print(f"Proximity agent time : {proximity_end - proximity_start} seconds")
       
        ## 4. ranking agent
        ranking_start = time.time()
        print(f"Ranking agent ...\n##########################################")
        hyp_after_tournament = elo_tournament(llm, research_goal, research_plan_config, paired_hypotheses) # keys : "id_", "hyp_full", "hyp_main", "initial_review", "full_review", "related_articles_text", "deep_review", "observation_review", "simulation_review", "elo_score", "ranking_win_results", "ranking_lose_results"

        ranking_end = time.time()
        print(f"Ranking agent time : {ranking_end - ranking_start} seconds")
        
        if len(hyp_after_meta_review) == 0:

            ## 5. evolution agent
            evolution_start = time.time()
            print(f"Evolution agent ...\n##########################################")
            hyp_after_evolution = evolve_hypotheses(llm, research_goal, research_plan_config["Preferences"], hyp_after_tournament)
            # keys: "id_", "hyp_full"
            evolution_end = time.time()
            print(f"Evolution agent time : {evolution_end - evolution_start} seconds")

            ## 6. meta-review agent
            meta_start = time.time()
            print(f"Meta-review agent ...\n##########################################")
            hyp_after_meta_review = metareview_generator(llm, research_goal, research_plan_config["Preferences"], hyp_after_tournament)

            meta_end = time.time()
            print(f"Meta agent time : {meta_end - meta_start} seconds")
            
            # save intermediate results
            with open(os.path.join(args.save_path, "results_iter1.jsonl"), "w", encoding="utf-8") as wf:
                for line in hyp_after_meta_review:
                    wf.write(json.dumps(line) + "\n")

            with open(os.path.join(args.save_path, "results_iter1_evolution.jsonl"), "w", encoding="utf-8") as wf:
                for line in hyp_after_evolution:
                    wf.write(json.dumps(line) + "\n")

        else:
            with open(os.path.join(args.save_path, "results_iter2.jsonl"), "w", encoding="utf-8") as wf:
                for line in hyp_after_tournament:
                    wf.write(json.dumps(line) + "\n")
                    
            ### top scorer is selected
            best_dict = max(hyp_after_tournament, key=lambda x: x["elo_score"])
            print(f"BEST HYPOTHESIS: {best_dict}\n##########################################")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--input_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "research_goal_enhertu.txt"))
    parser.add_argument("--save_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "results_enhertu"))
    parser.add_argument("--log_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs"))
    parser.add_argument("--num_init_hyp", type=int, default=8)
    parser.add_argument("--command", type=str, help="The command that was run")


    args = parser.parse_args()
    main(args)