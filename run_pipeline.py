import argparse
import os
from openai import OpenAI
from models import StructuredLLM
from agents.generation import retrieve_and_reasoner, explorator, debate_simulator, assumption_identifier, research_expander 
from agents.reflection import initial_reviewer, full_reviewer, deep_reviewer, observation_reviewer, simulation_reviewer, tournament_reviewer
from agents.ranking import hyp_comparator, hyp_comparator_scientific_debate
from agents.evolution import feasibility_improver, outofthebox_thinker
from agents.meta_review import metareview_generator

from api_config import API_CONFIG
for key, api_list in API_CONFIG.items():
    os.environ[key] = api_list[0]

def main(args):

    llm = StructuredLLM(args.llm)
    
    with open(args.input_path) as rf:
        research_goal = rf.read() # Develop a novel hypothesis for the key factor or process which causes ALS ...
    
    # research_plan_config = research_plan_generator(research_goal) # skip for now
    research_plan_config = {
        "Preferences" : "Focus on providing a novel hypothesis, with detailed explanation of the mechanism of action.",
        "Attributes" : "novel, feasible",
        "Constraints" : "should be correct, should be novel."
    }
    
    visited_hyp_list = []
    reviews_list = []
    FINISH = False
    while FINISH==False:
        # 1. generation agent
        print(f"Generation agent ...\n##########################################")
        articles_with_reasoning = retrieve_and_reasoner(llm, research_goal) # need update (deep research API)
        source_hypothesis = "" if len(visited_hyp_list) == 0 else visited_hyp_list[-1]
        explorator_hyp = explorator(llm, research_goal, research_plan_config["Preferences"], source_hypothesis, articles_with_reasoning)
        # debate_simulator_hyp = debate_simulator(llm, research_plan_config["Attributes"], research_goal, research_plan_config["Preferences"], reviews_overview, "") # transcript is updated in multi-turn debate
        
        ### exclude assumptions, unexplored_areas for now
        # assumptions = assumption_identifier(llm, research_goal) # maybe return "assumptions" and their "sub-assumptions" dict?
        # unexplored_areas = research_expander(llm, visited_hyp_list, reviews_overview)

        if len(reviews_list) > 0:
            generated_hypothesis = debate_simulator(llm, research_plan_config["Attributes"], research_goal, research_plan_config["Preferences"], reviews_list[-1]) 
        else:
            generated_hypothesis = explorator_hyp

        print(f"generated hypothesis : {generated_hypothesis}\n##########################################")

        ## 2. reflection agent
        print(f"Reflection agent ...\n##########################################")
        initial_review = initial_reviewer(llm, generated_hypothesis) # return review & pass/fail
        print(f"initial review : {initial_review}\n##########################################")

        if "INAPPROPRIATE" not in initial_review: # needs improvement in matching!
            # full_review, articles_from_full_review = full_reviewer(llm, generated_hypothesis) # return correctness/quality critique & novelty critique & full review
            deep_review = deep_reviewer(llm, generated_hypothesis) # return correctness of assumptions / subassumptions in hypothesis
            print(f"deep review : {deep_review}\n##########################################")
            # observation_review = observation_reviewer(llm, generated_hypothesis, articles_from_full_review)
            simulation_review = simulation_reviewer(llm, generated_hypothesis)
            print(f"simulation review : {simulation_review}\n##########################################")
            # tournament_review = tournament_reviewer(llm, generated_hypothesis, tournament_results)

            ## 3. ranking agent (after discussing how to create multiple hypotheses)
            
            ## 4, meta-review agent
            print(f"Meta-review agent ...\n##########################################")
            reviews = {
                "deep_review":deep_review,
                "simulation_review":simulation_review
            } # need to add more reviews
            meta_review = metareview_generator(llm, research_goal, research_plan_config["Preferences"], reviews)
            print(f"meta review : {meta_review}\n##########################################")
            
            FINISH=True
        else:
            print("Inappropriate initial review, aborting...")
            FINISH=True

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--input_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "research_goal.txt"))
    parser.add_argument("--save_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "results"))
    parser.add_argument("--log_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs"))
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--command", type=str, help="The command that was run")

    args = parser.parse_args()
    main(args)