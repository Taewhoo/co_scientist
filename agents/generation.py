# from tavily import TavilyClient

# tavily_client = TavilyClient()

hyp_gen_prompt = """\
Describe the proposed hypothesis in detail, including specific entities, mechanisms, and anticipated outcomes.

This description is intended for an audience of domain experts.

You have conducted a thorough review of relevant literature and developed a logical framework for addressing the objective. The articles consulted, along with your analytical reasoning, are provided below.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Existing hypothesis (if applicable):
{source_hypothesis}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):

{articles_with_reasoning}

Proposed hypothesis (detailed description for domain experts):\
"""
scientific_debate_prompt = """\
You are an expert participating in a collaborative discourse concerning the generation of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts. The overarching objective of this discourse is to collaboratively develop a {idea_attributes} hypothesis.

Goal: {goal}

Criteria for a high-quality hypothesis: {preferences}

Review Overview: {reviews_overview}

Procedure:

Initial contribution (if initiating the discussion): 
   Propose three distinct {idea_attributes} hypotheses.
   
Subsequent contributions (continuing the discussion):
   * Pose clarifying questions if ambiguities or uncertainties arise.
   * Critically evaluate the hypotheses proposed thus far, addressing the following aspects:
      - Adherence to {idea_attributes} criteria.
      - Utility and practicality.
      - Level of detail and specificity. 
   * Identify any weaknesses or potential limitations.
   * Propose concrete improvements and refinements to address identified weaknesses.
   * Conclude your response with a refined iteration of the hypothesis.
   
General guidelines:
   * Exhibit boldness and creativity in your contributions.
   * Maintain a helpful and collaborative approach.
   * Prioritize the generation of a high-quality {idea_attributes} hypothesis.
   
Termination condition:
   When sufficient discussion has transpired (typically 3-5 conversational turns, with a maximum of 10 turns) and all relevant questions and points have been thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS" (in all capital letters) followed by a concise and self-contained exposition of the finalized idea.
   
#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:\
"""

def retrieve_and_reasoner(llm, goal, articles_with_reasoning_path): # to be updated if deep research is available via API
    instruction = """
    Given a specific research goal, search the web for relevant and credible articles that contribute to achieving or addressing that goal. For each article you select, provide a reasoning that explains in detail how the article supports, informs, or relates to the research goal. The reasoning should reference specific elements of the article (e.g., findings, arguments, data, or methodology) and clearly articulate the connection to the research objective. 
    Return a list of (article, reasoning) pairs in reverse chronological order, beginning with the most recent analysis or publication. Each reasoning should demonstrate a thoughtful and well-supported link between the article’s content and the research goal.
    
    Output Format:
        - Title: [Title]
        - Article: [Article]
        - Reasoning: [Detailed reasoning that explains the article’s relevance to the research goal]

    Research Goal: {goal}
    """

    ## tmp (web)
    with open(articles_with_reasoning_path) as rf:
        articles_with_reasoning = rf.read()

    return articles_with_reasoning

def explorator(llm, goal, preferences, source_hypothesis, articles_with_reasoning):
    system_prompt = "You are an expert tasked with formulating a novel and robust hypothesis to address the following objective."
    hyp_gen_input = hyp_gen_prompt.format(goal=goal, preferences=preferences, source_hypothesis=source_hypothesis, articles_with_reasoning=articles_with_reasoning)
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": hyp_gen_input
        }
    ]
    llm_result = llm.chat(input_messages) # no return format for now
    return llm_result

def debate_simulator(llm, attributes, goal, preferences, reviews_overview):
    transcript = "" # updated with iteration
    return

def assumption_identifier(llm, goal):
    return

def research_expander(llm, visited_hyp_list, reviews_overview):
    return