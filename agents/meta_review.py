system_prompt = "You are an expert in scientific research and meta-analysis."
metareview_prompt = """\
You are an expert in scientific research and meta-analysis.
Synthesize a comprehensive meta-review of provided reviews pertaining to the following research goal:

Goal: {goal}

Preferences:
{preferences}

Provided reviews for meta-analysis:
{reviews}

Instructions:
   * Generate a structured meta-analysis report of the provided reviews.
   * Focus on identifying recurring critique points and common issues raised by reviewers.
   * The generated meta-analysis should provide actionable insights for researchers developing future proposals.
   * Refrain from evaluating individual proposals or reviews; focus on producing a synthesized meta-analysis.
   
Response:\
"""

def metareview_generator(llm, goal, preferences, reviews):
    reviews_parsed = ""
    for review_type, review in reviews.items():
        reviews_parsed += f"[{review_type}]\n{review}\n\n"
    initial_review_input = metareview_prompt.format(goal=goal, preferences=preferences, reviews=reviews_parsed.strip())
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": initial_review_input
        }
    ]
    llm_result = llm.chat(input_messages) # no return format for now
    return llm_result
