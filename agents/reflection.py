
system_prompt = "You are an expert in scientific hypothesis evaluation."

initial_review_prompt = """\
Given a specific hypothesis, perform an initial review assessing its overall suitability. Your review should address the following four criteria:

Correctness – Does the hypothesis align with known scientific or logical principles? Are there any apparent flaws, contradictions, or assumptions that undermine its validity?

Quality – Is the hypothesis well-formed, precise, and coherent? Does it demonstrate clarity of thought and logical structure?

Novelty – Does the hypothesis offer a new or non-obvious perspective, question, or idea? Or does it simply restate well-known concepts?

Preliminary Safety and Ethics – Are there any immediate ethical, social, or safety concerns associated with pursuing this hypothesis (e.g., risks to humans, privacy issues, misuse potential)?

Return a structured initial_review that clearly addresses each of the four criteria above.

Output Format:

- Correctness: [Assessment]
- Quality: [Assessment]
- Novelty: [Assessment]
- Safety & Ethics: [Assessment]
- FINAL REVIEW : [Select from ("APPROPRIATE", "INAPPROPRIATE")]

Hypothesis: {hypothesis}
"""

deep_review_prompt = """\
Given a specific hypothesis, conduct a deep verification review. Your goal is to rigorously assess the hypothesis by decomposing it into its underlying assumptions and evaluating their validity. Follow these steps:

Decomposition:
Break the hypothesis down into its constituent assumptions. For each assumption, identify and articulate any sub-assumptions that underpin it.

Decontextualized Evaluation:
For each assumption and sub-assumption, evaluate its correctness independently of the hypothesis context. Determine whether each is logically or scientifically valid on its own.

Error Identification:
Identify any incorrect or questionable assumptions. Summarize how each could potentially invalidate the hypothesis, and explain the reasoning behind this conclusion.

Fundamentality Assessment:
For each invalid or uncertain assumption, assess whether it is fundamental to the hypothesis. If an incorrect assumption is non-fundamental, note that it may be resolved or refined later and does not invalidate the core hypothesis.

Output Format:

- Assumption 1:
    - Sub-assumptions:
        - [Sub-assumption A1]
        - [Sub-assumption A2]
    - Evaluation:
        - [Assessment of each sub-assumption]
    - Is Fundamental: [Yes/No]
    - Impact on Hypothesis: [Explain if/how it invalidates or weakens the hypothesis]

- Assumption 2:
    ...
    
- Summary of Potential Invalidations:
    - [Concise summary of how any incorrect assumptions may impact the hypothesis]

Hypothesis: {hypothesis}
"""

observation_review_prompt = """\
Your task is to analyze the relationship between a provided hypothesis and observations from a scientific article. Specifically, determine if the hypothesis provides a novel causal explanation for the observations, or if they contradict it.

Instructions:

1. Observation extraction: list relevant observations from the article.
2. Causal analysis (individual): for each observation: 
   a. State if its cause is already established.
   b. Assess if the hypothesis could be a causal factor (hypothesis => observation).
   c. Start with: "would we see this observation if the hypothesis was true:".
   d. Explain if it’s a novel explanation. If not, or if a better explanation exists, state: "not a missing piece."
3. Causal analysis (summary): determine if the hypothesis offers a novel explanation for a subset of observations. Include reasoning. Start with: "would we see some of the observations if the hypothesis was true:".
4. Disproof analysis: determine if any observations contradict the hypothesis. Start with: "does some observations disprove the hypothesis:".
5. Conclusion: state: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".

Scoring:
   * Already explained: hypothesis consistent, but causes are known. No novel explanation.
   * Other explanations more likely: hypothesis *could* explain, but better explanations exist.
   * Missing piece: hypothesis offers a novel, plausible explanation.
   * Neutral: hypothesis neither explains nor is contradicted.
   * Disproved: observations contradict the hypothesis.
   
Important: if observations are expected regardless of the hypothesis, and don’t disprove it, it’s neutral.

Article:
{article}

Hypothesis:
{hypothesis}

Response {provide reasoning. end with: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".)\
"""

simulation_review_prompt = """\
Given a specific hypothesis, conduct a simulation-based review by mentally simulating the hypothesis in a step-wise manner. This may involve simulating the mechanism of action, the logical flow, or the experimental process proposed by the hypothesis.

Use your internal model of the world to reason through how the hypothesis would play out if tested or implemented. At each step, identify and document:

What happens based on the hypothesis.

Where and how failure might occur, including contradictions, implausible transitions, or missing mechanisms.

Why these failure points matter—i.e., how they could undermine the intended outcome or interpretation.

Return a structured simulation_review that outlines the step-by-step simulation, highlights potential failure scenarios, and reflects on their implications for the viability of the hypothesis.

Output Format:

- Step 1: [Simulated outcome or process step]
- Step 2: [Next step in the simulated process]
- ...
- Failure Scenarios:
    - [Scenario 1: Description of potential failure, when it occurs, and why]
    - [Scenario 2: ...]
- Implications:
    - [Explain how identified failures affect the strength, validity, or design of the hypothesis]

Hypothesis: {hypothesis}
"""


def initial_reviewer(llm, hypothesis):
    initial_review_input = initial_review_prompt.format(hypothesis=hypothesis)
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

def full_reviewer(llm, hyp):
    return

def deep_reviewer(llm, hypothesis):
    deep_review_input = deep_review_prompt.format(hypothesis=hypothesis)
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": deep_review_input
        }
    ]
    llm_result = llm.chat(input_messages) # no return format for now
    return llm_result

def observation_reviewer(llm, articles_from_full_review):
    return

def simulation_reviewer(llm, hypothesis):
    simulation_review_input = simulation_review_prompt.format(hypothesis=hypothesis)
    input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": simulation_review_input
        }
    ]
    llm_result = llm.chat(input_messages) # no return format for now
    return llm_result

def tournament_reviewer(llm):
    return
