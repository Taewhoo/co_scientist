import re
from collections import defaultdict

import numpy as np
import pubmed_parser as pp
import torch
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
full_review_prompt = """\
Your task is to conduct a full review of given hypothesis/rationales and its related articles. Specifically, you should evaluate the hypothesis with its rationales in terms of correctness, quality, and novelty. For correctness and quality, analyze the underlying assumptions and reasoning. For novelty, summarize what is already known about the hypothesis and evaluate how original it is based on existing literature.

---

Instructions:
Follow the steps below and provide your findings under each clearly titled section.

- **Related Articles:**
    - Assess the relevance of the retrieved literature to the given hypothesis.
    - List the related articles in order of relevance.
    - Provide a brief summary explaining why each article is relevant.
    - Follow a format: "[citation index] title: explanation"
    - Use citation numbers in this section to refer to specific literature in later sections, if needed.
- **Assumptions of the Idea:**
    - Clearly identify and summarize the assumptions that the hypothesis relies on.
    - Include both explicit assumptions (directly stated) and implicit assumptions (unstated but necessary for the reasoning to work).
    - Split the assumptions into fine-grained ideas.
    - Follow a format: "summary: explanation"
- **Reasoning about assumptions:**
    - Analyze and assess the plausibility of the assumptions identified in the previous section.
    - Consider whether the reasoning based on these assumptions is sound.
- **Aspects already explored:**
    - Based on the related literature, identify and summarize the elements of the hypothesis that have already been addressed in prior work.
    - Include both explicit aspects (directly stated) and implicit aspects (unstated but necessary for the reasoning to work).
    - Split the aspects into fine-grained elements.
    - Provide the referred citation numbers.
- **Novel Aspects:**
    - Highlight any elements of the hypothesis that have not been addressed or explored in the existing literature.
    - Include both explicit aspects (directly stated) and implicit aspects (unstated but necessary for the reasoning to work).
    - Split the aspects into fine-grained elements.
- **Critique Summary:**
    - Provide critiques of the hypothesis, drawing on the findings from the previous sections.
    - You must list individual, concrete limitations. Do not follow the criteria categories. 

---

Articles:
{related_articles}

Hypothesis:
```
{hypothesis}
```\
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

Response (Provide reasoning. End with: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".)\
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

def initial_reviewer(llm, hypotheses):

    results = []

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]

        initial_review_input = initial_review_prompt.format(hypothesis=hyp_full)
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

        print(f"initial review of id {id_}: {llm_result}\n##########################################") # needs improvement in logging (e.g. save in cache dir)

        hyp_dict["initial_review"] = llm_result
        results.append(hyp_dict)

    return results

def full_reviewer(llm, hypotheses):
    
    # # load model for reranker
    # tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "ncbi/MedCPT-Cross-Encoder"
    # )
    # model = model.eval().cuda().requires_grad_(False)

    # class & functions
    Base = declarative_base()

    class JournalImpactFactor(Base):
        __tablename__ = "journal_impact_factors"

        id = Column(Integer, primary_key=True, autoincrement=True)
        issn_linking = Column(String, unique=True, index=True, nullable=False)
        impact_factor = Column(Float)

    class PubMedArticle(Base):
        __tablename__ = "pubmed_articles"
        pmid = Column(Integer, primary_key=True, nullable=False)
        pmc_id = Column(String)
        pmc_type = Column(String)
        doi = Column(String)
        title = Column(String)
        abstract = Column(String)
        authors = Column(ARRAY(String))
        mesh_terms = Column(ARRAY(String))
        publication_types = Column(ARRAY(String))
        keywords = Column(ARRAY(String))
        chemical_list = Column(ARRAY(String))
        pubdate = Column(Integer)
        journal = Column(String)
        medline_ta = Column(String)
        nlm_unique_id = Column(String)
        issn_linking = Column(String)
        country = Column(String)
        references = Column(ARRAY(Integer))
        delete = Column(Boolean)
        languages = Column(ARRAY(String))
        vernacular_title = Column(String)

        journal_metrics = relationship(
            "JournalImpactFactor",
            primaryjoin="PubMedArticle.issn_linking == JournalImpactFactor.issn_linking",
            foreign_keys=[issn_linking],
            uselist=False,
        )

        @property
        def impact_factor(self):
            return self.journal_metrics.impact_factor if self.journal_metrics else None

    class PubMedChunk(Base):
        __tablename__ = "pubmed_chunks"

        id = Column(Integer, primary_key=True, autoincrement=True)
        pmid = Column(Integer, ForeignKey("pubmed_articles.pmid"), nullable=False)
        chunk_index = Column(Integer, nullable=False)
        chunk_text = Column(String, nullable=False)

        article_relation = relationship(
            "PubMedArticle", foreign_keys=[pmid], uselist=False
        )

        @property
        def article(self):
            return self.article_relation

    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"[^\w\s]", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def retrieve_chunks(text: str, limit: int = 1000) -> list[PubMedChunk]:
        return (
            session.query(
                func.paradedb.score(PubMedChunk.id).label("score"), PubMedChunk
            )
            .filter(PubMedChunk.chunk_text.op("@@@")(sanitize_text(text)))
            .order_by(func.paradedb.score(PubMedChunk.id).desc())
            .limit(limit)
            .all()
        )
    
    # retrieve & rerank
    engine = create_engine("postgresql://user:1234@localhost/pubmed_refdb")
    Base.metadata.create_all(engine)
    session = sessionmaker(autoflush=True, bind=engine)()
    
    results = []

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]
        hyp_main = hyp_dict["hyp_main"]

        chunks = retrieve_chunks(hyp_main)
        # encoded = tokenizer(
        # [[hyp_main, chunk.chunk_text] for _, chunk in chunks],
        # truncation=True,
        # padding=True,
        # return_tensors="pt",
        # max_length=512,
        # )
        # logits = model(**encoded.to("cuda")).logits.squeeze(dim=1)
        # reranking_indices = logits.argsort(descending=True).cpu().numpy()

        # original_ranking = np.arange(len(reranking_indices))
        # rrf_score = 1 / (30 + np.argsort(reranking_indices)) + 1 / (30 + original_ranking)
        # rrf_indices = np.argsort(-rrf_score)

        # Collect and group the retrieved article chunks.
        related_articles = defaultdict(list)
        
        ### with reranking
        # for i in rrf_indices:
        #     related_articles[chunks[i][1].pmid].append(chunks[i][1])
        #     if len(related_articles) >= 20:
        #         break

        ### without reranking
        for chunk in chunks:
            related_articles[chunk[1].pmid].append(chunk[1])
            if len(related_articles) >= 10:
                break

        related_articles_text = ""
        for i, xs in enumerate(related_articles.values()):
            chunk_texts = [
                "```\n" + "\n".join(x.chunk_text.splitlines()[1:]).strip() + "\n```"
                for x in xs[:10]
            ]
            chunk_texts = "\n".join(chunk_texts)
            related_articles_text += f"[{i + 1}] {xs[0].article.title}\n{chunk_texts}\n\n"
            
        # Run LLM.
        full_review_input = full_review_prompt.format(
            related_articles=related_articles_text, hypothesis=hyp_full
        )
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_review_input},
        ]
        llm_result = llm.chat(input_messages)

        print(f"full review of id {id_}: {llm_result}\n##########################################") # needs improvement in logging (e.g. save in cache dir)

        hyp_dict["full_review"] = llm_result
        hyp_dict["related_articles_text"] = related_articles_text
        results.append(hyp_dict)

    return results

def deep_reviewer(llm, hypotheses):

    results = []

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]

        deep_review_input = deep_review_prompt.format(hypothesis=hyp_full)
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

        print(f"deep review of id {id_}: {llm_result}\n##########################################") # needs improvement in logging (e.g. save in cache dir)

        hyp_dict["deep_review"] = llm_result
        results.append(hyp_dict)

    return results

def observation_reviewer(llm, hypotheses):

    results = []

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]
        related_articles_text = hyp_dict["related_articles_text"]

        observation_review_input = observation_review_prompt.format(article=related_articles_text, hypothesis=hyp_full)
        input_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": observation_review_input
            }
        ]
        llm_result = llm.chat(input_messages) # no return format for now

        print(f"observation review of id {id_}: {llm_result}\n##########################################") # needs improvement in logging (e.g. save in cache dir)

        hyp_dict["observation_review"] = llm_result
        results.append(hyp_dict)

    return results

def simulation_reviewer(llm, hypotheses):

    results = []

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]

        simulation_review_input = simulation_review_prompt.format(hypothesis=hyp_full)
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

        print(f"simulation review of id {id_}: {llm_result}\n##########################################") # needs improvement in logging (e.g. save in cache dir)

        hyp_dict["simulation_review"] = llm_result
        results.append(hyp_dict)
    
    return results

def tournament_reviewer(llm):
    return
