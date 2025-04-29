import re
import pubmed_parser as pp
import torch
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Integer,
    ForeignKey,
    Float,
    String,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uuid

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

Proposed hypothesis in square brackets (i.e., [HYPOTHESIS: ...]), followed by a detailed description for domain experts:\
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

def extract_main_hypothesis(text):
    pattern = r"\[\s*hypothesis\s*:\s*(.*?)\s*\]"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

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

def retrieve_from_db(goal, top_k):

    # # load model for reranker
    # tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    # model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
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

        article_relation = relationship("PubMedArticle", foreign_keys=[pmid], uselist=False)

        @property
        def article(self):
            return self.article_relation
        
    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"[^\w\s]", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def retrieve_chunks(text: str, limit: int = 1000) -> list[PubMedChunk]:
        return (
            session.query(func.paradedb.score(PubMedChunk.id).label("score"), PubMedChunk)
            .filter(PubMedChunk.chunk_text.op("@@@")(sanitize_text(text)))
            .order_by(func.paradedb.score(PubMedChunk.id).desc())
            .limit(limit)
            .all()
        )

    def find_article_by_pmid(pmid: int) -> PubMedArticle:
        return session.query(PubMedArticle).where(PubMedArticle.pmid == pmid).one()
    
    # retrieve & rerank
    engine = create_engine("postgresql://user:1234@localhost/pubmed_refdb")
    Base.metadata.create_all(engine)
    session = sessionmaker(autoflush=True, bind=engine)()

    chunks = retrieve_chunks(goal)
    # encoded = tokenizer(
    #     [[goal, chunk.chunk_text] for _, chunk in chunks],
    #     truncation=True,
    #     padding=True,
    #     return_tensors="pt",
    #     max_length=512,
    # )
    # logits = model(**encoded.to("cuda")).logits.squeeze(dim=1)
    # reranking_indices = logits.argsort(descending=True).cpu().numpy()

    topk_texts = []

    ### with reranking
    # for i in reranking_indices[:top_k]:
    #     score, chunk = chunks[i]
    #     topk_texts.append(chunk.chunk_text)

    ### without reranking
    for chunk in chunks[:top_k]:
        topk_texts.append(chunk[1].chunk_text)
    
    return topk_texts

def explorator(llm, goal, preferences, source_hypothesis, articles_with_reasoning, num_init_hyp):
    
    results = []
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

    for _ in range(num_init_hyp):
        llm_result = llm.chat(input_messages) # no return format for now
        main_hypothesis = extract_main_hypothesis(llm_result)
        results.append({
            "id_": str(uuid.uuid4()),
            "hyp_full": llm_result,
            "hyp_main": main_hypothesis
        })

    return results

def debate_simulator(llm, attributes, goal, preferences, hyp_after_meta_review, max_turns):

    results = []
    system_prompt = "You are an expert tasked with developing and refining a scientific hypothesis."

    for hyp_dict in hyp_after_meta_review:
        
        transcript = "" # updated with iteration
        reviews_overview = hyp_dict["meta_review"]
        prev_hyp_id = hyp_dict["id_"]
        final_hypothesis = None

        for turn in range(1, max_turns + 1):
            # Fill in the transcript into the static prompt template
            prompt_input = scientific_debate_prompt.format(
                idea_attributes=attributes,
                goal=goal,
                preferences=preferences,
                reviews_overview=reviews_overview,
                transcript=transcript.strip()
            )
            input_messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt_input
                }
            ]
            # Call the LLM
            llm_result = llm.chat(input_messages)
            transcript += f"\n[Expert {turn}]: {llm_result}\n"

            # Check for termination
            if "HYPOTHESIS" in llm_result:
                final_hypothesis = llm_result.split("HYPOTHESIS", 1)[-1].strip()
                break
        
        results.append({
            "id_": str(uuid.uuid4()),
            "hyp_full": final_hypothesis,
            "hyp_main": final_hypothesis, # assume that debate_simulator returns clean hypothesis
            "prev_id": prev_hyp_id
        })

    return results

def assumption_identifier(llm, goal):
    return

def research_expander(llm, visited_hyp_list, reviews_overview):
    return