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
import re
import uuid

search_query_prompt = """\
You are given a scientific hypothesis and a list of match results where it was judged to be weaker than another competing hypothesis.

Your task is to:
1. Analyze the hypothesis in the context of the research goal and match results to identify possible weaknesses, gaps, or limitations in the given hypothesis.
2. Based on those weaknesses, list up search queries that can be used to retrieve relevant scientific articles for further investigation or refinement.

Goal: {goal}

Hypothesis: {hypothesis}

Match results (lost): {match_results}

Now, respond in the following format:

- Weakness Analysis: [weaknesses]

- Search Queries: [search queries separated by newlines]
"""
enhancement_prompt = """\
Given a scientific hypothesis, analyze its weaknesses in the context of the research goal and insights from the retrieved scientific articles.

Your task is to:
1. Use the previously identified weaknesses to suggest specific improvements or elaborations that would strengthen the hypothesis and fill any reasoning gaps.
2. Incorporate relevant insights from the retrieved articles to make the revised hypothesis more detailed, coherent, and scientifically grounded.

Goal: {goal}

Original Hypothesis: {hypothesis}

Weakness Analysis: {weaknesses}

Retrieved Articles: {articles}

Now, respond in the following format:

- Suggested Improvements: [improvements]

- Revised Hypothesis: [rewritten hypothesis]
"""
feasibility_prompt = """\
Your task is to refine the provided conceptual idea, enhancing its practical implementability by leveraging contemporary technological capabilities. Ensure the revised concept retains its novelty, logical coherence, and specific articulation.

Goal: {goal}

Guidelines:
1. Begin with an introductory overview of the relevant scientific domain.
2. Provide a concise synopsis of recent pertinent research findings and related investigations, highlighting successful methodologies and established precedents.
3. Articulate a reasoned argument for how current technological advancements can facilitate the realization of the proposed concept.
4. CORE CONTRIBUTION: Develop a detailed, innovative, and technologically viable alternative to achieve the objective, emphasizing simplicity and practicality.

Evaluation Criteria:
{preferences}

Original Conceptualization:
{hypothesis}

Output Format:

1. [Response to Guideline 1]
2. [Response to Guideline 2]
3. [Response to Guideline 3]
4. [Response to Guideline 4, FINAL HYPOTHESIS ONLY]
"""
outofthebox_prompt = """\
Goal: {goal}

Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Summarize recent findings and pertinent research, highlighting successful approaches.
3. Identify promising avenues for exploration that may yield innovative hypotheses.
4. CORE HYPOTHESIS: Develop a detailed, original, and specific single hypothesis for achieving the stated goal, leveraging analogous principles from the provided ideas. This should not be a mere aggregation of existing methods or entities. Think out-of-the-box.

Criteria for a robust hypothesis:
{preferences}

Inspiration may be drawn from the following concepts (utilize analogy and inspiration, not direct replication):
{hypothesis}

Output Format:

1. [Response to Instruction 1]
2. [Response to Instruction 2]
3. [Response to Instruction 3]
4. [Response to Instruction 4, FINAL HYPOTHESIS ONLY]
"""

def retrieve_from_db(query, top_k):

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

    chunks = retrieve_chunks(query)
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

def enhancement_grounding(llm, research_goal, hypotheses):

    system_prompt_enhancement_grounding = "You are an expert in scientific reasoning and hypothesis refinement."

    # identify weaknesses for given hypothesis
    results = []

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        hyp_full = hyp_dict["hyp_full"]
        ranking_lose_results = hyp_dict["ranking_lose_results"]

        # first, generate search queries
        search_query_input = search_query_prompt.format(goal=research_goal, hypothesis=hyp_full, match_results=ranking_lose_results)
        input_messages = [
            {"role": "system", "content": system_prompt_enhancement_grounding},
            {"role": "user", "content": search_query_input},
        ]

        # ensure parsing (temperature=0.2)
        parsed = False
        while not parsed:
            weakness_result = llm.chat(input_messages)
            match = re.search(r"Search Queries\s*:\s*(.+)", weakness_result, re.IGNORECASE | re.DOTALL)
            if match:
                parsed = True
        
        queries_block = match.group(1).strip()
        queries = [line.strip() for line in queries_block.splitlines() if line.strip()]

        # second, retrieve articles
        total_articles = []
        for query in queries:
            try:
                retrieved_articles = retrieve_from_db(query=query, top_k=5)
                retrieved_articles_merged = "\n".join(retrieved_articles)
                total_articles.append(f"Query: {query}\nArticles:\n{retrieved_articles_merged}")
            except:
                continue
        
        # third, suggest improvements and elaborate on details to fill reasoning gaps, resulting in a revised hypothesis
        enhancement_input = enhancement_prompt.format(goal=research_goal, hypothesis=hyp_full, weaknesses=weakness_result, articles="\n\n".join(total_articles))  
        input_messages = [
            {"role": "system", "content": system_prompt_enhancement_grounding},
            {"role": "user", "content": enhancement_input},
        ]

        parsed = False
        while not parsed:
            enhancement_result = llm.chat(input_messages)
            match = re.search(
                r"Revised Hypothesis\s*:\s*(.*)", 
                enhancement_result.strip(), 
                re.IGNORECASE | re.DOTALL
            )
            if match:
                parsed = True
        
        hyp_revised = match.group(1).strip()
        results.append({
            "id_": str(uuid.uuid4()),
            "hyp_full": hyp_revised,
            "hyp_main": hyp_revised,
            "prev_id": id_
        })

    # return new hypotheses (keys "id_", "hyp_full") only for now
    return results

def feasibility_improver(llm, research_goal, preferences, hypotheses):

    results = []
    system_prompt = "You are an expert in scientific research and technological feasibility analysis."

    for hyp_dict in hypotheses:

        id_ = hyp_dict["id_"]
        feasibility_input = feasibility_prompt.format(goal=research_goal,preferences=preferences, hypothesis=hyp_dict["hyp_full"])
        input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": feasibility_input
        }
        ]

        parsed = False
        while not parsed:
            feasibility_result = llm.chat(input_messages)
            match = re.search(
                r"4\.\s*(.*)", 
                feasibility_result.strip(), 
                re.IGNORECASE | re.DOTALL
            )
            if match:
                parsed = True
        
        hyp_revised = match.group(1).strip()
        results.append({
            "id_": str(uuid.uuid4()),
            "hyp_full": hyp_revised,
            "hyp_main": hyp_revised,
            "prev_id": id_
        })

    return results

def inspiration(llm, research_goal, hypotheses):
    return

def combination(llm, research_goal, hypotheses):
    return

def simplification(llm, research_goal, hypotheses):
    return

def out_of_box(llm, research_goal, preferences, hypotheses):

    results = []
    system_prompt = "You are an expert researcher tasked with generating a novel, singular hypothesis inspired by analogous elements from provided concepts."

    for hyp_dict in hypotheses:
        
        id_ = hyp_dict["id_"]
        outofbox_input = outofthebox_prompt.format(goal=research_goal,preferences=preferences, hypothesis=hyp_dict["hyp_full"])
        input_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": outofbox_input
        }
        ]

        parsed = False
        while not parsed:
            feasibility_result = llm.chat(input_messages)
            match = re.search(
                r"4\.\s*(.*)", 
                feasibility_result.strip(), 
                re.IGNORECASE | re.DOTALL
            )
            if match:
                parsed = True
        
        hyp_revised = match.group(1).strip()
        results.append({
            "id_": str(uuid.uuid4()),
            "hyp_full": hyp_revised,
            "hyp_main": hyp_revised,
            "prev_id": id_
        })

    return results

#####

def evolve_hypotheses(llm, research_goal, preferences, hypotheses):

    results = []
    # results += hypotheses # evolution agent preserves initial hypotheses

    results += enhancement_grounding(llm, research_goal, hypotheses)
    results += feasibility_improver(llm, research_goal, preferences, hypotheses)
    # results += inspiration(llm, research_goal, hypotheses)
    # results += combination(llm, research_goal, hypotheses)
    # results += simplification(llm, research_goal, hypotheses)
    results += out_of_box(llm, research_goal, preferences, hypotheses)

    return results