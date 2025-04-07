# co_scientist

## Setup

```shell
# create a new environment
conda create -n coscientist python==3.9.21
conda activate coscientist

# install openai
pip install openai==1.70.0
```

Make sure to enter your own api keys in api_config.py.

## Quickstart

** Deep research integration is in progress; in the meantime, please use the following prompt to obtain relevant articles (with reasoning), and save them to _articles_with_reasoning.txt_ (sample is given in _articles_with_reasoning_sample.txt_)  
```
    Given a specific research goal, search the web for relevant and credible articles that contribute to achieving or addressing that goal. For each article you select, provide a reasoning that explains in detail how the article supports, informs, or relates to the research goal. The reasoning should reference specific elements of the article (e.g., findings, arguments, data, or methodology) and clearly articulate the connection to the research objective. 
    Return a list of (article, reasoning) pairs in reverse chronological order, beginning with the most recent analysis or publication. Each reasoning should demonstrate a thoughtful and well-supported link between the article’s content and the research goal.
    
    Output Format:
        - Title: [Title]
        - Article: [Article]
        - Reasoning: [Detailed reasoning that explains the article’s relevance to the research goal]

    Research Goal: {your research goal here}
```

1. Write your research goal in _research_goal.txt_ (sample is given in _research_goal_sample.txt_)
2. Execute the command below:
```shell
python run_pipeline.py --llm gpt-4o --input_path research_goal.txt --articles_with_reasoning_path articles_with_reasoning.txt --save_path results --log_path logs
```