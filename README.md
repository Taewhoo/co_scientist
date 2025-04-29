# co_scientist

## Setup

```shell
# create a new environment
conda create -n coscientist python==3.9.21
conda activate coscientist

# install requirements
pip install -r requirements.txt
```

Make sure to enter your own api keys in api_config.py.

## Quickstart

1. Write your research goal in _research_goal.txt_ (sample is given in _research_goal_sample.txt_)
2. Execute the command below:
```shell
python run_pipeline.py --llm gpt-4o --input_path research_goal.txt --save_path results --log_path logs --num_init_hyp 8
```