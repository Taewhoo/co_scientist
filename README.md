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

```shell
python run_pipeline.py --llm gpt-4o --input_path {path to text file with research goal} --save_path results --log_path logs
```