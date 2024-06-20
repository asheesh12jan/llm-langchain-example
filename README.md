# Run using Python command
```
python3 app.py
```

# Run using Docker

Go to project root folder
### Build
Build docker image
```
docker build -t llm-langchain-example .
```

### Run
Run docker image. Pass Open API key and data folder as env variable
```
docker run -e DATA_FOLDER="data" -e OPENAI_API_KEY="sk-proj-aUNoORbfhekhfjkewhfkjwebhgfjkw" llm-langchain-example
```
