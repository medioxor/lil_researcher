## What is it?
This is a multi-agent Deep Research implementation that's model agnostic.

## Setup
Firstly you'll need the following set in the `.env` file:

- `MODEL` is the name of the model to use e.g. https://ollama.com/MFDoom/deepseek-r1-tool-calling would be "MFDoom/deepseek-r1-tool-calling:14b"
- `CONTEXT_SIZE` is the context size of the model
- `TEMPERATURE` is the temperature of the model
- `SERPAPI_API_KEY` is a https://serpapi.com/ API key

An example can be seen below:
```
MODEL=MFDoom/deepseek-r1-tool-calling:14b
CONTEXT_SIZE=4000
TEMPERATURE=0.2
SERPAPI_API_KEY=a6asgbqwig1y3gkag11234125415rasfq
```

After that ensure docker is installed with the compose plugin and execute the following:

```
docker compose up
```