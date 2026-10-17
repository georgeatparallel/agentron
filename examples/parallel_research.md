# Web research with Parallel MCP

[parallel_research.py](parallel_research.py) connects Agentron's custom tools to
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) over
Streamable HTTP. It provides web search and page fetching without a Parallel
account or API key. Anonymous searches use Fast mode. The free endpoint has usage
limits and is intended for exploration and light use.

## Setup

Use Python 3.12 or newer. From the repository root, install Agentron and the
example's optional MCP dependency in a virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install agentron 'mcp>=1.26,<2'
```

This uses the published Agentron package, which includes its communication
backend. Running the agent also requires Node.js 20.19 or newer, as described in
the [main README](../README.md#requirements).

## Try search and fetch directly

These commands do not use a language model or require model credentials:

```sh
python examples/parallel_research.py --search 'Python 3.14 release highlights'
python examples/parallel_research.py --fetch 'https://docs.python.org/3/whatsnew/3.14.html'
```

## Ask an agent

Configure credentials for the model using Agentron's normal
[authentication setup](../README.md#authentication). For example, set
`OPENROUTER_API_KEY` for the default `openrouter:openrouter/free` model, then run:

```sh
python examples/parallel_research.py 'What changed in Python 3.14? Search for the release notes, read them, and cite the sources.'
```

Select another configured model with `--model`, for example
`--model openai:gpt-5.4`. Model access and billing follow that provider's rules.
The example's Parallel connection remains anonymous even if you have a Parallel
API key in your environment or saved credentials.

The example keeps the MCP session open while the agent runs and registers two
async functions as regular Agentron tools. `web_search` accepts an objective and
keyword queries; `web_fetch` reads one URL at a time and returns excerpts.
Both return the server's text, including source URLs, to the agent. Tool errors
are marked as failures by Agentron, and requests have a 60-second timeout. A
single conversation identifier is reused for related search and fetch calls.

To reuse these tools in another script, keep the agent inside
`async with connect_parallel() as tools:` and pass
`tools=[tools.web_search, tools.web_fetch]` to `make_agent`.

## Tests

With the optional dependency installed, run the focused tests from the repository
root:

```sh
python -m pip install pytest
python -m pytest tests/test_parallel_research.py
```

The tests use a local HTTP fixture and do not require service or model credentials.
