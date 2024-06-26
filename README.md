<h1 align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/restai-logo.png"/>
  </br>RestAI
</h1>

<p align="center">
  <strong>AIaaS (AI as a Service) for everyone. Create agents (projects) and consume them using a simple REST API.</strong>
</p>

<h2 align="center">
  Demo: <a href="https://ai.ince.pt">https://ai.ince.pt</a> Username: <code>demo</code> Password: <code>demo</code>
</h2>

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/out.gif"/>
</div>

## Features

- **Projects**: There are multiple types of agents (projects), each with its own features. ([rag](https://github.com/apocas/restai?tab=readme-ov-file#rag), [ragsql](https://github.com/apocas/restai?tab=readme-ov-file#ragsql), [inference](https://github.com/apocas/restai?tab=readme-ov-file#inference), [vision](https://github.com/apocas/restai?tab=readme-ov-file#vision), [router](https://github.com/apocas/restai?tab=readme-ov-file#router))
- **Users**: A user represents a user of the system. It's used for authentication and authorization (basic auth). Each user may have access to multiple projects.
- **LLMs**: Supports any public LLM supported by LlamaIndex or any local LLM suported by Ollama.
- **VRAM**: Automatic VRAM management. RestAI will manage the VRAM usage, automatically loading and unloading models as needed and requested.
- **API**: The API is a first-class citizen of RestAI. All endpoints are documented using [Swagger](https://apocas.github.io/restai/).
- **Frontend**: There is a frontend available at [restai-frontend](https://github.com/apocas/restai-frontend)

## Project Types

### RAG

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/rag.png" width="750"  style="margin: 10px;"/>
</div>

- **Embeddings**: You may use any embeddings model supported by llamaindex. Check embeddings [definition](modules/embeddings.py).
- **Vectorstore**: There are two vectorstores supported: `Chroma` and `Redis`
- **Retrieval**: It features an embeddings search and score evaluator, which allows you to evaluate the quality of your embeddings and simulate the RAG process before the LLM. Reranking is also supported, ColBERT and LLM based.
- **Loaders**: You may use any loader supported by llamaindex.
- **Sandboxed mode**: RAG agents (projects) have "sandboxed" mode, which means that a locked default answer will be given when there aren't embeddings for the provided question. This is useful for chatbots, where you want to provide a default answer when the LLM doesn't know how to answer the question, reduncing hallucination.
- **Evaluation**: You may evaluate your RAG agent using [deepeval](https://github.com/confident-ai/deepeval). Using the `eval` property in the RAG endpoint.

### RAGSQL

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/ragsql.jpg" width="750"  style="margin: 10px;"/>
</div>

- **Connection**: Supply a MySQL or PostgreSQL connection string and it will automatically crawl the DB schema, using table and column names it’s able to figure out how to translate the question to sql and then write a response.

### Inference

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/inference.png" width="750"  style="margin: 10px;"/>
</div>

### Vision

- **text2img**: RestAI supports local Stable Diffusion and Dall-E. It features prompt boosting, a LLM is internally used to boost the user prompt with more detail.
- **img2text**: RestAI supports LLaVA, BakLLaVA by default.
- **img2img**: RestAI supports InstantID and Qwen-VL default.

#### Stable Diffusion & [InstantID](https://github.com/InstantID/InstantID)

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/vision_sd.png" width="25%"  style="margin: 10px;"/>
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/avatar.png" width="25%"  style="margin: 10px;"/>
</div>

#### LLaVA

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/llava.png" width="25%"  style="margin: 10px;"/>
</div>

### Router

<div align="center">
  <img src="https://github.com/apocas/restai/blob/master/readme/assets/router.png" width="750"  style="margin: 10px;"/>
</div>

- **Routes**: Very similar to Zero Shot React strategy, but each route is a project. The router will route the question to the project that has the highest score. It's useful when you have multiple projects and you want to route the question to the most suitable one.

## LLMs

- You may use any LLM supported by Ollama and/or LlamaIndex.

## Installation

- RestAI uses [Poetry](https://python-poetry.org/) to manage dependencies. Install it with `pip install poetry`.

## Development

- `make install`
- `make dev` (starts restai in development mode)
- `make devfrontend` (starts restai's frontend in development mode)

## Production

- `make install`
- `make prod`

## API

- **Endpoints**: All the API endpoints are documented and available at: [Swagger](https://apocas.github.io/restai/)

## Frontend

- Source code at [https://github.com/apocas/restai-frontend](https://github.com/apocas/restai-frontend).
- `make install` automatically installs the frontend.

## Tests

- Tests are implemented using `pytest`. Run them with `make test`.

## License

Pedro Dias - [@pedromdias](https://twitter.com/pedromdias)

Licensed under the Apache license, version 2.0 (the "license"); You may not use this file except in compliance with the license. You may obtain a copy of the license at:

    http://www.apache.org/licenses/LICENSE-2.0.html

Unless required by applicable law or agreed to in writing, software distributed under the license is distributed on an "as is" basis, without warranties or conditions of any kind, either express or implied. See the license for the specific language governing permissions and limitations under the license.
