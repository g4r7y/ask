# Ask: simple command line AI chatbot

## Overview

This script provides a command line LLM chatbot using CloudFlare Workers AI API.

## Requirements

* Python 3.x
* CloudFlare account with Workers AI API enabled

## Setup

The following environment variables need to be set:

* `CLOUDFLARE_ACCOUNT`: Your CloudFlare account ID
* `CLOUDFLARE_API_TOKEN`: Your CloudFlare Workers AI API token

Install required python libs. e.g. using venv:

```bash
python3 -m venv ./myenv
source ./myenv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
./ask.py [prompt]
```

If no arguments are provided, the script enters conversation mode.

Or, you can provide your LLM prompt as command line arguments, in which case the script will answer the prompt and exit. For example:

```bash
./ask.py How many legs does a centipede have?
```

You can also pipe in a prompt from another command. For example:

```bash
echo "Why is the sky blue?" | ./ask.py
```

And you can combine command line arguments with file redirected to stdin:

```bash
./ask.py Please can you review this file < bad-code.js
```

## Conversation Mode

In conversation mode, the script will continuously prompt the user for input, building up the conversation context.

You can use the following commands:
* `clear`: Clears the conversation history and starts a new conversation.
* `exit` : Exits the script.

## Models

You can change the model in the script to other models supported by [Workers AI](https://developers.cloudflare.com/workers-ai/models/). OpenAI models are not currently supported as they use a different request format.

