# Ask: simple command line AI chatbot

## Overview

This script provides a command line LLM chatbot using CloudFlare Workers AI.

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
./ask.py --help
```

### Conversation mode

```bash
./ask.py
```

By default, conversation mode is used. In conversation mode, it will continuously prompt the user for input, building up the conversation context.

You can use the following commands:
* `clear`: Clears the conversation history and starts a new conversation.
* `exit` : Exits the script.

### Quick mode

```bash
./ask.py -q
```
If -q is given, then quick mode is used, where it prompts the LLM once then exits.

### Prompt text

You can provide your prompt text as a command line argument. For example:

```bash
./ask.py -q Tell me a joke
```

You can also pipe in a prompt from another command. For example:

```bash
echo "Why is the sky blue?" | ./ask.py -q
```

And you can combine command line arguments with file redirected to stdin:

```bash
./ask.py Please can you review this code < stuff.js
```

### Models

```bash
./ask.py --model=gpt
```

You can specify which model to use. Use the --help option to see the available models. These are a subset of the models supported by [Workers AI](https://developers.cloudflare.com/workers-ai/models/).

### Image to text mode

Use the -i option for image-to-text inference. In this mode, the llava model is used.

```bash
./ask.py -i cat.png 
```

You can provide your own prompt for a more specific description:

```bash
./ask.py -i vinyl.jpg Please extract the artist, title and record label
```

