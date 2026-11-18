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

### Image to text mode

Use the -i option for image-to-text inference. In this mode, the llava model is used.

```bash
./ask.py -i cat.png 
```

You can provide your own prompt for a more specific description:

```bash
./ask.py -i vinyl.jpg Please extract the artist, title and record label
```

### Prompt commands

Commands can be entered at the chat prompt using the `/` prefix:

* `/cd <dir>`: Change the working directory. No argument shows the current working directory. See [MCP servers](#mcp-servers).
* `/clear`: Clear the conversation history and start a new conversation.
* `/compact`: Compact the conversation history.
* `/model <model>`: Change the current model. No argument shows current model.
* `/exit` : Exit.
  
## Models

You can change model during use with the `/model` command. 
The supported models are a subset of the open-source models provided by [Workers AI](https://developers.cloudflare.com/workers-ai/models/).

You can also specify which model to use on startup:
```bash
./ask.py --model=gpt
```

## MCP servers

You can configure MCP servers to allow the model to use custom tools, such as file system access, web fetch and code editing. Tool usage is supported by most models.

To configure your choice of MCP servers and tools, create a `mcp.json` file in the same directory as the main script and add your MCP configuration. The JSON format closely matches Claude Desktop's MCP Server config. See the [example MCP config](./mcp.json.example) to get started. 

You will be asked for permission when a tool call is made. You can also choose to allow or deny all subsequent calls to the same tool.

## File system access

The working directory is used to restrict MCP tool access to your file system. It is initially the directory you launched from, and can be changed with the `/cd` command. 

If a tool call has a path argument outside the working directory, you are always prompted, even for tool calls that you have allowed always. 

In your MCP config, use the `${cwd}` placeholder in a MCP server's `args`, `cwd`, `env` or `url` parameters - it will be replaced with the working directory whenever a tool is called.
