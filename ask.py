#!/usr/bin/env python3

import os
import sys
import requests
import json
import argparse
import asyncio
from contextlib import asynccontextmanager
from prompt_toolkit import PromptSession, ANSI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import Tool
import mdv

GREEN   = '\033[92m'
CYAN    = '\033[36m'
BR_CYAN = '\033[96m'
RED     = '\033[31m'
DK_GREY = '\033[90m'
COL_END = '\033[0m'


# MCP server config, read from mcp.json file at startup
mcp_server_config: dict = {}

# tools fetched from MCP servers at startup: a flat list of (server_name, tool) tuples
mcp_tools: list[tuple[str, Tool]] = []


@asynccontextmanager
async def _mcp_session(config: dict):
  # determine the type of transport from the server's config
  if config.get('command'):
    transport = 'stdio' 
  elif config.get('url'):
    transport = 'http'
  else:
    transport = None

  # open a connection to the MCP server and yield an initialized session
  if transport == 'stdio':
    params = StdioServerParameters(command=config['command'], args=config.get('args', []), env=config.get('env'), cwd=config.get('cwd'))
    with open(os.devnull, 'w') as devnull:
      async with stdio_client(params, errlog=devnull) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
          await session.initialize()
          yield session

  elif transport == 'http':
    async with streamable_http_client(config['url']) as (read_stream, write_stream):
      async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        yield session

  else:
    raise ValueError(f"invalid MCP server config")

async def _fetch_mcp_tools(config: dict) -> list:
  async with _mcp_session(config) as session:
    result = await session.list_tools()
    return result.tools

async def _call_mcp_tool(config: dict, tool_function: str, tool_args: dict) -> dict:
  async with _mcp_session(config) as session:
    result = await session.call_tool(tool_function, tool_args)
    if result.structured_content is not None:
      return result.structured_content
    # if no structured content fall back to concatenating any text content blocks
    text = ''.join(block.text for block in result.content if block.type == 'text')
    return { 'result': text }

def connect_mcp_servers():
  # connect to the MCP server, fetch its tool list, then disconnect
  global mcp_tools
  mcp_tools = []
  if not mcp_server_config:
    return
  
  for server_name in mcp_server_config:
    print(f'{DK_GREY}Connecting to MCP server: {server_name}{COL_END} ', end='', flush=True)
    try:
      tools = asyncio.run(_fetch_mcp_tools(mcp_server_config[server_name]))
      for tool in tools:
        mcp_tools.append((server_name, tool))
      print(f'{DK_GREY}✔{COL_END}')
    except Exception as err:
      print(f'{DK_GREY}✘\n{RED}Failed to connect to MCP server {server_name}: {err}{COL_END}')

def _validate_mpc_server_config(config) -> str | None:
  if not isinstance(config, dict):
    return "entry must be an object"

  has_command = bool(config.get('command'))
  has_url = bool(config.get('url'))

  if not has_command and not has_url:
    return "must have either 'command' (stdio) or 'url' (http)"

  if has_command:
    if not isinstance(config['command'], str):
      return "'command' must be a string"
    if 'args' in config and not isinstance(config['args'], list):
      return "'args' must be a list"
    if 'env' in config and not isinstance(config['env'], dict):
      return "'env' must be an object"
    if 'cwd' in config and not isinstance(config['cwd'], str):
      return "'cwd' must be a string"

  if has_url and not isinstance(config['url'], str):
    return "'url' must be a string"

  return None

def load_mcp_config() -> dict:
  # read the mcp servers config from json file
  global mcp_server_config
  # default to no mcp servers if file not found or other error in config
  mcp_server_config = {}

  # config path should be in same dir as this script
  path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcp.json')
  try:
    with open(path, 'r') as f:
      mcp_config = json.load(f)
  except FileNotFoundError:
    pass
  except json.JSONDecodeError:
    print(RED + f'MCP server config file: {path} does not contain valid JSON. MCP servers will not be used.' + COL_END)
  else:
    servers = mcp_config.get('mcpServers') or {}
    for server_name, server_cfg in servers.items():
      err = _validate_mpc_server_config(server_cfg)
      if err:
        print(RED + f'Invalid MCP server config for {server_name}: {err}. MCP server will not be used.' + COL_END)
      else:
        mcp_server_config[server_name] = server_cfg
  

MODELS = {
  'gemma'         : { 'name': '@cf/google/gemma-4-26b-a4b-it', 'schema': 'openai' },
  'gpt'           : { 'name': '@cf/openai/gpt-oss-20b', 'schema': 'openai' },
  'gpt-large'     : { 'name': '@cf/openai/gpt-oss-120b', 'schema': 'openai' },
  'nemotron-large': { 'name': '@cf/nvidia/nemotron-3-120b-a12b', 'schema': 'openai' },
  'llama-small'   : { 'name': '@cf/meta/llama-3.2-3b-instruct', 'schema': 'llama' },
  'llama'         : { 'name': '@cf/meta/llama-4-scout-17b-16e-instruct', 'schema': 'llama' },
  'llava'         : { 'name': '@cf/llava-hf/llava-1.5-7b-hf', 'schema': 'image_description' }
}


def get_creds() -> dict[str,str]:
  cloudFlareAccount = os.environ.get('CLOUDFLARE_ACCOUNT')
  if cloudFlareAccount == None:
    print('You need to set CLOUDFLARE_ACCOUNT env var with your account ID.')
    exit(1)
  
  apiToken = os.environ.get('CLOUDFLARE_API_TOKEN')
  if apiToken == None:
    print('You need to set CLOUDFLARE_API_TOKEN with your CloudFlare Workers AI API token.')
    exit(1)

  creds = { 'account': cloudFlareAccount, 'token': apiToken }
  return creds

def get_tools() -> list[dict]:
  # append any tools discovered from the connected MCP server, converted to
  # the OpenAI-compatible function-calling schema.
  tools = []
  for server_name, tool in mcp_tools:
    tools.append({
      'type': 'function',
      'function': {
        'name': f'{server_name}__{tool.name}',
        'description': tool.description or '',
        'parameters': tool.input_schema
      }
    })
  return tools


def run_tool(tool_function, tool_args) -> dict:
  # call the tool on the connected MCP server and return its result.
  try:
    server_name, function_name = tool_function.split('__',1)
    return asyncio.run(_call_mcp_tool(mcp_server_config[server_name], function_name, tool_args))
  except Exception as err:
    print(RED + f'Failed to call MCP tool {tool_function}: {err}' + COL_END)
    return { 'error': str(err) }


def post_llm_request(creds: dict[str, str], model: str, payload: dict[str, str]):
  # /ai/run endpoint is the dynamic request format endpoint, supported for all models
  urlTemplate = 'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{cf_model}'
  url = urlTemplate.format(account=creds['account'], cf_model=MODELS[model]['name'])
  hdrs = {'Authorization': 'Bearer '+ creds['token']}

  try:
    response = requests.post(url, headers=hdrs, data=json.dumps(payload))
    response.raise_for_status()
  except Exception as err:
    print('Request failed.\n', err)
    return None

  result = response.json().get('result')
  if result is None:
    print('Unexpected response from Workers AI:\n', response.json())
    return None
  return result



def prompt_llm(creds: dict[str, str], model: str, messages: list[dict[str, str]], token_multiplier: int):
  payload = { 
    'messages': messages
  }

  # assuming all our 'openai' type models support tool calling
  if MODELS[model]['schema'] == 'openai':
    payload['tools'] = get_tools()

  if MODELS[model]['schema'] == 'openai':
    payload['max_tokens'] = 512 + 200 * token_multiplier

  result = post_llm_request(creds, model, payload)
  if result is None:
    return '','',False,[]
  return parse_result(model, result)


def parse_result(model: str, result):
  truncated = False
  reasoning = ''
  tool_calls = []

  if MODELS[model]['schema'] == 'openai':
    choice = result.get('choices', [None])[0]
    answer = choice.get('message',{}).get('content')
    reasoning = choice.get('message',{}).get('reasoning_content')
    tool_calls = choice.get('message',{}).get('tool_calls')
    if (choice.get('finish_reason') == 'length'):
      truncated = True
      if answer is None:
        answer = ''

  elif MODELS[model]['schema'] == 'llama':
    answer = result.get('response')
    # our system prompt asks to append special stop char, so if it's not there then response is (probably) truncated
    truncated = '⏎' not in answer[-10:] and len(answer) > 800

  elif MODELS[model]['schema'] == 'image_description':
    answer = result.get('description')

  if answer is None and tool_calls is []:
    print(f'Failed to parse {model} response:', result)
    answer = ''

  return answer, reasoning, truncated, tool_calls

def get_summary(creds: dict[str, str], messages: list[dict[str, str]]):
  # summarise the conversation history, using a smaller model
  system_message = { 'role': 'system', 'content': 'You are a concise chat bot' }
  user_message = { 'role': 'user', 'content': 'Condense the conversation history into a brief summary (100 words or less) that captures the essential information and context, focussing mainly on details provided by the user' }
  messages = [system_message] + messages + [user_message]
  answer, _, _, _ = prompt_llm(creds, 'llama-small', messages, 1)
  return answer

def print_markdown(text: str):
  # convert markdown to ANSI
  if len(text):
    ansiText = mdv.main(text, theme='963.4449') #theme='757.2295'
    print(ansiText)


def chat(prompt: str, model: str, one_shot: bool):
  creds = get_creds()
  system_message = { 'role': 'system', 'content': "You are a helpful assistant called Bob. Please answer questions briefly and professionally, without asking follow up questions. Format all responses using markdown. Don't use markdown tables for large amounts of text. You must finish each answer with a '⏎' stop character." }
  conversation = []
  conversation.append(system_message)

  # Long responses will be truncated, so when we detect truncation we do a continuation prompt.
  # System prompt asks for a stop character so we can detect truncation, or for OpenAI etc. we check for it reaching max_tokens.
  # Either way, we limit the number of continuation requests in case the model fails to stop.
  truncated = False
  truncation_count = 0
  TRUNCATION_LIMIT = 5

  # tracks the conversation message index where tool_calls started, so tool calls can be pruned afterwards
  tool_call_start = None

  if len(prompt):
    conversation.append({ 'role': 'user', 'content': prompt })
    ask_prompt = False
  else:
    ask_prompt = True


  while True:
    if ask_prompt:
      # get user prompt 
      session = PromptSession(
        ANSI(f"{GREEN}Ask: "), 
        multiline=False # use True to allow CR in input
      )
      prompt = session.prompt()
      if (prompt.lower() == 'clear'):
        conversation = []
        conversation.append(system_message)
        print(BR_CYAN + 'Conversation cleared.' + COL_END + '\n')
        ask_prompt = True
        continue
      conversation.append({ 'role': 'user', 'content': prompt })

    token_multiplier = truncation_count if truncated else 1
    answer, reasoning, truncated, tool_calls = prompt_llm(creds, model, conversation, token_multiplier)
    
    if tool_calls:
      if tool_call_start is None:
        tool_call_start = len(conversation)
      conversation.append({ 'role': 'assistant', 'content': '', 'tool_calls': tool_calls })
      for tc in tool_calls:
        fn = tc['function']['name']
        args = json.loads(tc['function']['arguments'])
        print(f'{DK_GREY}Calling tool: {fn.replace('__', ' ')} {args}{COL_END}\n')
        tool_result = run_tool(fn, args) 
        conversation.append({ 'role': 'tool', 'tool_call_id': tc['id'], 'content': json.dumps(tool_result) })
      ask_prompt = False
      continue
    
    # if we were making tool_calls, then at this point the tool calling is over and answer should contain the final result 
    if tool_call_start is not None:
      # we can now prune the tool_calls and tool results to save tokens.
      conversation = conversation[:tool_call_start]
      tool_call_start = None
    
    if answer is None:
      answer = ''
    print_markdown(answer)

    truncated = truncated and truncation_count < TRUNCATION_LIMIT

    if prompt.lower() == 'exit' or prompt.lower() == 'bye' or (one_shot and not truncated):
      break

    if truncated:
      print(f'...\n')
      if truncation_count > 0:
        # already done a continuation prompt, so strip the last user and assistant prompt
        conversation = conversation[:-2]

      assistant_message = f"{reasoning}\n\n{answer}".strip()
      conversation.append({ 'role': 'assistant', 'content': assistant_message })
      conversation.append({ 'role': 'user', 'content': 'Please continue.' })
      truncation_count += 1
      ask_prompt = False
    else:
      conversation.append({ 'role': 'assistant', 'content': answer })
      truncation_count = 0
      ask_prompt = True

    # to keep tokens down, compact oldest messages in the conversation, replacing with a summary of those messages
    if len(conversation) > 20:
      summary = get_summary(creds, conversation[1:11])   # ignoring initial system prompt, summarise oldest 10 messages (i.e. 5 exchanges)
      summary_message = { 'role': 'assistant', 'content': summary }
      conversation = [system_message] + [summary_message] + conversation[11:]
      print(BR_CYAN + 'Conversation compacted.' + COL_END + '\n')


def image_to_text(prompt: str, image_filename: str, one_shot: bool):
  if len(prompt)==0:
    prompt = f'Please describe the image in the attached file {image_filename}'
  with open(image_filename, "rb") as file:
    blob = file.read()
    payload = {
      "image": list(blob),
      "prompt": prompt,
      "max_tokens": 1024,
    }

  creds = get_creds()
  model = 'llava'
  result = post_llm_request(creds, model, payload)
  answer, _, _, _ = parse_result(model, result)
  print_markdown(answer)




try:

  parser = argparse.ArgumentParser(description='Ask: your personal command line chatbot')
  parser.add_argument('text', type=str, nargs='*', default=[], help='initial question to ask.')
  parser.add_argument('-q', '--quick', action='store_true', help='quick mode. Ask a single question then exit. If not set, defaults to conversation mode.')
  chat_models = [key for key, value in MODELS.items() if value['schema'] != 'image_description']
  parser.add_argument('-m', '--model', type=str, default='llama', choices=chat_models,  help='choose which LLM to use for chat. Defaults to llama 4 17b model.')
  parser.add_argument('-i', '--image', type=str, default='',  help='path to an image file, for image-to-text inference (using llava model).')
  args = parser.parse_args()
  # take any command line text
  prompt = ' '.join(args.text)
  if not sys.stdin.isatty():
    # append any piped stdin
    prompt = prompt + ' ' + sys.stdin.read()
  if len(args.image):
    if not os.path.exists(args.image):
      argparse.ArgumentParser().error(f"Image file '{args.image}' not found.")
    image_to_text(prompt, args.image, args.quick)
  else:
    load_mcp_config()
    connect_mcp_servers()
    chat(prompt, args.model, args.quick)
except (KeyboardInterrupt, EOFError):
  print(RED + '\nExiting...' + COL_END + '\n')
  sys.exit(0)


