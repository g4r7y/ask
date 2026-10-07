#!/usr/bin/env python3

import inspect
import os
import re
import shutil
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

MODELS = {
  'gemma'         : { 'name': '@cf/google/gemma-4-26b-a4b-it', 'schema': 'openai' },
  'gpt'           : { 'name': '@cf/openai/gpt-oss-20b', 'schema': 'openai' },
  'gpt-large'     : { 'name': '@cf/openai/gpt-oss-120b', 'schema': 'openai' },
  'nemotron-large': { 'name': '@cf/nvidia/nemotron-3-120b-a12b', 'schema': 'openai' },
  'llama-small'   : { 'name': '@cf/meta/llama-3.2-3b-instruct', 'schema': 'llama' },
  'llama'         : { 'name': '@cf/meta/llama-4-scout-17b-16e-instruct', 'schema': 'llama' },
  'llava'         : { 'name': '@cf/llava-hf/llava-1.5-7b-hf', 'schema': 'image_description' }
}

# when size of conversation exceeds this threshold the conversation will be compacted to reduce token usage
AUTO_COMPACT_THRESHOLD = 25

# the system prompt asks for this character at the end of each answer, so we can detect truncation
STOP_CHAR = '⏎'

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
  """Connect to the MCP server, fetch its tool list, then disconnect."""
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
  """Read the mcp servers config from json file."""
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
  """Append any tools discovered from the connected MCP server, converted to
  the OpenAI-compatible function-calling schema.
  """
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
  """Call the tool on the connected MCP server and return its result."""
  try:
    server_name, function_name = tool_function.split('__',1)
    return asyncio.run(_call_mcp_tool(mcp_server_config[server_name], function_name, tool_args))
  except Exception as err:
    print(RED + f'Failed to call MCP tool {tool_function}: {err}' + COL_END)
    return { 'error': str(err) }


def as_text(value) -> str | None:
  """Helper to return value as a string.

  Response token may have been co-erced to a non-string when serialised,
  so, if necessary, this reverses that to ensure it is a string.
  """
  if value is None or isinstance(value, str):
    return value
  return json.dumps(value)

def synchronous_llm_request(creds: dict[str, str], model: str, payload: dict[str, str]):
  """Send a request to cloudflare workers AI and wait for complete reponse (only used for images)."""
  urlTemplate = 'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{cf_model}'
  url = urlTemplate.format(account=creds['account'], cf_model=MODELS[model]['name'])
  hdrs = {'Authorization': 'Bearer '+ creds['token']}

  try:
    response = requests.post(url, headers=hdrs, data=json.dumps(payload))
    response.raise_for_status()
  except Exception as err:
    print('Request failed.\n', err)
    return ''

  result = response.json().get('result')
  if result is None:
    print('Unexpected response from Workers AI:\n', response.json())
    return ''

  answer = as_text(result.get('description'))
  if answer is None:
    print(f'Failed to parse {model} response:', result)
    answer = ''

  return answer

def streaming_llm_request(creds: dict[str, str], model: str, payload: dict, on_text, on_reasoning):
  """Send a request to cloudflare workers AI and receive server-sent event (SSE) stream with
  incremental response. Response deltas are passed to callbacks for rendering as they come in.
  """
  on_text = on_text or (lambda _: None)
  on_reasoning = on_reasoning or (lambda _: None)
  urlTemplate = 'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{cf_model}'
  url = urlTemplate.format(account=creds['account'], cf_model=MODELS[model]['name'])
  hdrs = { 'Authorization': 'Bearer ' + creds['token'], 'Accept': 'text/event-stream' }
  payload = { **payload, 'stream': True }

  schema = MODELS[model]['schema']
  answer = ''
  reasoning = ''
  finish_reason = None
  # tool call fragments, keyed by their index in the response, accumulated across chunks
  tool_frags: dict[int, dict] = {}
  complete = False

  try:
    with requests.post(url, headers=hdrs, data=json.dumps(payload), stream=True, timeout=(10, 300)) as response:
      response.raise_for_status()
      response.encoding = 'utf-8'
      for line in response.iter_lines(decode_unicode=True):
        # blank lines separate events, and ':' lines are keep-alive comments
        if not line or line.startswith(':') or not line.startswith('data:'):
          continue
        data = line[len('data:'):].strip()
        if data == '[DONE]':
          complete = True
          break
        try:
          chunk = json.loads(data)
        except json.JSONDecodeError:
          continue

        if schema == 'openai':
          choices = chunk.get('choices') or []
          if not choices:
            # final usage-only chunk
            continue
          choice = choices[0]
          delta = choice.get('delta') or {}

          text = as_text(delta.get('content')) or ''
          if text:
            answer += text
            on_text(text)

          # some models have 'reasoning', others have 'reasoning_content'
          thinking = as_text(delta.get('reasoning_content')) or as_text(delta.get('reasoning')) or ''
          if thinking:
            reasoning += thinking
            on_reasoning(thinking)

          for tc in delta.get('tool_calls') or []:
            index = tc.get('index', len(tool_frags))
            frag = tool_frags.setdefault(index, { 'id': None, 'name': None, 'arguments': '' })
            if tc.get('id'):
              frag['id'] = tc['id']
            function = tc.get('function') or {}
            if function.get('name'):
              frag['name'] = function['name']
            # arguments arrive as partial JSON string fragments
            frag['arguments'] += function.get('arguments') or ''

          if choice.get('finish_reason'):
            finish_reason = choice['finish_reason']

        elif schema == 'llama':
          # doesn't support tool calls or reasoning
          text = as_text(chunk.get('response')) or ''
          if text:
            answer += text
            on_text(text)

  except Exception as err:
    print(f'\n{RED}Request failed.\n{err}{COL_END}')
    return None

  if not complete:
    # the stream should always end with a [DONE] sentinel, so this means it was cut short
    print(f'\n{RED}Response stream ended unexpectedly.{COL_END}')

  tool_calls = [
    { 'id': frag['id'], 'type': 'function', 'function': { 'name': frag['name'], 'arguments': frag['arguments'] or '{}' } }
    for _, frag in sorted(tool_frags.items()) if frag['name']
  ]

  if schema == 'llama':
    # our system prompt asks to append special stop char, so if it's not there then response is (probably) truncated
    truncated = STOP_CHAR not in answer[-10:] and len(answer) > 800
    # strip the stop char from the answer
    answer = answer.rstrip().removesuffix(STOP_CHAR).rstrip()
  else:
    # openai models report truncation as either 'model_length' or 'length'
    truncated = finish_reason in ('length', 'model_length')

  return answer, reasoning, truncated, tool_calls

def prompt_llm(creds: dict[str, str], model: str, messages: list[dict[str, str]], on_text=None, on_reasoning=None):
  payload = { 
    'messages': messages
  }

  if MODELS[model]['schema'] == 'openai':
    # assuming all our 'openai' type models support tool calling
    payload['tools'] = get_tools()
    # set generous truncation limit (unlike llama, we don't expect early truncation)
    payload['max_tokens'] = 2048

  # all our chat models support streaming
  result = streaming_llm_request(creds, model, payload, on_text, on_reasoning)
  if result is None:
    return '','',False,[]
  return result

def get_summary(creds: dict[str, str], messages: list[dict[str, str]]):
  """Summarise the conversation history, using a smaller model."""
  system_message = { 'role': 'system', 'content': 'You are a concise chat bot' }
  user_message = { 'role': 'user', 'content': 'Condense the conversation history into a brief summary (100 words or less) that captures the essential information and context, focussing mainly on details provided by the user' }
  messages = [system_message] + messages + [user_message]
  answer, _, _, _ = prompt_llm(creds, 'llama-small', messages)
  return answer

# ---------- Markdown handling ------------------------------------------------

def split_table_cells(line: str) -> list[str]:
  """Split a markdown table row into its cells, respecting escaped pipes."""
  line = line.strip()
  cells = re.split(r'(?<!\\)\|', line)
  # the outer pipes produce empty cells at each end, which are not real columns
  if line.startswith('|'):
    cells = cells[1:]
  if line.endswith('|') and len(cells) > 1:
    cells = cells[:-1]
  return [cell.strip().replace('\\|', '|') for cell in cells]


def strip_emphasis(cell: str) -> str:
  """Remove any emphasis markers wrapping a whole cell.
  The first column of a table is often bold, so this needs to be stripped to avoid
  us adding double emphasis later.
  """
  cell = cell.strip()
  for marker in ('***', '**', '*', '__', '_'):
    while len(cell) > 2 * len(marker) and cell.startswith(marker) and cell.endswith(marker):
      cell = cell[len(marker):-len(marker)].strip()
  return cell


def is_table_alignment_row(line: str) -> bool:
  """Check for the |---|:--:| second line of a markdown table."""
  line = line.strip()
  return line.startswith('|') and '-' in line and re.fullmatch(r'[\s|:-]+', line) is not None


def transpose_table(table_lines: list[str], width: int) -> list[str]:
  """Rewrite a markdown table into a transpose form, if it is too wide for the terminal.
  Each row is rewritten into several lines containing that row's content. The first line 
  becomes a heading (the first cell in the row), followed by a labelled list of the remaining
  cells. Tables that already fit the terminal width are returned as is.
  """
  headers = [strip_emphasis(cell) for cell in split_table_cells(table_lines[0])]
  rows = [[strip_emphasis(cell) for cell in split_table_cells(line)] for line in table_lines[2:]]
  if len(headers) < 2 or not rows:
    return table_lines

  # ragged rows are padded so every row lines up with the headers
  rows = [row + [''] * (len(headers) - len(row)) for row in rows]

  # mdv indents by 2 and separates columns by 2, so that is the narrowest it could draw
  columns = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
  if sum(columns) + 2 * len(headers) <= width:
    return table_lines

  records = []
  for row in rows:
    records.append(f'**{row[0]}**' if row[0] else '**—**')
    records.append('')
    for label, cell in zip(headers[1:], row[1:]):
      if cell:
        records.append(f'- **{label}:** {cell}')
    records.append('')
  return records

def transform_wide_tables(text: str, width: int) -> str:
  """Transform any markdown table that is too wide for the terminal, so that mdv doesn't split the
  table into unreadable rows. Replaces each table row with a transposed equivalent over multiple lines.
  """
  lines = text.split('\n')
  output = []
  in_fence = False
  index = 0
  while index < len(lines):
    line = lines[index]
    if line.lstrip().startswith('```'):
      in_fence = not in_fence
    # a table is a header row followed by an alignment row, then its body
    elif not in_fence and line.strip().startswith('|') \
        and index + 1 < len(lines) and is_table_alignment_row(lines[index + 1]):
      end = index
      while end < len(lines) and lines[end].strip().startswith('|'):
        end += 1
      output.extend(transpose_table(lines[index:end], width))
      index = end
      continue
    output.append(line)
    index += 1
  return '\n'.join(output)

def render_markdown(text: str) -> str:
  """Convert markdown to ANSI."""
  width = shutil.get_terminal_size((80, 24)).columns
  text = transform_wide_tables(text, width)
  return mdv.main(text, cols=width, theme='963.4449') #theme='757.2295'


def split_markdown_blocks(text: str) -> tuple[list[str], str]:
  """Splits off the completed markdown blocks so they can be rendered.

  Returns list of complete blocks, along with the trailing remainder of text that is still in progress.
  A block end with a blank line, unless it falls inside a fenced code block.
  The last line is never treated as a boundary, as it may be mid-paragraph with more text yet to arrive.
  """
  lines = text.split('\n')
  blocks = []
  current = []
  in_fence = False
  consumed = 0
  for index, line in enumerate(lines):
    if line.lstrip().startswith('```'):
      in_fence = not in_fence
    current.append(line)
    # if not fenced code block and line is blank and not last line
    if not in_fence and line.strip() == '' and index < len(lines) - 1:
      # then this is end of block
      block = '\n'.join(current).strip()
      if block:
        blocks.append(block)
      current = []
      consumed = index + 1
  return blocks, '\n'.join(lines[consumed:])

# -----------------------------------------------------------------------------

class StreamPrinter:
  """Prints text deltas as they arrive. Each time a markdown block is completed its
  raw text is erased and replaced with the rendered version, so the response is
  formatted progressively as it streams in.
  """
  def __init__(self):
    # the raw text currently on screen, below the last rendered output
    self.live = ''
    # answer text that has not been rendered yet
    self.pending = ''
    # set once some text is received
    self.started = False
    # set once some text is rendered
    self.rendered = False
    # set if the live region grows too tall to erase. if set we will only show raw text
    self.raw_only = False

  def _write(self, ansi_text: str, plain_text: str):
    self.live += plain_text
    sys.stdout.write(ansi_text)
    sys.stdout.flush()

  def _erase_live(self) -> bool:
    # erase the raw text written since the last rendered output
    if not self.live:
      return True

    width, height = shutil.get_terminal_size((80, 24))
    # count screen lines, accounting for wrapping of long lines
    lines = sum(max(1, -(-len(line) // width)) for line in self.live.split('\n'))
    if lines >= height:
      # too tall to erase reliably, so leave the raw text in place from here on
      self.raw_only = True
      return False

    sys.stdout.write('\r')
    if lines > 1:
      sys.stdout.write(f'\033[{lines - 1}A')
    sys.stdout.write('\033[J')
    self.live = ''
    return True

  def _commit_live(self):
    # keep what is on screen, but stop treating it as erasable
    if self.live and not self.live.endswith('\n'):
      sys.stdout.write('\n')
    self.live = ''

  def _render(self, blocks: list[str], remainder: str):
    if not blocks or not self._erase_live():
      return
    for block in blocks:
      sys.stdout.write(render_markdown(block).rstrip('\n') + '\n')
    self.rendered = True
    self.pending = remainder
    # re-print the tail that is not yet a complete block
    self._write(remainder, remainder)

  def on_reasoning(self, delta: str):
    # reasoning is only shown while we wait for the answer, so stop once it starts
    if self.raw_only or self.started:
      return
    self._write(f'{DK_GREY}{delta}{COL_END}', delta)

  def on_text(self, delta: str):
    if not self.started:
      # keep any reasoning on screen, separated from the answer that follows it
      self.started = True
      if self.live:
        self._commit_live()
        sys.stdout.write('\n')
    self.pending += delta
    self._write(delta, delta)
    if not self.raw_only:
      self._render(*split_markdown_blocks(self.pending))

  def finish(self):
    # render whatever is left once the response is complete
    if self.raw_only:
      sys.stdout.write('\n')
    else:
      if self.pending.strip():
        self._render([self.pending.strip()], '')
      elif self.live:
        # nothing to render, so just keep whatever reasoning is on screen
        self._commit_live()
      elif not self.rendered:
        # nothing was rendered at all so leave stdout alone
        return
      # leave a gap before whatever will be printed next
      sys.stdout.write('\n')
    sys.stdout.flush()

# ----------------------------------------------------------------------------

class Chat:
  def __init__(self, initial_model: str):
    self.conversation = []
    self.creds = {}
    self.model = initial_model
    self.done = False

  def _system_message(self):
    main_clause = "You are a helpful assistant called Bob. Please answer questions briefly and professionally, without asking follow up questions. Format all responses using markdown. Don't use markdown tables with more than 4 columns."
    stop_clause = f" You must finish each answer with a '{STOP_CHAR}' stop character."
    prompt = main_clause + (stop_clause if (MODELS[self.model]['schema'] == 'llama') else '')
    return {
      'role': 'system',
      'content': prompt
    }

  def start(self, initial_prompt: str, one_shot: bool):
    self.creds = get_creds()
    self.conversation.append(self._system_message())
    prompt = initial_prompt

    # Long responses from llama will be truncated, so when we detect truncation we do a continuation prompt.
    # System prompt asks for a stop character so we can detect truncation, or for OpenAI etc. we check for it reaching max_tokens.
    # Either way, we limit the number of continuation requests in case the model fails to stop.
    truncated = False
    truncation_count = 0
    TRUNCATION_LIMIT = 5

    # tracks the conversation message index where tool_calls started, so tool calls can be pruned afterwards
    tool_call_start = None

    if len(prompt):
      self.conversation.append({ 'role': 'user', 'content': prompt })
      ask_prompt = False
    else:
      ask_prompt = True

    while not self.done:
      if ask_prompt:
        # get user prompt 
        session = PromptSession(
          ANSI(f"{GREEN}Ask: "), 
          multiline=False # use True to allow CR in input
        )
        prompt = session.prompt()
        if (prompt[:1] == '/'):
          self._dispatch_command(prompt[1:].lower())
          ask_prompt = True
          continue
          
        self.conversation.append({ 'role': 'user', 'content': prompt })

      printer = StreamPrinter()
      answer, reasoning, truncated, tool_calls = prompt_llm(self.creds, self.model, self.conversation, printer.on_text, printer.on_reasoning)
      printer.finish()
      
      if tool_calls:
        if tool_call_start is None:
          tool_call_start = len(self.conversation)
        self.conversation.append({ 'role': 'assistant', 'content': '', 'tool_calls': tool_calls })
        for tc in tool_calls:
          fn = tc['function']['name']
          args = json.loads(tc['function']['arguments'])
          print(f'{DK_GREY}Calling tool: {fn.replace('__', ' ')} {args}{COL_END}')
          tool_result = run_tool(fn, args) 
          self.conversation.append({ 'role': 'tool', 'tool_call_id': tc['id'], 'content': json.dumps(tool_result) })
        ask_prompt = False
        continue
      
      # if we were making tool_calls, then at this point the tool calling is over and answer should contain the final result 
      if tool_call_start is not None:
        # we can now prune the tool_calls and tool results to save tokens.
        self.conversation = self.conversation[:tool_call_start]
        tool_call_start = None
      
      if answer is None:
        answer = ''

      truncated = truncated and truncation_count < TRUNCATION_LIMIT

      if (one_shot and not truncated):
        exit()
        continue

      if truncated:
        print(f'...\n')
        if truncation_count > 0:
          # already done a continuation prompt, so strip the last user and assistant prompt
          self.conversation = self.conversation[:-2]

        assistant_message = f"{reasoning}\n\n{answer}".strip()
        self.conversation.append({ 'role': 'assistant', 'content': assistant_message })
        self.conversation.append({ 'role': 'user', 'content': 'Please continue.' })
        truncation_count += 1
        ask_prompt = False
      else:
        self.conversation.append({ 'role': 'assistant', 'content': answer })
        truncation_count = 0
        ask_prompt = True

      if len(self.conversation) > AUTO_COMPACT_THRESHOLD:
        # compact conversation to reduce token usage
        self.compact()

  def clear(self):
    self.conversation = []
    self.conversation.append(self._system_message())
    print(f'{BR_CYAN}Conversation cleared.{COL_END}')

  def compact(self):
    conv_len = len(self.conversation)
    # the minimum number of messages to keep in conversation after compacting 
    MIN_MESSAGES = 10
    # only compact if conversation is big enough (also allowing for system message and a couple of exchanges)
    if conv_len < MIN_MESSAGES + 5: 
      print(f'{BR_CYAN}Compaction not needed.{COL_END}')
      return
    # the index which we compact to
    compaction_index = conv_len - MIN_MESSAGES
    # keeping initial system prompt, summarise oldest messages
    summary = get_summary(self.creds, self.conversation[1:compaction_index])
    summary_message = { 'role': 'assistant', 'content': summary }
    self.conversation = [self._system_message()] + [summary_message] + self.conversation[compaction_index:]
    print(f'{BR_CYAN}Conversation compacted.{COL_END}')

  def set_model(self, cmd_args):
    chat_models = [key for key, value in MODELS.items() if value['schema'] != 'image_description']
    if len(cmd_args)==0:
      print(f'{BR_CYAN}Current model: {self.model}{COL_END}')
      print(f'{BR_CYAN}Available models: {', '.join(chat_models)}{COL_END}')
    elif cmd_args[0] not in chat_models:
      print(f'{BR_CYAN}Invalid model name.{COL_END}')
      print(f'{BR_CYAN}Available models: {', '.join(chat_models)}{COL_END}')
    else:
      self.model = cmd_args[0]
      if len(self.conversation) > 0:
        # system message can be model dependent, so update it
        self.conversation[0] = self._system_message()
      print(f'{BR_CYAN}Model changed to: {self.model}{COL_END}')

  def _quit(self):
    self.done = True
    print(f'{BR_CYAN}Bye!{COL_END}')

  def _dispatch_command(self, command: str):
    commands = {
      'clear': self.clear,
      'compact': self.compact,
      'model': self.set_model,
      'exit': self._quit,
    }

    cmd_parts = command.split(' ')
    handler = commands.get(cmd_parts[0])
    if handler:
      if len(inspect.signature(handler).parameters) == 0:
        handler()
      else:
        handler(cmd_parts[1:])
    else:
      print(f'{BR_CYAN}Available commands: {', '.join(commands.keys())}{COL_END}')
    print('')
      

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
  answer = synchronous_llm_request(creds, model, payload)
  if len(answer):
    print(render_markdown(answer))


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
    chat = Chat(args.model)
    chat.start(prompt, args.quick)
except (KeyboardInterrupt, EOFError):
  print(RED + '\nExiting...' + COL_END + '\n')
  sys.exit(0)


