#!/usr/bin/env python3

import os
import sys
import requests
import json
import argparse
from prompt_toolkit import PromptSession, ANSI
import mdv

GREEN   = '\033[92m'
CYAN    = '\033[36m'
BR_CYAN = '\033[96m'
RED     = '\033[31m'
COL_END = '\033[0m'

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
  if MODELS[model]['schema'] == 'openai':
    payload['max_tokens'] = 512 + 200 * token_multiplier

  result = post_llm_request(creds, model, payload)
  if result is None:
    return '','',False
  return parse_result(model, result)


def parse_result(model: str, result):
  truncated = False
  reasoning = ''

  if MODELS[model]['schema'] == 'openai':
    choice = result.get('choices', [None])[0]
    answer = choice.get('message',{}).get('content')
    reasoning = choice.get('message',{}).get('reasoning_content')
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

  if answer is None:
    print(f'Failed to parse {model} response:', result)
    answer = ''

  return answer, reasoning, truncated

def get_summary(creds: dict[str, str], messages: list[dict[str, str]]):
  # summarise the conversation history, using a smaller model
  system_message = { 'role': 'system', 'content': 'You are a concise chat bot' }
  user_message = { 'role': 'user', 'content': 'Condense the conversation history into a brief summary (100 words or less) that captures the essential information and context, focussing mainly on details provided by the user' }
  messages = [system_message] + messages + [user_message]
  answer, _, _ = prompt_llm(creds, 'llama-small', messages, 1)
  return answer

def print_markdown(text: str):
  # convert markdown to ANSI
  if len(text):
    ansiText = mdv.main(text, theme='963.4449') #theme='757.2295'
    print(ansiText)


def chat(prompt: str, model: str, one_shot: bool):
  creds = get_creds()
  system_message = { 'role': 'system', 'content': "You are a helpful assistant called Bob. Please answer questions briefly and professionally, without asking follow up questions. Format all responses using markdown. You must finish each answer with a '⏎' stop character." }
  conversation = []
  conversation.append(system_message)

  # Long responses will be truncated, so when we detect truncation we do a continuation prompt.
  # System prompt asks for a stop character so we can detect truncation, or for OpenAI etc. we check for it reaching max_tokens.
  # Either way, we limit the number of continuation requests in case the model fails to stop.
  truncated = False
  truncation_count = 0
  TRUNCATION_LIMIT = 5

  while True:
    if len(prompt)==0:
      # get user prompt 
      session = PromptSession(
        ANSI(f"{GREEN}Ask: "), 
        multiline=False # use True to allow CR in input
      )
      prompt = session.prompt()
      
      if (prompt.lower() == 'clear'):
        conversation = []
        conversation.append(system_message)
        prompt = ''
        print(BR_CYAN + 'Conversation cleared.' + COL_END + '\n')
        continue

    conversation.append({ 'role': 'user', 'content': prompt })
    token_multiplier = truncation_count if truncated else 1
    answer, reasoning, truncated = prompt_llm(creds, model, conversation, token_multiplier)
    truncated = truncated and truncation_count < TRUNCATION_LIMIT

    print_markdown(answer)

    if prompt.lower() == 'exit' or prompt.lower() == 'bye' or (one_shot and not truncated):
      break

    if truncated:
      print(f'...\n')
      if truncation_count > 0:
        # already done a continuation prompt, so strip the last user and assistant prompt
        conversation = conversation[:-2]

      assistant_message = f"{reasoning}\n\n{answer}".strip()
      conversation.append({ 'role': 'assistant', 'content': assistant_message })
      prompt = 'Please continue.'
      truncation_count += 1
    else:
      conversation.append({ 'role': 'assistant', 'content': answer })
      truncation_count = 0
      prompt = ''

    # to keep tokens down, summarise oldest messages in the conversation with a summary of those messages
    if len(conversation) > 20:
      summary = get_summary(creds, conversation[1:11])   # ignoring initial system prompt, summarise oldest 10 messages (i.e. 5 exchanges)
      summary_message = { 'role': 'assistant', 'content': summary }
      conversation = [system_message] + [summary_message] + conversation[11:]


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
  answer, _, _ = parse_result(model, result)
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
    chat(prompt, args.model, args.quick)
except (KeyboardInterrupt, EOFError):
  print(RED + '\nExiting...' + COL_END + '\n')
  sys.exit(0)


