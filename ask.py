#!/usr/bin/env python3

import os
import sys
import requests
import json
import argparse
from prompt_toolkit import PromptSession, ANSI
import mdv

GREEN   = '\033[92m'
YELLOW  = '\033[33m'
CYAN    = '\033[36m'
BR_CYAN = '\033[96m'
RED     = '\033[31m'
COL_END = '\033[0m'

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


def send_ai_request(creds: dict[str, str], messages: list[dict[str, str]]):
  # model = '@cf/meta/llama-3.2-3b-instruct'
  # model = '@cf/meta/llama-3.2-3b-instruct'
  # model = '@cf/meta/llama-3.1-8b-instruct'
  model = '@cf/meta/llama-4-scout-17b-16e-instruct'

  payload = { 
    'messages': messages
  }

  urlTemplate = 'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}'
  url = urlTemplate.format(account=creds['account'], model=model)
  hdrs = {'Authorization': 'Bearer '+ creds['token']}

  try:
    response = requests.post(url, headers=hdrs, data=json.dumps(payload))
    response.raise_for_status()
  except Exception as err:
    print('Request failed.\n', err)
    return ''
  else:
    answer = response.json().get('result',{}).get('response')
    if answer is not None:
      return answer
    print('Received unexpected response from Workers AI\n', response.json())
    return ''

def run_chat(prompt: str):
  creds = get_creds()
  systemMessage = { 'role': 'system', 'content': "You are a helpful assistant called Bob. Please answer questions briefly and professionally, without asking follow up questions. Format all responses using markdown. You must finish each answer with the '⏎' character. If the user prompt is '>' then continue where your previous response was truncated." }
  conversation = []
  conversation.append(systemMessage)

  # if prompt provided just do a single prompt, otherwise do conversation mode
  oneShot = prompt != ''
  truncated = False

  while True:
    if truncated:
      # ask for next part of truncated response
      prompt = '>'
    elif not oneShot:
      session = PromptSession(
        ANSI(f"{GREEN}Ask: "), 
        multiline=False # enable to allow CR in input
      )
      prompt = session.prompt()
      concattedAnswer = ''

      
    if (prompt.lower() == 'clear'):
      conversation = []
      conversation.append(systemMessage)
      print(BR_CYAN + 'Conversation cleared.' + COL_END + '\n')

    else:
      conversation.append({ 'role': 'user', 'content': prompt })
      answer = send_ai_request(creds, conversation)
      conversation.append({ 'role': 'assistant', 'content': answer })

      # our system prompt asks to append special char when complete, so if it's not there then response is (probably) truncated 
      truncated = '⏎' not in answer[-10:] and len(answer) > 800

      # convert markdown to ANSI
      ansiText = mdv.main(answer, theme='963.4449') #theme='757.2295'
      print(ansiText)

    if prompt.lower() == 'exit' or prompt.lower() == 'bye' or (oneShot and not truncated):
      break


try:
  parser = argparse.ArgumentParser()
  parser.add_argument('text', type=str, nargs='*', default=[])
  args = parser.parse_args()
  # take any command line text
  prompt = ' '.join(args.text)
  if not sys.stdin.isatty():
    # append any piped stdin
    prompt = prompt + ' ' + sys.stdin.read()
  run_chat(prompt)
except (KeyboardInterrupt, EOFError):
  print(RED + '\nExiting...' + COL_END + '\n')
  sys.exit(0)
