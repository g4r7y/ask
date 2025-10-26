#!/usr/bin/env python3

import os
import sys
import requests
import json

GREEN = "\033[92m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
COL_END = "\033[0m"

def intialiseCreds():
  cloudFlareAccount = os.environ.get('CLOUDFLARE_ACCOUNT')
  if cloudFlareAccount == None:
    print('You need to set CLOUDFLARE_ACCOUNT env var with your account ID.')
    exit(1)
  
  apiToken = os.environ.get('CLOUDFLARE_API_TOKEN')
  if apiToken == None:
    print('You need to set CLOUDFLARE_API_TOKEN with your CloudFlare Workers AI API token.')
    exit(1)

  creds = { "account": cloudFlareAccount, "token": apiToken }
  return creds


def sendAiRequest(creds, messages):
  # model = '@cf/meta/llama-3.2-1b-instruct'
  model = '@cf/meta/llama-3.2-3b-instruct'
  # model = '@cf/meta/llama-3.1-8b-instruct'

  payload = { 
    'messages': messages
  }

  urlTemplate = 'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}'
  url = urlTemplate.format(account=creds["account"], model=model)
  hdrs = {'Authorization': 'Bearer '+ creds["token"]}

  try:
    response = requests.post(url, headers=hdrs, data=json.dumps(payload))
    response.raise_for_status()
  except requests.exceptions.HTTPError as err:
    print("Request failed.\n", err)
    return ""
  else:
    # todo - validate response 
    answer = response.json()['result']['response']
    return answer

def runChat():
  creds = intialiseCreds()
  systemMessage = { "role": "system", "content": "You are a helpful assistant called Bob. Please answer questions briefly. If you are not sure of the answer, just say you don't know. If the user prompt is 'more' or '>' then continue where your previous response was truncated." }
  conversation = []
  conversation.append(systemMessage)

  prompt = ""
  while (prompt.lower() != "exit" and prompt.lower() != "bye"):
    prompt = input(GREEN + "Ask: " + YELLOW)
    if (prompt.lower() == "clear"):
      conversation = []
      conversation.append(systemMessage)
      print(CYAN + "Conversation cleared." + COL_END + "\n")
    else:
      conversation.append({ "role": "user", "content": prompt })
      answer = sendAiRequest(creds, conversation)
      print(CYAN + answer + COL_END + "\n")
      conversation.append({ "role": "assistant", "content": answer })


try:
  runChat()
except (KeyboardInterrupt, EOFError):
  print(RED + "\nExiting..." + COL_END + "\n")
  sys.exit(0)
