# First, make your own Slack app and type in the token
# Select the channel you want to send the message to

import slack_sdk

SLACK_TOKEN = 'your slack bot token'
SLACK_CHANNEL = '#your slack channel'

class SendSlackMessage:
    # argument : smile detected object
    def send_message(list):
        # slack_token
        slack_token = SLACK_TOKEN
        
        # slack channel
        channel = SLACK_CHANNEL

        # slack_message
        if len(list) > 0:
            slack_message = 'Smile Detected'
        else:
            slack_message = 'Smile Not Detected'

        # client
        client = slack_sdk.WebClient(token=slack_token)
        
        # send message
        client.chat_postMessage(channel=channel, text=slack_message)