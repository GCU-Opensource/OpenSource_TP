import slack_sdk

SLACK_TOKEN = 'your slack token'
SLACK_CHANNEL = '#your slack channel'

class SendSlackMessage:

    #argument : self, detected object
    def __init__(self, is_detected):
        self.is_detected = is_detected

    def send_message(self):
        #slack_token
        slack_token = SLACK_TOKEN
        
        #slack channel
        channel = SLACK_CHANNEL

        #slack_message
        if self.is_detected:
            slack_message = 'Smile Detected'
        else:
            slack_message = 'Smile Not Detected'

        #client
        client = slack_sdk.WebClient(token=slack_token)
        
        #send message
        client.chat_postMessage(channel=channel, text=slack_message)