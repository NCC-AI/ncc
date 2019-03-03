import json
import requests

from keras.callbacks import LambdaCallback


def slack_logging(url):

    slack_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: requests.post(
            url=url,
            data=json.dumps(
                dict(
                    channel='#deep_callbacks',
                    attachments=[
                        dict(
                            fields=[
                                dict(value='epoch: %d' % epoch + json.dumps(dict(logs)))
                            ]
                        )
                    ]
                )
            )
        )
    )
    return slack_logging_callback
