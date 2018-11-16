import json
import requests

from keras.callbacks import LambdaCallback


def slack_logging():
    url = 'https://hooks.slack.com/services/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'

    slack_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: requests.post(
            url=url,
            data=json.dumps(
                dict(
                    channel='#deep_callbacks',
                    attachments=[
                        dict(
                            fields=[
                                dict(value=json.dumps(dict(logs)))
                            ]
                        )
                    ]
                )
            )
        )
    )
    return slack_logging_callback
