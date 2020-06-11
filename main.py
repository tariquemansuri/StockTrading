from flask import Flask

from Stock_Trading_Bot import mainFunc

try:
    import googleclouddebugger

    googleclouddebugger.enable()
except ImportError:
    pass

import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


@app.route('/')
def func():
    mainFunc()


if __name__ == '__main__':
    app.run(host='127.0.0.1', prot=8080, debug=True)
