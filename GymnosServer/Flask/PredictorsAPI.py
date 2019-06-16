from flask import Flask

app = Flask(__name__)


@app.route('/hello', methods=['GET'])
def helloIndex():
    return 'Hello World from Python Flask!'

@app.route('/whothis', methods=['GET'])
def whoIndex():
    return 'YOOOO'

app.debug = True
app.run(host='0.0.0.0', port=5000)