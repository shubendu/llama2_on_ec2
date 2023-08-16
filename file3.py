from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()  # Get JSON data from the request
    if 'text' in data:
        text = data['text']  # Extract the 'text' field from the JSON
        return jsonify({'result': text})  # Return the text in JSON format
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
