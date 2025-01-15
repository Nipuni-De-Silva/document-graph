from flask import Flask, request, jsonify
from src.handlers.knowledge_graph_handler import KnowledgeGraphHandler
import tempfile
import os

app = Flask(__name__)

# Initialize the KnowledgeGraphHandler
handler = KnowledgeGraphHandler()


@app.route('/process-document', methods=['POST'])
def process_document():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Check if the file has a valid name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Initialize the KnowledgeGraphHandler
    handler = KnowledgeGraphHandler()

    try:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        # Step 1: Process the document and create the knowledge graph
        handler.process_and_save_document(temp_file_path)
        
        # Remove the temporary file after processing
        os.remove(temp_file_path)

        return jsonify({"message": f"Knowledge graph successfully created for the document."}), 200

    except Exception as e:
        return jsonify({"error": f"Error while creating the knowledge graph: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)