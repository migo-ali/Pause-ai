# Pause-ai

This repository holds minimal examples for trying the Gemini REST API.

The `quickstarts/rest/Prompting_REST.ipynb` notebook demonstrates using `curl`.

A small command line interface is available in `gemini_interface.py`.
Set your API key in the `GOOGLE_API_KEY` environment variable and run:

```bash
python gemini_interface.py "Give me python code to sort a list."
```

## Project Dashboard

The repository also includes a minimal Flask dashboard for tracking projects that
interact with the OpenAI API.

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="YOUR_KEY_HERE"
   ```

3. Run the dashboard:

   ```bash
   python dashboard/app.py
   ```

Then open [http://localhost:5000](http://localhost:5000) to manage projects and
trigger updates via the OpenAI API.


