# ==============================================================================
#  Tool World: Advanced Prototype (Gemini Version)
# ==============================================================================
#
#  This script demonstrates an advanced framework for Large Language Model (LLM)
#  tool invocation using latent space representations.
#
#  Key Upgrades:
#  1.  **LLM Integration**: Now uses Google's Gemini 1.5 Flash for intelligent
#      argument extraction.
#  2.  **Structured Arguments**: Tools define their arguments using a JSON-like
#      schema, enabling robust and predictable input handling.
#  3.  **Richer Embeddings**: Tool embeddings are generated from a detailed
#      combination of name, description, and argument structure for more
#      accurate semantic matching.
#  4.  **Secure API Key Management**: Integrates with Google Colab's secrets
#      manager to securely access the GOOGLE_API_KEY.
#  5.  **Comprehensive Interface**: A Gradio app visualizes the entire process.
#
#  To Run in Google Colab:
#  - Add your Google AI Studio API key as a secret named 'GOOGLE_API_KEY'.
#  - Copy and paste this entire script into a single cell.
#  - Run the cell.
#
# ==============================================================================

# ------------------------------
#  1. INSTALL & IMPORT PACKAGES
# ------------------------------
# !pip install -q sentence-transformers umap-learn gradio google-generativeai

import numpy as np
import umap
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
import google.generativeai as genai

# --- Configuration for LLM-based Argument Extraction ---
# This will attempt to use the 'GOOGLE_API_KEY' secret in Google Colab.
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    USE_GEMINI_LLM = True
    print("✅ Successfully configured Gemini API.")
except (ImportError, userdata.SecretNotFoundError):
    USE_GEMINI_LLM = False
    print("⚠️ WARNING: Could not find 'GOOGLE_API_KEY' in Colab secrets.")
    print("   Using a mock function for argument extraction.")
    print("   To use the full power of this prototype, please add your Google API key as a secret in Colab.")
except Exception as e:
    USE_GEMINI_LLM = False
    print(f"An error occurred during Gemini configuration: {e}")


# ------------------------------
#  2. LOAD EMBEDDING MODEL
# ------------------------------
print("⚙️ Loading embedding model...")
# Using a powerful model for better semantic understanding
embedder = SentenceTransformer('all-mpnet-base-v2')
print("✅ Embedding model loaded.")


# ------------------------------
#  3. ADVANCED TOOL DEFINITION
# ------------------------------

class Tool:
    """
    Represents a tool with structured arguments and rich descriptive data
    for high-quality embedding.
    """
    def __init__(self, name, description, args_schema, function, examples=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.function = function
        self.examples = examples or []
        self.embedding = self._create_embedding()

    def _create_embedding(self):
        """
        Creates a rich embedding by combining the tool's name, description,
        argument structure, and examples.
        """
        schema_str = json.dumps(self.args_schema, indent=2)
        examples_str = "\n".join([f" - Example: {ex['prompt']} -> Args: {json.dumps(ex['args'])}" for ex in self.examples])
        
        embedding_text = (
            f"Tool Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Argument Schema: {schema_str}\n"
            f"Usage Examples:\n{examples_str}"
        )
        return embedder.encode(embedding_text, convert_to_tensor=True)

    def __repr__(self):
        return f"<Tool: {self.name}>"

# ------------------------------
#  4. TOOL IMPLEMENTATIONS
# ------------------------------
# These are the actual Python functions that get executed by the tools.

def get_weather_forecast(location: str, days: int = 1):
    """Simulates fetching a weather forecast."""
    if not isinstance(location, str) or not isinstance(days, int):
        return {"error": "Invalid argument types. 'location' must be a string and 'days' an integer."}
    
    weather_conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Snowy"]
    response = {"location": location, "forecast": []}
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        condition = np.random.choice(weather_conditions)
        temp = np.random.randint(5, 25)
        response["forecast"].append({
            "date": date,
            "condition": condition,
            "temperature_celsius": temp
        })
    return response

def create_calendar_event(title: str, date: str, duration_minutes: int = 60, participants: list = None):
    """Simulates creating a calendar event."""
    try:
        event_time = datetime.strptime(date, '%Y-%m-%d %H:%M')
        return {
            "status": "success",
            "event_created": {
                "title": title,
                "start_time": event_time.isoformat(),
                "end_time": (event_time + timedelta(minutes=duration_minutes)).isoformat(),
                "participants": participants or ["organizer"]
            }
        }
    except ValueError:
        return {"error": "Invalid date format. Please use 'YYYY-MM-DD HH:MM'."}

def summarize_text(text: str, compression_level: str = 'medium'):
    """Summarizes a given text based on a compression level."""
    word_count = len(text.split())
    ratios = {'high': 0.2, 'medium': 0.4, 'low': 0.7}
    ratio = ratios.get(compression_level, 0.4)
    summary_length = int(word_count * ratio)
    summary = " ".join(text.split()[:summary_length])
    return {"summary": summary + "...", "original_word_count": word_count, "summary_word_count": summary_length}

def search_web(query: str, domain: str = None):
    """Simulates a web search, with an optional domain filter."""
    results = [
        f"Simulated result 1 for '{query}'",
        f"Simulated result 2 for '{query}'",
        f"Simulated result 3 for '{query}'"
    ]
    if domain:
        return {"status": f"Searching for '{query}' within '{domain}'...", "results": results}
    return {"status": f"Searching for '{query}'...", "results": results}


# ------------------------------
#  5. DEFINE THE TOOLSET
# ------------------------------

tools = [
    Tool(
        name="weather_reporter",
        description="Provides the weather forecast for a specific location for a given number of days.",
        args_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g., 'San Francisco, CA'"},
                "days": {"type": "integer", "description": "The number of days to forecast", "default": 1}
            },
            "required": ["location"]
        },
        function=get_weather_forecast,
        examples=[
            {"prompt": "what's the weather like in London for the next 3 days", "args": {"location": "London", "days": 3}},
            {"prompt": "forecast for New York tomorrow", "args": {"location": "New York", "days": 1}}
        ]
    ),
    Tool(
        name="calendar_creator",
        description="Creates a new event in the user's calendar.",
        args_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the calendar event"},
                "date": {"type": "string", "description": "The start date and time in 'YYYY-MM-DD HH:MM' format"},
                "duration_minutes": {"type": "integer", "description": "The duration of the event in minutes", "default": 60},
                "participants": {"type": "array", "items": {"type": "string"}, "description": "List of email addresses of participants"}
            },
            "required": ["title", "date"]
        },
        function=create_calendar_event,
        examples=[
            {"prompt": "Schedule a 'Project Sync' for tomorrow at 3pm with bob@example.com", "args": {"title": "Project Sync", "date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d 15:00'), "participants": ["bob@example.com"]}},
            {"prompt": "new event: Dentist appointment on 2025-12-20 at 10:00 for 45 mins", "args": {"title": "Dentist appointment", "date": "2025-12-20 10:00", "duration_minutes": 45}}
        ]
    ),
    Tool(
        name="text_summarizer",
        description="Summarizes a long piece of text. Can be set to high, medium, or low compression.",
        args_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to be summarized."},
                "compression_level": {"type": "string", "enum": ["high", "medium", "low"], "description": "The level of summarization.", "default": "medium"}
            },
            "required": ["text"]
        },
        function=summarize_text,
        examples=[
            {"prompt": "summarize this article for me, make it very short: [long text...]", "args": {"text": "[long text...]", "compression_level": "high"}}
        ]
    ),
    Tool(
        name="web_search",
        description="Performs a web search to find information on a topic.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "domain": {"type": "string", "description": "Optional: a specific website domain to search within (e.g., 'wikipedia.org')."}
            },
            "required": ["query"]
        },
        function=search_web,
        examples=[
            {"prompt": "who invented the light bulb", "args": {"query": "who invented the light bulb"}},
            {"prompt": "search for 'transformer models' on arxiv.org", "args": {"query": "transformer models", "domain": "arxiv.org"}}
        ]
    )
]

print(f"✅ {len(tools)} tools defined and embedded.")

# ------------------------------
#  6. CORE LOGIC: TOOL SELECTION & ARGUMENT EXTRACTION
# ------------------------------

def find_best_tool(user_intent: str):
    """Finds the most semantically similar tool for a user's intent."""
    intent_embedding = embedder.encode(user_intent, convert_to_tensor=True)
    similarities = [util.pytorch_cos_sim(intent_embedding, tool.embedding).item() for tool in tools]
    best_index = int(np.argmax(similarities))
    best_tool = tools[best_index]
    best_score = similarities[best_index]
    return best_tool, best_score, similarities

def extract_arguments_gemini(user_prompt: str, tool: Tool):
    """
    Uses Google's Gemini model to extract structured arguments from a user prompt.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    system_prompt = f"""
    You are an expert at extracting structured data from natural language.
    Your task is to analyze the user's prompt and extract the arguments required
    to call the tool: '{tool.name}'.

    You must adhere to the following JSON schema for the arguments:
    {json.dumps(tool.args_schema, indent=2)}

    - If a value is not present in the prompt for a non-required field, omit it from the JSON.
    - If a required value is missing, return a JSON object with an "error" key explaining what is missing.
    - Today's date is {datetime.now().strftime('%Y-%m-%d')}.
    - Respond ONLY with a JSON object inside a ```json ... ``` code block. Do not include any other text or explanation.
    """

    try:
        response = model.generate_content([system_prompt, user_prompt])
        text_response = response.text
        
        # Clean the response to extract the JSON part from a markdown code block
        json_start = text_response.find('```json')
        json_end = text_response.rfind('```')
        
        if json_start != -1 and json_end != -1:
            json_str = text_response[json_start + 7 : json_end].strip()
            return json.loads(json_str)
        else:
            # Fallback if the model doesn't use a code block
            return json.loads(text_response)

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": f"Failed to communicate with the Gemini LLM. Details: {str(e)}"}

def extract_arguments_mock(user_prompt: str, tool: Tool):
    """A mock function for argument extraction. Used if no API key is provided."""
    args = {}
    if tool.name == "weather_reporter":
        words = user_prompt.split()
        for word in words:
            if word.isdigit():
                args['days'] = int(word)
        if 'in ' in user_prompt:
            args['location'] = user_prompt.split('in ')[-1].split(' for ')[0]
        elif 'for ' in user_prompt:
            args['location'] = user_prompt.split('for ')[-1].split(' for ')[0]
        else:
            args['location'] = "unknown"
    elif tool.name == "calendar_creator":
        args = {"title": "Example Event (from mock)", "date": "2025-07-10 15:00"}
    else:
        args = {key: "mock_value" for key in tool.args_schema.get('required', [])}
    
    for req_arg in tool.args_schema.get("required", []):
        if req_arg not in args:
            args[req_arg] = f"MISSING_{req_arg.upper()}"
            
    return args

def execute_tool(user_prompt: str):
    """The main pipeline: Find tool, extract args, execute."""
    selected_tool, score, _ = find_best_tool(user_prompt)
    
    print(f"⚙️ Selected Tool: {selected_tool.name}. Now extracting arguments...")
    if USE_GEMINI_LLM:
        extracted_args = extract_arguments_gemini(user_prompt, selected_tool)
    else:
        extracted_args = extract_arguments_mock(user_prompt, selected_tool)

    if 'error' in extracted_args:
        print(f"❌ Argument extraction failed: {extracted_args['error']}")
        return (
            user_prompt,
            selected_tool.name,
            f"{score:.3f}",
            json.dumps(extracted_args, indent=2),
            "Execution failed during argument extraction."
        )

    print(f"✅ Arguments extracted: {json.dumps(extracted_args, indent=2)}")

    try:
        print(f"🚀 Executing tool function: {selected_tool.name}...")
        output = selected_tool.function(**extracted_args)
        print(f"✅ Execution complete.")
        output_str = json.dumps(output, indent=2)
    except Exception as e:
        print(f"❌ Tool execution failed: {e}")
        output_str = f'{{"error": "Tool execution failed", "details": "{str(e)}"}}'

    return (
        user_prompt,
        selected_tool.name,
        f"{score:.3f}",
        json.dumps(extracted_args, indent=2),
        output_str
    )


# ------------------------------
#  7. VISUALIZATION
# ------------------------------

def plot_tool_world(user_intent=None):
    """Generates a 2D UMAP plot of the tool latent space."""
    tool_vectors = [tool.embedding.cpu().numpy() for tool in tools]
    labels = [tool.name for tool in tools]
    all_vectors = tool_vectors
    
    if user_intent:
        intent_vector = embedder.encode(user_intent, convert_to_tensor=True).cpu().numpy()
        all_vectors.append(intent_vector)
        labels.append("Your Intent")

    n_neighbors = min(len(all_vectors) - 1, 5)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, metric='cosine', random_state=42)
    reduced_vectors = reducer.fit_transform(all_vectors)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, label in enumerate(labels):
        x, y = reduced_vectors[i]
        if label == "Your Intent":
            ax.scatter(x, y, color='red', s=150, zorder=5, label=label, marker='*')
            ax.text(x, y + 0.05, label, fontsize=12, ha='center', color='red', weight='bold')
        else:
            ax.scatter(x, y, s=100, alpha=0.8, label=label)
            ax.text(x, y + 0.05, label, fontsize=10, ha='center')

    ax.set_title("Tool World: Latent Space Map", fontsize=16)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.grid(True)
    
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    return fig


# ------------------------------
#  8. GRADIO INTERFACE
# ------------------------------

print("🚀 Launching Gradio interface...")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ Tool World: Advanced Prototype (Gemini Version)")
    gr.Markdown(
        "Enter a natural language command. The system will select the best tool, "
        "extract structured arguments with **Gemini 1.5 Flash**, and execute it."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="Your Intent",
                placeholder="e.g., What's the weather in Paris for 2 days?",
                lines=3
            )
            run_btn = gr.Button("Invoke Tool", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    "Schedule a 'Team Meeting' for tomorrow at 10:30 am",
                    "What is the weather forecast in Tokyo for the next 5 days?",
                    "search for the latest news on generative AI on reuters.com",
                    "Please give me a very short summary of this text: The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States, in the period from about 1760 to sometime between 1820 and 1840."
                ],
                inputs=inp
            )

        with gr.Column(scale=2):
            gr.Markdown("### Invocation Details")
            with gr.Row():
                out_tool = gr.Textbox(label="Selected Tool", interactive=False)
                out_score = gr.Textbox(label="Similarity Score", interactive=False)
            
            # FIX: Removed the 'interactive' argument from gr.JSON
            out_args = gr.JSON(label="Extracted Arguments")
            out_result = gr.JSON(label="Tool Execution Output")

    with gr.Row():
        gr.Markdown("---")
        gr.Markdown("### Latent Space Visualization")
        plot_output = gr.Plot(label="Tool World Map")

    def process_and_plot(user_prompt):
        if not user_prompt:
             return "", "", None, None, plot_tool_world()
        
        prompt, tool_name, score, args_json, result_json = execute_tool(user_prompt)
        fig = plot_tool_world(user_prompt)
        
        # Safely load JSON strings into objects for the UI
        try:
            args_obj = json.loads(args_json)
        except (json.JSONDecodeError, TypeError):
            args_obj = {"error": "Invalid JSON in arguments"}

        try:
            result_obj = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            result_obj = {"error": "Invalid JSON in result"}

        return tool_name, score, args_obj, result_obj, fig

    run_btn.click(
        fn=process_and_plot,
        inputs=inp,
        outputs=[out_tool, out_score, out_args, out_result, plot_output]
    )
    
    demo.load(fn=plot_tool_world, inputs=None, outputs=plot_output)

demo.launch(debug=True)
