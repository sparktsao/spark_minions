import streamlit as st
from minions.minion import Minion
from minions.minions_ex import Minions
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.together import TogetherClient
import os
import time
import pandas as pd
from openai import OpenAI
import fitz  # PyMuPDF
from PIL import Image
import io
from pydantic import BaseModel
import json
from streamlit_theme import st_theme


# Set custom sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 350px;
            max-width: 750px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# OpenAI model pricing per 1M tokens
OPENAI_PRICES = {
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60
    },
    "o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40
    }
}

PROVIDER_TO_ENV_VAR_KEY = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Together": "TOGETHER_API_KEY",
}

# for Minions protocol
class JobOutput(BaseModel):
    answer: str | None
    explanation: str | None
    citation: str | None

import pymupdf
def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        #doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        doc  = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def extract_text_from_image(image_bytes):
    """Extract text from an image file using pytesseract OCR."""
    try:
        import pytesseract
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def jobs_callback(jobs):
    """Display a list of jobs with toggleable details."""
    total_jobs = len(jobs)
    successful_jobs = sum(1 for job in jobs if job.include)
    st.write(f"### Jobs ({successful_jobs}/{total_jobs} successful)")
    for job_idx, job in enumerate(jobs):
        icon = "✅" if job.include else "❌"
        with st.expander(f"{icon} Job {job_idx + 1} (Task: {job.manifest.task_id}, Chunk: {job.manifest.chunk_id})"):
            st.write("**Task:**")
            st.write(job.manifest.task)
            st.write("**Chunk Preview:**")
            chunk_preview = job.manifest.chunk[:100] + "..." if len(job.manifest.chunk) > 100 else job.manifest.chunk
            st.write(chunk_preview)
            if job.output.answer:
                st.write("**Answer:**")
                st.write(job.output.answer)
            if job.output.explanation:
                st.write("**Explanation:**")
                st.write(job.output.explanation)
            if job.output.citation:
                st.write("**Citation:**")
                st.write(job.output.citation)


placeholder_messages = {}
THINKING_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2xhc3QzaHZyYWJ0M3czZXVjMGQ0YW50ZTBvcDdlNXVxNWhvZHdhOCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7bu3XilJ5BOiSGic/giphy.gif"
GRU_GIF = "https://media.giphy.com/media/ySMINwPzf50IM/giphy.gif?cid=790b7611vozglgf917p8ou0vjzydpgk9p8hpdwq9x95euttp&ep=v1_gifs_search&rid=giphy.gif&ct=g"
MINION_GIF = "https://media.giphy.com/media/1MTLxzwvOnvmE/giphy.gif?cid=ecf05e47aulvh0ffnr7p49bk2c8cvtopnincdtg8wsz9hlx4&ep=v1_gifs_search&rid=giphy.gif&ct=g"
MINION_VIDEO = "https://www.youtube.com/embed/JgdRgjgCrOc?si=6AcobGeBSHrtIg3w&autoplay=1&mute=1"

def is_dark_mode():
    theme = st_theme()
    if theme and "base" in theme:
        if theme['base'] == "dark":
            return True
    return False

# Check theme setting
dark_mode = is_dark_mode()

# Choose image based on theme
if dark_mode:
    image_path = "assets/minions_logo_no_background.png"  # Replace with your dark mode image
else:
    image_path = "assets/minions_logo_light.png"  # Replace with your light mode image


# Display Minions logo at the top
st.image(image_path, use_container_width=True)

# add a horizontal line that is width of image
st.markdown("<hr style='width: 100%;'>", unsafe_allow_html=True)



def message_callback(role, message, is_final=True):
    """Show messages for both Minion and Minions protocols, 
       labeling the local vs remote model clearly."""
    # Map supervisor -> Remote, worker -> Local
    chat_role = "Remote" if role == "supervisor" else "Local"

    if role == "General":
        chat_role = "Remote"
        path = "assets/llama3.1.png"
        #path = GRU_GIF
    else:
        chat_role = "Primus"
        path = "assets/minion_resized.png"
        #path = MINION_GIF

   
    # If we are not final, render a placeholder.
    if not is_final:
        # Create a placeholder container and store it for later update.
        placeholder = st.empty()
        with placeholder.chat_message(chat_role, avatar=path):
            st.markdown("**Working...**")
            if role == "supervisor":
                #st.image(GRU_GIF, width=50)
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center;">
                        <img src="{GRU_GIF}" width="200">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                #st.image(MINION_GIF, width=50)
                video_html = f"""
                    <style>
                    .video-container {{
                        position: relative;
                        padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
                        height: 0;
                        overflow: hidden;
                        max-width: 100%;
                        background: #000;
                    }}
                    .video-container iframe {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                    }}
                    </style>

                    <div class="video-container">
                        <iframe src="{MINION_VIDEO}" 
                        frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
                    </div>
                """

                st.markdown(video_html, unsafe_allow_html=True)
            #st.image(THINKING_GIF, width=50)
        placeholder_messages[role] = placeholder
    else:
        if role in placeholder_messages:
            placeholder_messages[role].empty()
            del placeholder_messages[role]
        with st.chat_message(chat_role, avatar=path):
            if role == "worker" and isinstance(message, list):
                # For Minions protocol, messages are a list of jobs
                st.markdown("#### Here are the outputs from all the minions!")
                tasks = {}
                for job in message:
                    task_id = job.manifest.task_id
                    if task_id not in tasks:
                        tasks[task_id] = {
                            "task": job.manifest.task,
                            "jobs": []
                        }
                    tasks[task_id]["jobs"].append(job)


                for task_id, task_info in tasks.items():
                    # first srt task_info[jobs] by job_id
                    task_info["jobs"] = sorted(task_info["jobs"], key=lambda x: x.manifest.job_id)
                    include_jobs = [job for job in task_info["jobs"] if job.output.answer and job.output.answer.lower().strip() != "none"]

                    st.markdown(f"_Note: {len(task_info['jobs']) - len(include_jobs)} jobs did not have relevant information._")
                    st.markdown(f"**Jobs with relevant information:**")
                    # print all the relevant information
                    for job in include_jobs:
                        st.markdown(f"**✅ Job {job.manifest.job_id + 1} (Chunk {job.manifest.chunk_id + 1})**")
                        answer = job.output.answer.replace("$", "\\$")
                        st.markdown(f"Answer: {answer}")
                    
            elif isinstance(message, dict):
                if 'content' in message and isinstance(message['content'], (dict, str)):
                    try:
                        # Try to parse as JSON if it's a string
                        content = message['content'] if isinstance(message['content'], dict) else json.loads(message['content'])
                        st.json(content)
                    except json.JSONDecodeError:
                        st.write(message['content'])
                else:
                    st.write(message)
            else:
                message = message.replace("$", "\\$")
                st.markdown(message)

# Remove global variables
# local_client = None
# remote_client = None
# method = None

def initialize_clients(local_model_name, remote_model_name, provider, protocol, 
                      local_temperature, local_max_tokens, remote_temperature, remote_max_tokens,
                      api_key, num_ctx=4096):
    """Initialize the local and remote clients outside of the run_protocol function."""
    # Use session_state instead of global variables
    
    # Store model parameters in session state for potential reinitialization
    st.session_state.local_model_name = local_model_name
    st.session_state.remote_model_name = remote_model_name
    st.session_state.local_temperature = local_temperature
    st.session_state.local_max_tokens = local_max_tokens
    st.session_state.remote_temperature = remote_temperature
    st.session_state.remote_max_tokens = remote_max_tokens
    st.session_state.provider = provider
    st.session_state.api_key = api_key
    
    # For Minions we want asynchronous local chunk processing:
    if protocol == "CybertronMinions":
        use_async = True
        # For Minions, we use a fixed context size since it processes chunks
        minions_ctx = 4096
        st.session_state.local_client = OllamaClient(
            model_name=local_model_name,
            temperature=local_temperature,
            max_tokens=int(local_max_tokens),
            num_ctx=minions_ctx,
            structured_output_schema=JobOutput,
            use_async=use_async
        )
    else:
        use_async = False
        st.session_state.local_client = OllamaClient(
            model_name=local_model_name,
            temperature=local_temperature,
            max_tokens=int(local_max_tokens),
            num_ctx=num_ctx,
            structured_output_schema=None,
            use_async=use_async
        )
    
    if provider == "Anthropic":
        st.session_state.remote_client = AnthropicClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key
        )
    elif provider == "Together":
        st.session_state.remote_client = TogetherClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key
        )
    else:  # OpenAI
        st.session_state.remote_client = OpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key
        )

    if protocol == "CybertronMinions":
        st.session_state.method = Minions(st.session_state.local_client, st.session_state.remote_client, callback=message_callback)
    else:
        st.session_state.method = Minion(st.session_state.local_client, st.session_state.remote_client, callback=message_callback)
    
    return st.session_state.local_client, st.session_state.remote_client, st.session_state.method

def run_protocol(task, context, doc_metadata, status, protocol):
    """Run the protocol with pre-initialized clients."""
    setup_start_time = time.time()
    
    with status.container():
        messages_container = st.container()
        st.markdown(f"**Query:** {task}")
        
        # If context size has changed, we need to update the local client's num_ctx
        # But only for Minion protocol, not Minions (which processes chunks)
        if ('local_client' in st.session_state and 
            hasattr(st.session_state.local_client, 'num_ctx') and 
            protocol == "Minion" and 
            st.session_state.current_protocol == "Minion"):
            
            padding = 8000
            estimated_tokens = int(len(context) / 4 + padding) if context else 4096
            num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
            closest_value = min([x for x in num_ctx_values if x >= estimated_tokens], default=131072)
            
            # Only reinitialize if num_ctx needs to change
            if closest_value != st.session_state.local_client.num_ctx:
                st.write(f"Adjusting context window to {closest_value} tokens...")
                
                # According to Ollama documentation, num_ctx needs to be set during initialization
                # So we need to reinitialize the local client with the new num_ctx
                if ('local_model_name' in st.session_state and 
                    'local_temperature' in st.session_state and 
                    'local_max_tokens' in st.session_state and
                    'api_key' in st.session_state):
                    
                    # Reinitialize the local client with the new num_ctx
                    st.session_state.local_client = OllamaClient(
                        model_name=st.session_state.local_model_name,
                        temperature=st.session_state.local_temperature,
                        max_tokens=int(st.session_state.local_max_tokens),
                        num_ctx=closest_value,
                        structured_output_schema=None,  # Minion protocol doesn't use structured output
                        use_async=False  # Minion protocol doesn't use async
                    )
                    
                    # Reinitialize the method with the new local client
                    st.session_state.method = Minion(st.session_state.local_client, st.session_state.remote_client, callback=message_callback)
        
        setup_time = time.time() - setup_start_time
        st.write("Solving task...")
        execution_start_time = time.time()
        
        output = st.session_state.method(
            task=task,
            doc_metadata=doc_metadata,
            context=[context],
            max_rounds=5,
        )
        
        execution_time = time.time() - execution_start_time

    return output, setup_time, execution_time

def validate_openai_key(api_key):
    print("validate_openai_key")
    try:
        client = OpenAIClient(
            model_name="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say OpenAI!"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)

def validate_anthropic_key(api_key):
    try:
        client = AnthropicClient(
            model_name="claude-3-5-haiku-latest",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)

def validate_together_key(api_key):
    try:
        client = TogetherClient(
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------
#  Sidebar for LLM settings
# ---------------------------
with st.sidebar:
    st.subheader("LLM Provider Settings")

    provider_col, key_col = st.columns([1, 2])
    with provider_col:
        providers = ["OpenAI", "Together"]
        selected_provider = st.selectbox("Select LLM provider", options=providers, index=0)

    env_var_name = f"{selected_provider.upper()}_API_KEY"
    env_key = os.getenv(env_var_name)
    with key_col:
        user_key = st.text_input(
            f"{selected_provider} API Key (optional if set in environment)",
            type="password",
            value="",
            key=f"{selected_provider}_key"
        )
    api_key = user_key if user_key else env_key

    if api_key:
        if not "provider_key" in st.session_state:

            if selected_provider == "OpenAI":
                is_valid, msg = validate_openai_key(api_key)
            elif selected_provider == "Anthropic":
                is_valid, msg = validate_anthropic_key(api_key)
            elif selected_provider == "Together":
                is_valid, msg = validate_together_key(api_key)
            else:
                raise ValueError(f"Invalid provider: {selected_provider}")

            if is_valid:
                st.success("**✓ Valid API key.** You're good to go!")
                provider_key = api_key
            else:
                st.error(f"**✗ Invalid API key.** {msg}")
                provider_key = None
            st.session_state.provider_key = provider_key
        else:
            
            provider_key = st.session_state.provider_key
    else:
        st.error(f"**✗ Missing API key.** Input your key above or set the environment variable with `export {PROVIDER_TO_ENV_VAR_KEY[selected_provider]}=<your-api-key>`")
        provider_key = None
    
    # Protocol selection
    st.subheader("Protocol")
    protocol = st.segmented_control(
        "Communication protocol",
        options=["Minion", "CybertronMinions"],
        default="CybertronMinions"
    )

    # Model Settings
    st.subheader("Model Settings")

    # Create two columns for local and remote model settings
    local_col, remote_col = st.columns(2)

    # Local model settings
    with local_col:
        st.markdown("### Local Model")
        st.image("assets/minion_resized.png", use_container_width=True)
        local_model_options = {
            "llama3.2 (Recommended)": "llama3.2",
            "llama3.1:8b (Recommended)": "llama3.1:8b",
            "llama3.2:1b": "llama3.2:1b",
            "phi4": "phi4",
            "qwen2.5:1.5b": "qwen2.5:1.5b",
            "qwen2.5:3b (Recommended)": "qwen2.5:3b",
            "qwen2.5:7b (Recommended)": "qwen2.5:7b",
            "qwen2.5:14b": "qwen2.5:14b",
            "mistral7b": "mistral7b",
            "deepseek-r1:1.5b": "deepseek-r1:1.5b",
            "deepseek-r1:7b": "deepseek-r1:7b",
            "deepseek-r1:8b": "deepseek-r1:8b"
        }
        local_model_display = st.selectbox("Model", options=list(local_model_options.keys()), index=0)
        local_model_name = local_model_options[local_model_display]

        show_local_params = st.toggle("Change defaults", value=False, key="local_defaults_toggle")
        if show_local_params:
            local_temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.05, key="local_temp")
            local_max_tokens_str = st.text_input("Max tokens per turn", "4096", key="local_tokens")
            try:
                local_max_tokens = int(local_max_tokens_str)
            except ValueError:
                st.error("Local Max Tokens must be an integer.")
                st.stop()
        else:
            local_temperature = 0.0
            local_max_tokens = 4096

    # Remote model settings
    with remote_col:
        st.markdown("### Remote Model")
        st.image("assets/llama3.1.png", use_container_width=True)
        if selected_provider == "OpenAI":
            model_mapping = {
                "gpt-4o (Recommended)": "gpt-4o",
                "gpt-4o-mini": "gpt-4o-mini",
                "o3-mini": "o3-mini",
                "o1": "o1"
            }
            default_model_index = 0
        elif selected_provider == "Anthropic":
            model_mapping = {
                "claude-3-5-sonnet-latest (Recommended)": "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest": "claude-3-5-haiku-latest",
                "claude-3-opus-latest": "claude-3-opus-latest"
            }
            default_model_index = 0
        elif selected_provider == "Together":
            model_mapping = {
                "DeepSeek-V3 (Recommended)": "deepseek-ai/DeepSeek-V3",
                "Qwen 2.5 72B (Recommended)": "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "Meta Llama 3.1 405B (Recommended)": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
                "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            }
            default_model_index = 0
        else:
            model_mapping = {}
            default_model_index = 0

        remote_model_display = st.selectbox(
            "Model",
            options=list(model_mapping.keys()),
            index=default_model_index,
            key="remote_model"
        )
        remote_model_name = model_mapping[remote_model_display]

        show_remote_params = st.toggle("Change defaults", value=False, key="remote_defaults_toggle")
        if show_remote_params:
            remote_temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.05, key="remote_temp")
            remote_max_tokens_str = st.text_input("Max Tokens", "4096", key="remote_tokens")
            try:
                remote_max_tokens = int(remote_max_tokens_str)
            except ValueError:
                st.error("Remote Max Tokens must be an integer.")
                st.stop()
        else:
            remote_temperature = 0.0
            remote_max_tokens = 4096

# -------------------------
#   Main app layout
# -------------------------
# if protocol == "Minions":
#     st.title("Minions!")
# else:
#     st.title("Minion!")

st.subheader("Context")
text_input = st.text_area(
    "Optionally paste text here",
    value="",
    height=150
)

uploaded_files = st.file_uploader(
    "Or upload PDF / TXT (Not more than a 100 pages total!)",
    type=['txt', 'pdf', 'json'],
    accept_multiple_files=True
)

file_content = ""
if uploaded_files:
    all_file_contents = []
    total_size = 0
    file_names = []
    for uploaded_file in uploaded_files:
        try:
            file_type = uploaded_file.name.lower().split('.')[-1]
            current_content = ""
            file_names.append(uploaded_file.name)
            
            if file_type == 'pdf':
                current_content = extract_text_from_pdf(uploaded_file.read()) or ""
            # elif file_type in ['png', 'jpg', 'jpeg']:
            #     current_content = extract_text_from_image(uploaded_file.read()) or ""
            else:
                current_content = uploaded_file.getvalue().decode()
            
            if current_content:
                all_file_contents.append("\n--------------------")
                all_file_contents.append(f"### Content from {uploaded_file.name}:\n{current_content}")
                total_size += uploaded_file.size
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

    if all_file_contents:
        file_content = "\n".join(all_file_contents)
        # Create doc_metadata string
        doc_metadata = f"Input: {len(file_names)} documents ({', '.join(file_names)}). Total extracted text length: {len(file_content)} characters."
    else:
        doc_metadata = ""
else:
    doc_metadata = ""

if text_input and file_content:
    context = f"{text_input}\n## file upload:\n{file_content}"
    if doc_metadata:
        doc_metadata = f"Input: Text input and {doc_metadata[6:]}"  # Remove "Input: " from start
elif text_input:
    context = text_input
    doc_metadata = f"Input: Text input only. Length: {len(text_input)} characters."
else:
    context = file_content
padding = 8000
estimated_tokens = int(len(context) / 4 + padding) if context else 4096
num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
closest_value = min([x for x in num_ctx_values if x >= estimated_tokens], default=131072)
num_ctx = closest_value

if context:
    st.info(f"Extracted: {len(file_content)} characters. Ballpark estimated total tokens: {estimated_tokens - padding}")

with st.expander("View Combined Context"):
    st.text(context)

# Add required context description
context_description = st.text_input(
    "One-sentence description of the context (Required)",
    key="context_description"
)

# -------------------------
#  Chat-like user input
# -------------------------
user_query = st.chat_input("Enter your query or request here...", key="persistent_chat")

# A container at the top to display final answer
final_answer_placeholder = st.empty()

if user_query:
    # Validate context description is provided
    if not context_description.strip():
        st.error("Please provide a one-sentence description of the context before proceeding.")
        st.stop()

    with st.status(f"Running {protocol} protocol...", expanded=True) as status:
        if True:
            # Initialize clients first (only once) or if protocol has changed
            if ('local_client' not in st.session_state or 
                'remote_client' not in st.session_state or 
                'method' not in st.session_state or
                'current_protocol' not in st.session_state or
                st.session_state.current_protocol != protocol):
                
                st.write(f"Initializing clients for {protocol} protocol...")
                initialize_clients(
                    local_model_name,
                    remote_model_name,
                    selected_provider,
                    protocol,
                    local_temperature,
                    local_max_tokens,
                    remote_temperature,
                    remote_max_tokens,
                    provider_key,
                    num_ctx
                )
                # Store the current protocol in session state
                st.session_state.current_protocol = protocol
            
            # Then run the protocol with pre-initialized clients
            output, setup_time, execution_time = run_protocol(
                user_query,
                context,
                doc_metadata,
                status,
                protocol
            )

            status.update(label=f"{protocol} protocol execution complete!", state="complete")

            # Display final answer at the bottom with enhanced styling
            st.markdown("---")  # Add a visual separator
            # render the oriiginal query
            st.markdown("## 🚀 Query")
            st.info(user_query)
            st.markdown("## 🎯 Final Answer")
            st.info(output["final_answer"])

            # Timing info
            st.header("Runtime")
            total_time = setup_time + execution_time
            # st.metric("Setup Time", f"{setup_time:.2f}s", f"{(setup_time/total_time*100):.1f}% of total")
            st.metric("Execution Time", f"{execution_time:.2f}s")

            # Token usage for both protocols
            if "local_usage" in output and "remote_usage" in output:
                st.header("Token Usage")
                local_total = output["local_usage"].prompt_tokens + output["local_usage"].completion_tokens
                remote_total = output["remote_usage"].prompt_tokens + output["remote_usage"].completion_tokens
                c1, c2 = st.columns(2)
                c1.metric(
                    f"{local_model_name} (Local) Total Tokens",
                    f"{local_total:,}",
                    f"Prompt: {output['local_usage'].prompt_tokens:,}, "
                    f"Completion: {output['local_usage'].completion_tokens:,}"
                )
                c2.metric(
                    f"{remote_model_name} (Remote) Total Tokens",
                    f"{remote_total:,}",
                    f"Prompt: {output['remote_usage'].prompt_tokens:,}, "
                    f"Completion: {output['remote_usage'].completion_tokens:,}"
                )
                # Convert to long format DataFrame for explicit ordering
                df = pd.DataFrame({
                    "Model": [f"Local: {local_model_name}", f"Local: {local_model_name}", 
                             f"Remote: {remote_model_name}", f"Remote: {remote_model_name}"],
                    "Token Type": ["Prompt Tokens", "Completion Tokens", "Prompt Tokens", "Completion Tokens"],
                    "Count": [
                        output["local_usage"].prompt_tokens,
                        output["local_usage"].completion_tokens,
                        output["remote_usage"].prompt_tokens,
                        output["remote_usage"].completion_tokens
                    ]
                })
                st.bar_chart(
                    df,
                    x="Model",
                    y="Count",
                    color="Token Type"
                )

                # Display cost information for OpenAI models
                if selected_provider == "OpenAI" and remote_model_name in OPENAI_PRICES:
                    st.header("Remote Model Cost")
                    pricing = OPENAI_PRICES[remote_model_name]
                    prompt_cost = (output["remote_usage"].prompt_tokens / 1_000_000) * pricing["input"]
                    completion_cost = (output["remote_usage"].completion_tokens / 1_000_000) * pricing["output"]
                    total_cost = prompt_cost + completion_cost

                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Prompt Cost",
                        f"${prompt_cost:.4f}",
                        f"{output['remote_usage'].prompt_tokens:,} tokens (at ${pricing['input']:.2f}/1M)"
                    )
                    col2.metric(
                        "Completion Cost",
                        f"${completion_cost:.4f}",
                        f"{output['remote_usage'].completion_tokens:,} tokens (at ${pricing['output']:.2f}/1M)"
                    )
                    col3.metric(
                        "Total Cost",
                        f"${total_cost:.4f}",
                        f"{remote_total:,} total tokens"
                    )

            # Display meta information for minions protocol
            if "meta" in output:
                st.header("Meta Information")
                for round_idx, round_meta in enumerate(output["meta"]):
                    st.subheader(f"Round {round_idx + 1}")
                    if "local" in round_meta:
                        st.write(f"Local jobs: {len(round_meta['local']['jobs'])}")
                    if "remote" in round_meta:
                        st.write(f"Remote messages: {len(round_meta['remote']['messages'])}")
        try:
            a = 1 
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")