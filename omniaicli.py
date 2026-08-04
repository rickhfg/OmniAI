# omniaicli.py
import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- Configuration ---
_LOCAL_ENV_FILE = Path(__file__).resolve().with_name(".env")
if _LOCAL_ENV_FILE.is_file():
    load_dotenv(dotenv_path=_LOCAL_ENV_FILE)

PUBLIC_PROVIDERS = frozenset({
    "openai", "anthropic", "openrouter", "deepseek", "gemini"
})


def _default_proxy_url():
    configured_url = os.getenv("OMNIAI_PROXY_URL", "").strip()
    if configured_url:
        return configured_url.rstrip("/")

    host = os.getenv("OMNIAI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("OMNIAI_PORT", "8000").strip() or "8000"
    return f"http://{host}:{port}".rstrip("/")


PROXY_URL = _default_proxy_url()
PROXY_AUTH_KEY = os.getenv("PROXY_AUTH_KEY")


class ConfigurationError(RuntimeError):
    """Raised when the CLI cannot safely authenticate with the proxy."""


def get_proxy_url(args=None):
    url = getattr(args, "proxy_url", None) or PROXY_URL
    return str(url).strip().rstrip("/")


def add_proxy_url_option(parser):
    """Allow the endpoint flag before or after a subcommand."""
    parser.add_argument(
        "--url",
        dest="proxy_url",
        default=argparse.SUPPRESS,
        help="Proxy base URL (overrides the global --url and environment).",
    )

# Default System Prompt (Editable)
DEFAULT_SYSTEM_PROMPT = (
    "Please provide your response in plain text. "
    "Avoid using Markdown formatting such as headers, bold text, or bullet lists. "
    "NSFW is permitted for this session."
)

# --- ANSI Color Codes for better UX ---
class Colors:
    RESET = '\033[0m'
    DIM = '\033[2m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    # AI-like color (Golden Yellow)
    AI_GOLD = '\033[38;5;220m'  # Using 256-color mode for a golden shade

def print_color(color, *args, **kwargs):
    """Prints text in a specified color."""
    print(f"{color}{' '.join(map(str, args))}{Colors.RESET}", **kwargs)

def clear_terminal():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API Interaction Logic ---

def get_headers():
    """Constructs the required authentication headers."""
    key = os.getenv("PROXY_AUTH_KEY")
    if key is None:
        key = PROXY_AUTH_KEY
    key = (key or "").strip()
    if not key:
        raise ConfigurationError(
            "PROXY_AUTH_KEY is required. Set the same non-empty Bearer token used by the proxy."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

def handle_status(args):
    """Checks the status of the OmniAI proxy server."""
    # Lazy import requests to speed up startup time.
    # The 'requests' library is heavy, so we only import it when strictly needed.
    import requests
    proxy_url = get_proxy_url(args)
    print_color(Colors.CYAN, f"Checking OmniAI proxy status at {proxy_url}...")
    try:
        response = requests.get(proxy_url, timeout=5)
        if response.status_code == 200:
            print_color(Colors.GREEN, "Proxy is running and reachable.")
            print(response.json())
        else:
            print_color(Colors.YELLOW, f"Proxy returned status {response.status_code}. It might be running but with issues.")
    except requests.exceptions.RequestException as e:
        print_color(Colors.RED, f"Error: Could not connect to the OmniAI proxy at {proxy_url}.")
        print_color(Colors.RED, f"Details: {e}")
        sys.exit(1)

def handle_models(args):
    """Lists all available models from the proxy."""
    import requests
    try:
        headers = get_headers()
    except ConfigurationError as e:
        print_color(Colors.RED, f"Configuration error: {e}")
        sys.exit(2)

    proxy_url = get_proxy_url(args)
    print_color(Colors.CYAN, "Fetching available models...")
    try:
        response = requests.get(f"{proxy_url}/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        models = [
            model for model in response.json().get('data', [])
            if str(model.get("owned_by", "")).lower() in PUBLIC_PROVIDERS
        ]
        if not models:
            print_color(Colors.YELLOW, "No public models found.")
            return

        print_color(Colors.GREEN, "Available Public Models:")
        for model in sorted(models, key=lambda item: item.get("id", "")):
            print(f"  - {model.get('id', '')}")

    except requests.exceptions.RequestException as e:
        print_color(Colors.RED, f"Error fetching models: {e}")
        sys.exit(1)

def construct_payload(model, messages, stream, reasoning_effort):
    """Constructs the API payload with model-specific parameters."""
    payload = {"model": model, "messages": messages, "stream": stream}

    model_name = str(model).lower()
    is_anthropic_model = "claude" in model_name
    is_deepseek_v4 = "deepseek-v4" in model_name
    is_latest_gemini = any(
        marker in model_name for marker in ("gemini-3.6", "gemini-3.5-flash-lite")
    )
    is_deepseek_thinking = is_deepseek_v4 and reasoning_effort not in {"off", "none"}
    is_openai_reasoning_model = (
        not is_anthropic_model and not is_deepseek_v4
        and any(
            marker in model_name
            for marker in ("o1", "o3", "o4", "gpt-5", "gemini-3")
        )
    )
    is_reasoning_on = reasoning_effort and reasoning_effort not in {"off", "none"}

    if not is_deepseek_thinking and not is_latest_gemini:
        payload["top_p"] = 1

    # Reasoning-capable public models and Anthropic thinking requests do not
    # accept the ordinary temperature parameter.
    if (
        not is_deepseek_thinking
        and not is_latest_gemini
        and not is_openai_reasoning_model
        and not (is_anthropic_model and is_reasoning_on)
    ):
        payload["temperature"] = 1

    # Add reasoning_effort or Claude thinking to the payload
    if is_reasoning_on:
        if is_anthropic_model:
            # Claude's Extended Thinking requires a specific 'thinking' object.
            # 'budget_tokens' dictates the thinking depth.
            # 'max_tokens' MUST be greater than 'budget_tokens'.
            payload["thinking"] = {"type": "enabled", "budget_tokens": 16000}
            payload["max_tokens"] = 20000
        else:
            # Standard OpenAI-style reasoning effort.
            payload["reasoning_effort"] = reasoning_effort
    elif is_deepseek_v4 and reasoning_effort in {"off", "none"}:
        payload["thinking"] = {"type": "disabled"}

    return payload

def handle_prompt(args):
    """Sends a single prompt to a model."""
    import requests
    try:
        headers = get_headers()
    except ConfigurationError as e:
        print_color(Colors.RED, f"Configuration error: {e}")
        sys.exit(2)

    # Ensure the content is a string, not an object
    content = args.prompt if isinstance(args.prompt, str) else str(args.prompt)
    messages = [{"role": "user", "content": content}]

    payload = construct_payload(args.model, messages, args.stream, args.reasoning_effort)

    try:
        response = requests.post(
            f"{get_proxy_url(args)}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=args.stream,
            timeout=300
        )
        response.raise_for_status()

        if args.stream:
            process_streaming_response(response)
        else:
            process_non_streaming_response(response)

    except requests.exceptions.RequestException as e:
        print_color(Colors.RED, f"API request failed: {e}")
        if e.response:
            print_color(Colors.RED, f"Response: {e.response.text}")
        sys.exit(1)

# --- Session Management ---

# Configuration for reasoning models.
# Keys are model IDs (or prefixes), values are supported reasoning effort levels.
# This dictionary is used by the /ot command to validate and switch reasoning levels.
REASONING_MODELS = {
    "deepseek-v4-pro": ["off", "low", "medium", "high", "xhigh", "max"],
    "deepseek-v4-flash": ["off", "low", "medium", "high", "xhigh", "max"],
    "gemini-3.6-flash": ["minimal", "low", "medium", "high"],
    "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro-preview": ["minimal", "low", "medium", "high"],
    "gpt-5.6-sol": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-5.6-terra": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-5.6-luna": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-5.5": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.4": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.4-mini": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.4-nano": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.2": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5": ["low", "medium", "high"],
    "gpt-5.1": ["low", "medium", "high"],
    "o4-mini": ["low", "medium", "high"],
    "o3": ["low", "medium", "high"],
    "claude-fable-5": ["low", "medium", "high", "xhigh", "max"],
    "claude-sonnet-5": ["off", "low", "medium", "high", "xhigh", "max"],
    "claude-opus-5": ["off", "low", "medium", "high", "xhigh", "max"],
    "claude-sonnet-4.5": ["off", "on"],
    "claude-sonnet-4-5": ["off", "on"],
    "claude-sonnet-4": ["off", "on"],
    "claude-3-7-sonnet": ["off", "on"],
    "claude-haiku-4.5": ["off", "on"],
    "claude-haiku-4-5": ["off", "on"],
    "claude-opus-4.6": ["off", "on"],
    "claude-opus-4-6": ["off", "on"],
    "claude-opus-4.5": ["off", "on"],
    "claude-opus-4-5": ["off", "on"],
    "claude-opus-4.1": ["off", "on"],
    "claude-opus-4-1": ["off", "on"],
    "claude-opus-4": ["off", "on"],
}
REASONING_MODELS_SORTED = tuple(sorted(REASONING_MODELS, key=len, reverse=True))

def get_base_dir():
    """Returns the base directory for the application data."""
    # Logic to ensure portability:
    # 1. If running as a frozen executable (e.g., PyInstaller), use the executable's directory.
    # 2. If running as a script, use the script's directory.
    # This avoids storing data in the user's home directory, making the app self-contained.
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    # Otherwise use the script's directory
    return os.path.dirname(os.path.abspath(__file__))

def get_session_path(name="default"):
    """Returns the full path for a session file."""
    base_dir = get_base_dir()
    session_dir = os.path.join(base_dir, ".omniaicli", "sessions")
    if not os.path.exists(session_dir):
        os.makedirs(session_dir, exist_ok=True)
    return os.path.join(session_dir, f"{name}.json")

def save_session(name, messages, model):
    """Saves the current conversation state to a file."""
    path = get_session_path(name)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"model": model, "messages": messages}, f, indent=2)
        print_color(Colors.GREEN, f"Session '{name}' saved to {path}.")
    except Exception as e:
        print_color(Colors.RED, f"Error saving session: {e}")

def load_session(name):
    """Loads a conversation state from a file."""
    path = get_session_path(name)
    if not os.path.exists(path):
        print_color(Colors.RED, f"Session '{name}' not found.")
        return None, None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print_color(Colors.GREEN, f"Session '{name}' loaded.")
        return data.get("messages", []), data.get("model", "")
    except Exception as e:
        print_color(Colors.RED, f"Error loading session: {e}")
        return None, None

def get_favorites_path():
    """Returns the full path for the favorites file."""
    base_dir = get_base_dir()
    config_dir = os.path.join(base_dir, ".omniaicli")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "favorites.json")

def load_favorites():
    """Loads the list of favorite models."""
    path = get_favorites_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_favorites(favorites):
    """Saves the list of favorite models."""
    path = get_favorites_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, indent=2)
    except Exception as e:
        print_color(Colors.RED, f"Error saving favorites: {e}")

def handle_chat(args):
    """Starts an interactive chat session."""
    clear_terminal()
    print_color(Colors.CYAN, f"Starting interactive chat with '{args.model}'.")
    if args.reasoning_effort:
        print_color(Colors.CYAN, f"Reasoning effort set to '{args.reasoning_effort}'.")
    print_color(Colors.CYAN, "Commands:")
    print_color(Colors.CYAN, "  /clear, /c        : Clear screen and context.")
    print_color(Colors.CYAN, "  /model <id>       : Switch model mid-conversation.")
    print_color(Colors.CYAN, "  /save <name>      : Save session (default: 'default').")
    print_color(Colors.CYAN, "  /load <name>      : Load session.")
    print_color(Colors.CYAN, "  /del, /delete     : Delete the latest message.")
    print_color(Colors.CYAN, "  /restore, /undo   : Restore the last deleted message.")
    print_color(Colors.CYAN, "  exit, quit        : End session.")
    print_color(Colors.CYAN, "End lines with '\\' for multi-line input.")

    # Ensure system prompt content is a string
    user_system = args.system if isinstance(args.system, str) else str(args.system) if args.system else ""

    # Combine user system prompt with default constraint
    system_content = f"{user_system} {DEFAULT_SYSTEM_PROMPT}".strip()

    system_prompt = {"role": "system", "content": system_content}
    messages = [system_prompt]
    deleted_buffer = []

    if args.system:
        print_color(Colors.MAGENTA, f"System Prompt: {args.system}")

    # Lazy import requests to speed up startup time
    import requests
    try:
        headers = get_headers()
    except ConfigurationError as e:
        print_color(Colors.RED, f"Configuration error: {e}")
        sys.exit(2)

    while True:
        try:
            print(f"{Colors.YELLOW}You: {Colors.RESET}", end='', flush=True)
            # Handle multi-line input
            lines = []
            while True:
                try:
                    line = input()
                    if line.endswith('\\'):
                        # Remove the trailing backslash and add to lines
                        lines.append(line[:-1])
                    else:
                        # Add the final line and break
                        lines.append(line)
                        break
                except (KeyboardInterrupt, EOFError):
                    # Re-raise these exceptions to be caught by the outer except
                    raise

            prompt = '\n'.join(lines)

            # Check for single letter command to clear context
            if len(prompt) == 1 and prompt.isalpha():
                clear_terminal()
                messages = [system_prompt]
                if args.system:
                    print_color(Colors.MAGENTA, f"System Prompt: {args.system}")
                continue

            if prompt.lower() in ['exit', 'quit']:
                break

            if prompt.lower() in ['/clear', '/c']:
                clear_terminal()
                print_color(Colors.GREEN, "Screen and context cleared.")
                messages = [system_prompt]
                if args.system:
                    print_color(Colors.MAGENTA, f"System Prompt: {args.system}")
                continue

            # --- New Commands ---

            if prompt.startswith('/model'):
                parts = prompt.split(' ', 1)
                if len(parts) > 1 and parts[1].strip():
                    new_model = parts[1].strip()
                    args.model = new_model
                    print_color(Colors.GREEN, f"Model switched to '{args.model}'.")
                else:
                    print_color(Colors.CYAN, f"Current model: {args.model}")
                continue

            if prompt == '/ot':
                # Determine supported levels for current model
                supported_levels = []
                # Sort keys by length descending to match specific models first (e.g. gpt-5.2 before gpt-5)
                for model_key in REASONING_MODELS_SORTED:
                    if model_key in args.model:
                        supported_levels = REASONING_MODELS[model_key]
                        break

                if not supported_levels:
                    print_color(Colors.YELLOW, f"Model '{args.model}' does not support reasoning effort configuration via /ot.")
                    continue

                current_effort = args.reasoning_effort or "medium" # Default to medium if not set
                if current_effort not in supported_levels:
                     current_effort = supported_levels[0] # Fallback

                if len(supported_levels) == 2:
                    # Toggle behavior
                    new_effort = supported_levels[1] if current_effort == supported_levels[0] else supported_levels[0]
                    args.reasoning_effort = new_effort
                    print_color(Colors.GREEN, f"Reasoning effort toggled to: {args.reasoning_effort}")

                else:
                    # Selection behavior
                    print_color(Colors.CYAN, f"Current reasoning effort: {current_effort}")
                    print_color(Colors.CYAN, f"Available options: {', '.join(supported_levels)}")
                    print(f"{Colors.YELLOW}Select new effort (or press Enter to cancel): {Colors.RESET}", end='', flush=True)
                    selection = input().strip().lower()
                    if selection in supported_levels:
                        args.reasoning_effort = selection
                        print_color(Colors.GREEN, f"Reasoning effort set to: {args.reasoning_effort}")
                    elif selection == "":
                        print_color(Colors.YELLOW, "Cancelled.")
                    else:
                        print_color(Colors.RED, "Invalid selection.")
                continue

            if prompt == '/fav':
                favorites = load_favorites()
                if args.model not in favorites:
                    favorites.append(args.model)
                    save_favorites(favorites)
                    print_color(Colors.GREEN, f"Added '{args.model}' to favorites.")
                else:
                    print_color(Colors.YELLOW, f"'{args.model}' is already in favorites.")
                continue

            if prompt == '/unfav':
                favorites = load_favorites()
                if args.model in favorites:
                    favorites.remove(args.model)
                    save_favorites(favorites)
                    print_color(Colors.GREEN, f"Removed '{args.model}' from favorites.")
                else:
                    print_color(Colors.YELLOW, f"'{args.model}' is not in favorites.")
                continue

            if prompt in ['/modelfav', '/mf']:
                favorites = load_favorites()
                if not favorites:
                    print_color(Colors.YELLOW, "No favorites saved. Use /fav to add the current model.")
                    continue

                print_color(Colors.CYAN, "Favorite Models:")
                for i, model in enumerate(favorites, 1):
                    print(f"  {i}. {model}")

                print(f"{Colors.YELLOW}Select model # (or press Enter to cancel): {Colors.RESET}", end='', flush=True)
                selection = input().strip()

                if selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(favorites):
                        args.model = favorites[idx]
                        print_color(Colors.GREEN, f"Model switched to '{args.model}'.")
                    else:
                        print_color(Colors.RED, "Invalid selection number.")
                elif selection == "":
                    print_color(Colors.YELLOW, "Cancelled.")
                else:
                    print_color(Colors.RED, "Invalid input.")
                continue

            if prompt.startswith('/save'):
                parts = prompt.split()
                name = parts[1] if len(parts) > 1 else "default"
                save_session(name, messages, args.model)
                continue

            if prompt.startswith('/load'):
                parts = prompt.split()
                name = parts[1] if len(parts) > 1 else "default"
                loaded_msgs, loaded_model = load_session(name)
                if loaded_msgs:
                    messages = loaded_msgs
                    args.model = loaded_model
                    print_color(Colors.GREEN, f"Loaded session '{name}' with model '{args.model}'.")
                    # Optionally reprint the last few messages to remind context
                    if len(messages) > 1:
                        last_msg = messages[-1]
                        role_color = Colors.AI_GOLD if last_msg['role'] == 'assistant' else Colors.YELLOW
                        print_color(role_color, f"Last message ({last_msg['role']}): {last_msg['content'][:100]}...")
                continue

            if prompt in ['/help', '/h', '/?']:
                print_color(Colors.CYAN, "Commands:")
                print_color(Colors.CYAN, "  /clear, /c        : Clear screen and context.")
                print_color(Colors.CYAN, "  /model <id>       : Switch model mid-conversation.")
                print_color(Colors.CYAN, "  /ot               : Optimize/Toggle reasoning effort.")
                print_color(Colors.CYAN, "  /fav              : Add current model to favorites.")
                print_color(Colors.CYAN, "  /unfav            : Remove current model from favorites.")
                print_color(Colors.CYAN, "  /modelfav, /mf    : Select from favorite models.")
                print_color(Colors.CYAN, "  /save <name>      : Save session (default: 'default').")
                print_color(Colors.CYAN, "  /load <name>      : Load session.")
                print_color(Colors.CYAN, "  /del, /delete     : Delete the latest message.")
                print_color(Colors.CYAN, "  /restore, /undo   : Restore the last deleted message.")
                print_color(Colors.CYAN, "  /help, /h, /?     : Show this help message.")
                print_color(Colors.CYAN, "  exit, quit        : End session.")
                continue

            if prompt in ['/del', '/delete']:
                if len(messages) > 1: # Keep system prompt
                    removed = messages.pop()
                    deleted_buffer.append(removed)
                    if len(deleted_buffer) > 3:
                        deleted_buffer.pop(0) # Maintain buffer size of 3
                    print_color(Colors.YELLOW, f"Deleted last message ({removed['role']}).")
                else:
                    print_color(Colors.YELLOW, "No messages to delete.")
                continue

            if prompt in ['/restore', '/undo']:
                if deleted_buffer:
                    restored = deleted_buffer.pop()
                    messages.append(restored)
                    print_color(Colors.GREEN, f"Restored message ({restored['role']}): {restored['content'][:100]}...")
                else:
                    print_color(Colors.YELLOW, "Nothing to restore.")
                continue

            # --------------------

            # Handle empty input
            if not prompt.strip():
                continue

            # Ensure user prompt content is a string
            content = prompt if isinstance(prompt, str) else str(prompt)
            messages.append({"role": "user", "content": content})

            payload = construct_payload(args.model, messages, True, args.reasoning_effort)

            response = requests.post(
                f"{get_proxy_url(args)}/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()

            full_response = process_streaming_response(response)
            # Filter out asterisks before saving to message history
            filtered_response = full_response.replace('*', '')
            messages.append({"role": "assistant", "content": filtered_response})

        except requests.exceptions.RequestException as e:
            print_color(Colors.RED, f"\nAPI request failed: {e}")
            if e.response:
                print_color(Colors.RED, f"Response: {e.response.text}")
            if messages and messages[-1]["role"] == "user":
                messages.pop() # Remove the failed user message
        except KeyboardInterrupt:
            print_color(Colors.YELLOW, "\nExiting chat.")
            break
        except EOFError:
            # Handle Ctrl+D or end of input
            print_color(Colors.YELLOW, "\nExiting chat.")
            break
        except Exception as e:
            print_color(Colors.RED, f"\nAn unexpected error occurred: {e}")
            break

# --- Response Processing ---

def process_streaming_response(response):
    """Processes and prints a streaming API response with <think> tag support."""
    print_color(Colors.AI_GOLD, "Assistant: ", end='')
    sys.stdout.flush()
    full_response = ""

    # buffers and state
    buffer = ""
    in_think_block = False
    start_tag = "<think>"
    start_tag_keep_len = len(start_tag) - 1
    end_tag = "</think>"
    end_tag_keep_len = len(end_tag) - 1

    # We want to print <think> content in DIM/CYAN, and normal content in WHITE.
    # We need to parse the stream character by character (or chunk by chunk)
    # to detect <think> and </think> tags.

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[len('data: '):].strip()
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get('choices', [{}])[0].get('delta', {})

                    # 1. Standard Content
                    content = delta.get('content', '')

                    # 2. Reasoning Content (OpenAI o1/o3 style)
                    reasoning = delta.get('reasoning_content', '')

                    # 3. Claude-specific thinking
                    if not reasoning and delta.get('type') == 'thinking_delta':
                        reasoning = delta.get('thinking', '')

                    # Handle provider-neutral reasoning content fields.
                    # If this field is present, we treat it as being inside a think block automatically.
                    if reasoning:
                        if not in_think_block:
                            # Start of a new reasoning block (implicit)
                            # We might want to print a visual indicator like "Thinking..."
                            # but simpler is just to switch color.
                            pass

                        print_color(Colors.DIM, reasoning, end='')
                        sys.stdout.flush()
                        # We don't add reasoning to full_response usually,
                        # but for <think> tags in content, we might want to strip them later.
                        continue

                    # Handle content fields that may contain thinking tags.
                    if content:
                        buffer += content

                        while True:
                            if not in_think_block:
                                # Look for opening tag
                                start_idx = buffer.find(start_tag)

                                if start_idx != -1:
                                    # Found <think>
                                    # Print everything before the tag as normal text
                                    pre_content = buffer[:start_idx]
                                    if pre_content:
                                        print_color(Colors.WHITE, pre_content.replace('*', ''), end='')

                                    # Switch state
                                    in_think_block = True
                                    # Remove processed part including tag
                                    buffer = buffer[start_idx + len(start_tag):]

                                    # Visual indicator for start of thinking (optional)
                                    # print_color(Colors.DIM, "\n<think>\n", end='')
                                else:
                                    # No tag found yet.
                                    # To be safe against split tags (e.g. "<th" + "ink>"),
                                    # we keep the last few chars in buffer and print the rest.
                                    if len(buffer) > start_tag_keep_len:
                                        to_print = buffer[:-start_tag_keep_len]
                                        print_color(Colors.WHITE, to_print.replace('*', ''), end='')
                                        buffer = buffer[-start_tag_keep_len:]
                                    break

                            else: # in_think_block is True
                                # Look for closing tag
                                end_idx = buffer.find(end_tag)

                                if end_idx != -1:
                                    # Found </think>
                                    # Print everything before tag as thinking text
                                    think_content = buffer[:end_idx]
                                    if think_content:
                                        print_color(Colors.DIM, think_content, end='')

                                    # Switch state
                                    in_think_block = False
                                    # Remove processed part including tag
                                    buffer = buffer[end_idx + len(end_tag):]

                                    # Visual separation after thinking (optional)
                                    print()
                                else:
                                    # No closing tag found yet.
                                    # Keep last few chars to handle split tags
                                    if len(buffer) > end_tag_keep_len:
                                        to_print = buffer[:-end_tag_keep_len]
                                        print_color(Colors.DIM, to_print, end='')
                                        buffer = buffer[-end_tag_keep_len:]
                                    break

                        sys.stdout.flush()
                        full_response += content

                except json.JSONDecodeError:
                    print_color(Colors.RED, f"\nError decoding JSON chunk: {data_str}")

    # Print any remaining buffer
    if buffer:
        if in_think_block:
            print_color(Colors.DIM, buffer, end='')
        else:
            print_color(Colors.WHITE, buffer.replace('*', ''), end='')

    print() # Final newline
    return full_response

def process_non_streaming_response(response):
    """Processes and prints a non-streaming API response."""
    data = response.json()
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '[No content]')
    # Filter out asterisks
    filtered_content = content.replace('*', '')
    print_color(Colors.AI_GOLD, "Assistant:")
    print(filtered_content)

# --- Main CLI Setup ---

def main():
    parser = argparse.ArgumentParser(
        description="OmniAI CLI: A command-line interface for the OmniAI proxy.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--url",
        dest="proxy_url",
        default=PROXY_URL,
        help="Proxy base URL (default: OMNIAI_PROXY_URL or http://127.0.0.1:8000).",
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Status Command ---
    parser_status = subparsers.add_parser('status', help='Check the status of the OmniAI proxy server.')
    add_proxy_url_option(parser_status)
    parser_status.set_defaults(func=handle_status)

    # --- Models Command ---
    parser_models = subparsers.add_parser('models', help='List all available models from the proxy.')
    add_proxy_url_option(parser_models)
    parser_models.set_defaults(func=handle_models)

    # --- Prompt Command ---
    parser_prompt = subparsers.add_parser('prompt', help='Send a single prompt to a model.')
    add_proxy_url_option(parser_prompt)
    parser_prompt.add_argument('model', help='The model ID to use.')
    parser_prompt.add_argument('prompt', help='The prompt to send to the model.')
    parser_prompt.add_argument('--no-stream', dest='stream', action='store_false', help='Disable streaming and wait for the full response.')
    parser_prompt.add_argument(
        '--reasoning-effort',
        choices=['off', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'],
        help='Set the reasoning effort for the model (if supported).',
    )
    parser_prompt.set_defaults(func=handle_prompt)

    # --- Chat Command ---
    parser_chat = subparsers.add_parser('chat', help='Start an interactive chat session with a model.')
    add_proxy_url_option(parser_chat)
    parser_chat.add_argument('model', help='The model ID to use for the chat session.')
    parser_chat.add_argument('--system', help='An initial system prompt to set the context.')
    parser_chat.add_argument(
        '--reasoning-effort',
        choices=['off', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'],
        help='Set the reasoning effort for the model (if supported).',
    )
    parser_chat.set_defaults(func=handle_chat)

    args = parser.parse_args()

    if args.command != "status":
        try:
            get_headers()
        except ConfigurationError as e:
            print_color(Colors.RED, f"Configuration error: {e}")
            return 2

    # Handle reasoning effort suffix in model name (e.g., gpt-5.6-sol-max).
    if args.command in ['prompt', 'chat'] and args.model:
        for suffix in ['-minimal', '-none', '-low', '-medium', '-high', '-xhigh', '-max', '-off']:
            if args.model.endswith(suffix):
                base_model = args.model[:-len(suffix)]
                # Check if base model matches reasoning-capable families
                if any(
                    family in base_model.lower()
                    for family in [
                        'o1', 'o3', 'o4', 'gpt-5', 'deepseek-v4', 'gemini-3',
                        'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
                    ]
                ):
                    # Set reasoning effort if not already specified by flag
                    if not args.reasoning_effort:
                        args.reasoning_effort = suffix[1:]
                    args.model = base_model
                break

    args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
