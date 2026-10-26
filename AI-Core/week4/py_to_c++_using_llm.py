"""Port a Python program to high-performance C++ with an LLM, then compile and run it.

Module layout (top to bottom): imports, configuration, clients, prompts,
helpers, the Python program under test, and finally the entry point.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import platform
import shutil
import subprocess
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Read the API keys from the .env file
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Which model each bot uses
GPT_MODEL = "gpt-4.1-mini"
CLAUDE_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Only the reasoning families accept reasoning_effort; gpt-4.1 and friends
# reject it outright with "Unrecognized request argument supplied".
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

OUTPUT_FILE = "main.cpp"
EXECUTABLE = "main.exe" if os.name == "nt" else "main"

# Generated code can fail to terminate, so never block on it forever
COMPILE_TIMEOUT = 300
RUN_TIMEOUT = 600

# How many times a model may see its own compiler errors and try again
MAX_REPAIR_ATTEMPTS = 2

# The winget LLVM installer does not add itself to PATH, so fall back to the
# default install location before giving up and letting subprocess report it.
LLVM_BIN = "C:/Program Files/LLVM/bin"
CLANG = "clang++"


def find_clang():
    found = shutil.which(CLANG)
    if found:
        return found
    fallback = os.path.join(LLVM_BIN, CLANG + ".exe")
    return fallback if os.path.isfile(fallback) else CLANG


# Build commands, as recommended by describe_build_setup() below.
# -O3 -ffast-math rather than -Ofast: the latter is deprecated as of clang 23.
# LTO on the MSVC target requires the lld linker to be selected explicitly.
COMPILE_COMMAND = [
    find_clang(), "-std=c++17", "-O3", "-ffast-math", "-march=native",
    "-flto=thin", "-fuse-ld=lld", "-DNDEBUG", OUTPUT_FILE, "-o", EXECUTABLE,
]
RUN_COMMAND = [os.path.join(".", EXECUTABLE)]

# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

CANDIDATE_COMPILERS = [CLANG, "g++", "c++", "icpx", "cl"]


def _compiler_version(executable):
    """First line of `<executable> --version`, or None if it won't report one."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def retrieve_system_info():
    """Return a plain-text report of this machine, for the LLM to target."""
    lines = [
        f"Operating system: {platform.system()} {platform.release()}",
        f"Platform: {platform.platform()}",
        f"Architecture: {platform.machine()} ({platform.architecture()[0]})",
        f"Processor: {platform.processor() or 'unknown'}",
        f"Python: {platform.python_version()} at {sys.executable}",
    ]

    # Richer CPU/memory detail when the optional packages are installed
    try:
        import cpuinfo

        brand = cpuinfo.get_cpu_info().get("brand_raw")
        if brand:
            lines.append(f"CPU: {brand}")
    except Exception:
        pass

    logical = os.cpu_count()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        lines.append(f"CPU cores: {physical} physical / {logical} logical")
        lines.append(f"RAM: {psutil.virtual_memory().total / 1024 ** 3:.1f} GB")
    except Exception:
        lines.append(f"CPU cores: {logical} logical")

    found = []
    for name in CANDIDATE_COMPILERS:
        # Not every Windows toolchain puts itself on PATH, so check the usual
        # install locations too before reporting a compiler as missing.
        path = shutil.which(name)
        if not path:
            candidate = os.path.join(LLVM_BIN, name + ".exe")
            path = candidate if os.path.isfile(candidate) else None
        if not path:
            continue
        version = _compiler_version(path)
        found.append(f"  {name}: {path}" + (f" -- {version}" if version else ""))
    lines.append("C++ compilers available:")
    lines.extend(found or ["  none found"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
# All three are reached through the OpenAI-compatible interface; comment out the
# Claude or Gemini entries if you're not using them.

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = OpenAI(
    api_key=ANTHROPIC_API_KEY,
    base_url="https://api.anthropic.com/v1/",
)
gemini_client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""


def build_setup_prompt(system_info):
    return f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.

If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
```python
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
```
Please tell me exactly what I should use for the compile_command and run_command.

System information:
{system_info}
"""


def user_prompt_for(python, system_info):
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called {OUTPUT_FILE} and then compiled and executed; the compilation command is:
{COMPILE_COMMAND}
Respond only with C++ code.
Watch for Python syntax that is not valid C++: in particular, write numeric literals
such as 200_000_000 as 200'000'000 or 200000000, since C++ digit separators use an
apostrophe and an underscore is parsed as a user-defined literal suffix.
Python code to port:

```python
{python}
```
"""


def repair_prompt_for(errors):
    return f"""
That C++ failed to compile with these errors:

{errors}

Fix the code so it compiles with the same command. Respond only with the complete corrected C++ code.
"""


def messages_for(python, system_info):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_for(python, system_info)},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def describe_build_setup(system_info):
    """Ask GPT which compiler to install and which commands to use."""
    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": build_setup_prompt(system_info)}],
    )
    return response.choices[0].message.content


def write_output(cpp):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(cpp)


def strip_code_fences(reply):
    """Remove a surrounding ``` fence, whatever language tag it carries."""
    text = reply.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"


def request_cpp(client, model, messages):
    """Send one chat request and return the C++ it contains."""
    extra = {}
    if model.startswith(REASONING_MODEL_PREFIXES):
        extra["reasoning_effort"] = "high"
    response = client.chat.completions.create(model=model, messages=messages, **extra)
    reply = response.choices[0].message.content
    if not reply:
        raise RuntimeError(f"{model} returned an empty response")
    return strip_code_fences(reply)


def compile_cpp():
    """Compile OUTPUT_FILE. Returns None on success, else the compiler's errors."""
    try:
        compiled = subprocess.run(
            COMPILE_COMMAND, text=True, capture_output=True, timeout=COMPILE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return f"Compilation timed out after {COMPILE_TIMEOUT}s"
    if compiled.returncode != 0:
        return (compiled.stderr or compiled.stdout).strip()
    return None


def port(client, model, python, system_info):
    """Get a C++ port from one model and compile it, returning True on success.

    When the build fails, the compiler errors go back to the model so it can
    correct its own code - up to MAX_REPAIR_ATTEMPTS times.
    """
    messages = messages_for(python, system_info)
    cpp = request_cpp(client, model, messages)
    write_output(cpp)

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        errors = compile_cpp()
        if errors is None:
            return True
        # The diagnostics are the whole point of a failed build - show them
        print(f"Compilation failed; asking {model} to fix it "
              f"(repair {attempt}/{MAX_REPAIR_ATTEMPTS}):")
        print(errors)
        messages += [
            {"role": "assistant", "content": cpp},
            {"role": "user", "content": repair_prompt_for(errors)},
        ]
        cpp = request_cpp(client, model, messages)
        write_output(cpp)

    errors = compile_cpp()
    if errors is None:
        return True
    print(f"Compilation still failing after {MAX_REPAIR_ATTEMPTS} repairs:")
    print(errors)
    return False


def run_python(code):
    namespace = {"__builtins__": __builtins__}
    exec(code, namespace)


def run_cpp(runs=3):
    """Run the compiled executable. Returns True only if every run succeeded."""
    for attempt in range(1, runs + 1):
        try:
            result = subprocess.run(
                RUN_COMMAND, text=True, capture_output=True, timeout=RUN_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            print(f"Run {attempt} timed out after {RUN_TIMEOUT}s")
            return False
        if result.returncode != 0:
            print(f"Run {attempt} failed (exit {result.returncode}):")
            print((result.stderr or result.stdout).strip())
            return False
        print(result.stdout.strip())
    return True


# ---------------------------------------------------------------------------
# The Python program we want ported
# ---------------------------------------------------------------------------

PI = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if HF_TOKEN:
        login(HF_TOKEN, add_to_git_credential=True)

    system_info = retrieve_system_info()
    print(system_info)
    print(describe_build_setup(system_info))

    # Baseline: the original Python
    run_python(PI)

    # Then each model's C++ port, compiled with the commands above
    for client, model in (
        (openai_client, GPT_MODEL),
        (anthropic_client, CLAUDE_MODEL),
        (gemini_client, GEMINI_MODEL),
    ):
        print(f"\n--- {model} ---")
        try:
            compiled = port(client, model, PI, system_info)
            print(f"\n--- {model}-end ---")
        except Exception as error:
            # A model being unavailable shouldn't cost us the other two
            print(f"{model} failed to produce a port: {error}")
            continue
        if compiled:
            run_cpp()


if __name__ == "__main__":
    main()
