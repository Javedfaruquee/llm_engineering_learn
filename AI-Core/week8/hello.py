"""Hello world on Modal: functions that report which city they ran in.

Used by day1_modal_and_specialist_agent.py. Run locally, they report your location; run on
Modal, they report the data centre's.
"""

import modal
from modal import Image

# Setup: the app, and the container image its functions run in (Debian + requests)
app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")


def where_am_i() -> str:
    """Look up this machine's public location from its IP address."""
    import requests                         # imported here: available inside the container image

    data = requests.get("https://ipinfo.io/json").json()
    return f"Hello from {data['city']}, {data['region']}, {data['country']}!!"


@app.function(image=image)
def hello() -> str:
    """Report where this ran - wherever Modal chooses."""
    return where_am_i()


# Added thanks to student Tue H.: pin the function to EU data centres.
# Region selection costs marginally more credits: https://modal.com/docs/guide/region-selection
@app.function(image=image, region="eu")
def hello_europe() -> str:
    """Report where this ran - always an EU region."""
    return where_am_i()
