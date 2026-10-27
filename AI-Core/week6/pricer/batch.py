import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from functools import cache
from tqdm.auto import tqdm

load_dotenv(override=True)


@cache
def groq():
    """The Groq client, created on first use and then reused"""
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


MODEL = "openai/gpt-oss-20b"
BATCHES_FOLDER = "batches"                  # request files, under lite/ or full/
OUTPUT_FOLDER = "output"                    # result files, under lite/ or full/
state = Path("batches.pkl")                 # saved batch ids, so a run can be resumed

# Same prompt as pricer/preprocessor.py: five short labelled lines per product
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    """
    One Groq Batch API job summarising items[start:end]

    Lifecycle: make_file -> send_file -> submit_batch -> (wait) is_ready -> fetch_output
    -> apply_output. The class methods run that lifecycle for every batch at once, and
    the list of all batches is kept in the class attribute `batches`.
    Each item's `id` must be its position in `items`; it is sent as the request's
    custom_id so results can be matched back.
    """

    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        """Prepare a batch for items[start:end]; lite chooses the lite/ or full/ folder"""
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None                 # set by send_file
        self.batch_id = None                # set by submit_batch
        self.output_file_id = None          # set by is_ready once complete
        self.done = False                   # set by apply_output
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        """One JSONL line: a chat-completion request to summarise this item"""
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "reasoning_effort": "low",
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self):
        """Write this batch's requests to its JSONL file, one line per item"""
        batch_file = self.batches / self.filename
        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        """Upload the JSONL file to Groq and remember its file id"""
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:
            response = groq().files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        """Start the batch job on the uploaded file; Groq has up to 24 hours to finish it"""
        response = groq().batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        """Ask Groq whether the job has completed; if so, remember the results file id"""
        response = groq().batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        return status == "completed"

    def fetch_output(self):
        """Download the results file to the output folder"""
        output_file = str(self.output / self.filename)
        response = groq().files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        """Copy each summary from the downloaded results onto its item, matched by custom_id"""
        output_file = str(self.output / self.filename)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items, lite):
        """Split items into batches of BATCH_SIZE (nothing is sent yet)"""
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        """Write, upload and submit every batch"""
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        """Check each unfinished batch once, and download and apply the completed ones"""
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        """
        Pickle the batches (ids and progress) to `state`; the items themselves are left
        out to keep the file small, and put back afterwards
        """
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        """Restore the batches saved by save() and attach them to items"""
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
