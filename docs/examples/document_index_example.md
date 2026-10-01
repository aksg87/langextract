# Navigating Extractions from Long Structured Documents

`lx.extract` returns a flat list of grounded extractions. On a long structured
document such as a regulation, specification, filing or manual, that list can
run to hundreds or thousands of items. An agent that consumes the results has
to load all of them or search blindly. `lx.document_index` organizes the
results the way [PageIndex](https://github.com/VectifyAI/PageIndex) organizes
a document for retrieval: a tree of sections built from the headings, browsed
table of contents first.

**After extraction**, with no model calls:

1. **`build_index`** parses Markdown `#` headings or dotted numbered headings
   (`7.`, `1.2`, `15.3.1.`) into a tree of sections. Each section carries a
   character interval into the original text.
2. **`attach_extractions`** files every grounded extraction under the deepest
   section that contains it.
3. **`to_toc(hide_empty=True)`** gives a short overview with extraction counts
   per section, and **`section_view(node_id)`** returns one section's
   extractions as JSON data, with offsets and the subsections to open next.

**Before extraction**, the same tree can optionally decide where to extract:

4. **`summarize_index`** describes each section, which is optional and
   reusable across tasks, and **`select_sections`** makes one model call that
   reads the table of contents and picks sections relevant to the task.
5. **`lx.extract(..., chunk_filter=section_chunk_filter(sections))`** skips
   every chunk that does not overlap a selected section.

> **Warning:** This example indexes and extracts from RFC 9110 (~503,000
> characters) with `gemini-3.5-flash` and will incur API costs. Please review
> the [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
> before running it.

## Example: HTTP status codes from RFC 9110

RFC 9110 (HTTP Semantics) is a 292-section specification. The task below asks
for every status code it defines. All of them live under section 15, so a
uniform pass over the document spends most of its calls on text that cannot
contain an answer.

```python
import re
import textwrap

import requests
import langextract as lx
from langextract import document_index

text = requests.get("https://www.rfc-editor.org/rfc/rfc9110.txt").text

# 1. Build the section tree. No model calls.
index = document_index.build_index(text)
print(f"{len(list(index.iter_nodes()))} sections, {len(index.roots)} top-level")

# 2. Summarize sections so the navigator can judge relevance.
# Use a plain model for navigation; the extraction model carries a schema.
navigator = lx.factory.create_model_from_id("gemini-3.5-flash")
document_index.summarize_index(index, navigator)

# 3. Ask the model which sections matter for this task.
prompt = textwrap.dedent("""\
    Extract every HTTP status code defined in the text, with its reason
    phrase and a short statement of what the response means. Use the exact
    three-digit code from the text as extraction_text.""")

examples = [
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            15.9.1.  299 Example

               The 299 (Example) status code indicates that the server
               processed the request as an example and nothing was changed."""),
        extractions=[
            lx.data.Extraction(
                extraction_class="status_code",
                extraction_text="299",
                attributes={
                    "reason_phrase": "Example",
                    "meaning": "server processed the request as an example",
                },
            )
        ],
    )
]

sections = document_index.select_sections(
    index, prompt, navigator, examples=examples
)
for section in sections:
  print(section.node_id, " > ".join(index.heading_path(section)))

# 4. Extract only inside the selected sections.
result = lx.extract(
    text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-3.5-flash",
    max_char_buffer=1000,
    chunk_filter=document_index.section_chunk_filter(sections),
)

# 5. File the results under their sections for browsing.
index.attach_extractions(result)

lx.io.save_annotated_documents(
    [result], output_name="rfc9110_status_codes.jsonl", output_dir="."
)
```

## Sample output

Measured on 2026-10-01 with `gemini-3.5-flash`:

```
292 sections, 20 top-level
Summaries: 213 model calls (79 short sections used their own text)
Selected 2 sections:
  0180 Status Codes
  0282 IANA Considerations > Status Code Registration
Selected text: 60,656 of 502,907 chars
Chunks at max_char_buffer=1000: 540 total, 67 sent to the model
Extractions: 133, all grounded with exact spans, all inside the selected sections
Defined status codes found: 46/46
```

| | Uniform extraction | With section index |
|---|---|---|
| Extraction calls | 540 (chunk count, not run) | 67 |
| Navigation calls | 0 | 213 summaries + 1 selection |

Summaries are the one-time cost of indexing. They are reused for every later
task on the same document, while extraction savings apply on each run. For a
single run on a short document, `select_sections` over an index without
summaries is often enough, since headings alone carry most of the signal.

The model also returned the class ranges `1xx` to `4xx` from the overview
table, which a prompt can exclude if only concrete codes are wanted. Each
extraction carries a section path, for example
`Status Codes > Successful 2xx > 200 OK`.

## Browsing the results

The index is built from the result's own text, so this works on any
`lx.extract` output, filtered or not. Output below is from the run above.

```python
index = document_index.build_index(result.text)
index.attach_extractions(result)

print(index.to_toc(include_summaries=False, hide_empty=True))
```

```
0180 | Status Codes [87 status_code]
  0182 | Informational 1xx [2 status_code]
    0184 | 101 Switching Protocols [1 status_code]
  0185 | Successful 2xx [21 status_code]
    0186 | 200 OK [2 status_code]
  ...
  0206 | Client Error 4xx [29 status_code]
    0207 | 400 Bad Request [1 status_code]
    0208 | 401 Unauthorized [1 status_code]
  ...
```

```python
index.section_view("0206")
```

```json
{
  "node_id": "0206",
  "path": ["Status Codes", "Client Error 4xx"],
  "start_index": 354125,
  "end_index": 368845,
  "extractions": [
    {
      "extraction_class": "status_code",
      "extraction_text": "400",
      "attributes": {"reason_phrase": "Bad Request", "meaning": "..."},
      "start_index": 354582,
      "end_index": 354585
    }
  ],
  "subsections": [
    {"node_id": "0207", "title": "400 Bad Request", "extraction_count": 1}
  ]
}
```

| What an agent reads | Characters |
|---|---|
| All 133 extractions as a flat list | 35,153 |
| Overview of sections with extractions | 2,527 |
| One leaf section, such as 401 Unauthorized | 442 |

For an agent, expose the two calls as tools, for example a
`get_structure()` tool returning `index.to_toc(hide_empty=True)` and a
`get_section(node_id)` tool returning `index.section_view(node_id)`. The agent
reads the overview, opens the sections relevant to its question, and can
quote exact source spans through `start_index` and `end_index`. Use
`index.to_dict(include_extractions=True)` to store the whole tree as JSON.

## Notes

- `build_index` needs headings at the start of a line and uses one heading
  style per document: Markdown `#` headings when present, otherwise dotted
  numbered headings, so numbered list items in a Markdown file are not
  mistaken for sections. Documents without recognizable headings become a
  single section, which selects everything; pass your own `HeadingPattern`s
  for other conventions.
- Pass a model without an extraction schema to `summarize_index` and
  `select_sections`; the schema-constrained extraction model would answer in
  the extraction format. `lx.factory.create_model_from_id(model_id)` is enough.
- `select_sections` is told to include a section when unsure, and
  `section_chunk_filter` keeps any chunk overlapping a selected section, so a
  chunk straddling a section boundary is still processed. If the selection is
  empty a warning is logged and extraction returns no extractions.
- `chunk_filter` accepts any callable taking a `chunking.TextChunk`, so you
  can also filter by keywords or your own document knowledge without
  building an index.
