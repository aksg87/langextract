# Organizing Extractions into a Hierarchical Index

`lx.extract` returns a flat list of grounded extractions. On a long document,
such as a specification or a novel, that list can run to hundreds or thousands
of items. `lx.document_index` groups the extractions from the leaves upward
into a tree, so a reader or an agent can scan a short table of contents first
and open only the nodes it needs. Every leaf keeps its original `Extraction`
objects, with their character offsets into the source text.

Each level of the tree is a key function that maps an extraction to a group
name: `by_text`, `by_class`, `by_attribute(name)`, or any function with the
same signature. Building the index makes no model calls.

## Example: HTTP status codes in RFC 9110

Suppose `result` holds the 87 `status_code` extractions from one `lx.extract`
run with `gemini-3.5-flash` on the Status Codes section of RFC 9110 (HTTP
Semantics). Each extraction has a code such as `401` as its `extraction_text`,
`reason_phrase` and `meaning` attributes, and a `char_interval` into the RFC
text.

The first level merges repeated mentions of a code into one leaf. The second
level groups the leaves by the code's first digit:

```python
import json

import langextract as lx
from langextract import document_index


def http_class(extraction: lx.data.Extraction) -> str:
  """Maps a status code such as "404" to its class, such as "4xx"."""
  return extraction.extraction_text.strip()[:1] + "xx"


index = document_index.build_index(
    result, levels=[document_index.by_text, http_class]
)
print(index.to_toc(max_depth=0))
```

```
0001 | 1xx [3 status_code] — Includes 1xx, 100, 101.
0005 | 2xx [23 status_code] — Includes 2xx, 200, 204, 201, 202, 203, 205, 206.
0014 | 3xx [25 status_code] — Includes 3xx, 301, 302, 307, 308, 300, 303, 304, 305, 306.
0025 | 4xx [29 status_code] — Includes 4xx, 400, 401, 402, 403, 404, 410, 405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418, 421, 422, 426.
0049 | 5xx [7 status_code] — Includes 5xx, 500, 501, 502, 503, 504, 505.
```

The 87 extractions become 51 leaves under 5 roots: one leaf for each of the 46
status codes, plus one for each class name, such as `4xx`, that the RFC also
mentions. Each line shows the node id, the title, the number of extractions per
class, and a summary. These 5 lines take 447 characters. The full
`index.to_toc()` has one line per node, 56 in total, and each leaf's summary
lists its attribute values.

`node_view` returns one node as plain JSON, which is what an agent reads after
picking a node id from the table of contents. A parent's view lists its
children; a leaf's view lists its extractions with their offsets and, when
`context_chars` is set, the surrounding source text:

```python
print(json.dumps(index.node_view("0028", context_chars=40), indent=2))
```

```
{
  "node_id": "0028",
  "path": [
    "4xx",
    "401"
  ],
  "extraction_count": 1,
  "summary": "reason_phrase: Unauthorized; meaning: the request has not been applied because it lacks valid authentication credentials for the target resource",
  "children": [],
  "extractions": [
    {
      "extraction_class": "status_code",
      "extraction_text": "401",
      "attributes": {
        "reason_phrase": "Unauthorized",
        "meaning": "the request has not been applied because it lacks valid authentication credentials for the target resource"
      },
      "start_index": 354873,
      "end_index": 354876,
      "context": "r deceptive request routing).\n\n15.5.2.  401 Unauthorized\n\n   The 401 (Unauthorized)"
    }
  ]
}
```

## Example: grouping several classes in *Romeo and Juliet*

A key function can also group extractions of different classes. Suppose
`rj_result` holds the 1,048 extractions from one single-pass run with
`gemini-3.5-flash` on the full text of *Romeo and Juliet* (see the
[full-text example](longer_text_example.md)): 447 `character`, 377 `emotion`
and 224 `relationship` extractions. This key function files each one under the
character it is about:

```python
def character_key(extraction: lx.data.Extraction) -> str:
  """Returns the character an extraction is about, in upper case."""
  attrs = extraction.attributes or {}
  if extraction.extraction_class == "character":
    name = extraction.extraction_text
  else:
    name = attrs.get("character") or attrs.get("character_1") or ""
  return name.strip(" .,:;!?").upper()


rj_index = document_index.build_index(rj_result, levels=character_key)
print(rj_index.to_toc(include_summaries=False))
```

```
0001 | ROMEO [77 character, 87 emotion, 64 relationship]
0002 | JULIET [61 character, 100 emotion, 47 relationship]
0003 | FRIAR LAWRENCE [30 character, 21 emotion, 8 relationship]
...
```

The 1,048 extractions become 51 leaves, so the table of contents has 51 lines.
The `ROMEO` leaf alone holds 228 extractions.

Leaves are only as clean as the key function. In this run, `FRIAR LAWRENCE`
(59 extractions) and `FRIAR LAURENCE` (1) are separate leaves, and loose
extractions such as `THEE` get leaves of their own. To merge such leaves, map
the aliases to one name inside the key function.

## Notes

- **Every extraction is kept.** Each input extraction is in exactly one leaf,
  and `index.all_extractions()` returns all of them. `find(node_id)` returns a
  node, and `path(node)` gives its titles from the root down.
- **Parent levels read each node's first extraction.** Above the leaves, the
  key function is called on the first extraction in each node. Choose parent
  keys that every extraction in a node shares, as `http_class` does for leaves
  built with `by_text`.
- **Deterministic.** The same extractions and key functions always produce the
  same tree and node ids. Groups are ordered by their first extraction.
