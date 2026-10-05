# Organizing Extractions into a Hierarchical Index

`lx.extract` returns a flat list of grounded extractions. On a long document,
such as a specification or a novel, that list can run to hundreds or thousands
of items. `lx.document_index` groups the extractions from the leaves upward
into a tree, so a reader or an agent can scan a short table of contents first
and open only the nodes it needs. Every leaf keeps its original `Extraction`
objects, with their character offsets into the source text.

Each level of the tree is a rule for grouping the level below. The first level
is always a key function that maps an extraction to a group name: `by_text`,
`by_class`, `by_attribute(name)`, or any function with the same signature. It
makes no model calls. The levels above it can be key functions too, or
natural-language instructions such as "Group the characters by household" that
a language model applies.

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

### Letting the model organize a level

No key function can tell which household a character belongs to. For that,
describe the level in words and pass `model_id` (or a `model` instance):

```python
rj_index = document_index.build_index(
    rj_result,
    levels=[
        character_key,
        "Group the characters by household: Montague, Capulet, the Prince"
        " and his kinsmen, the Church, or Other for anything that is not a"
        " character.",
    ],
    model_id="gemini-3.5-flash",
)
print(rj_index.to_toc(max_depth=0))
```

```
0001 | Montague [132 character, 113 emotion, 88 relationship] — Members, relatives, and loyal servants of the Montague household.
0008 | Capulet [207 character, 186 emotion, 105 relationship] — Members, relatives, servants, and close associates of the Capulet household.
0028 | The Prince and his kinsmen [58 character, 41 emotion, 20 relationship] — Prince Escalus, his noble kinsmen Mercutio and Paris, and the officers of the watch.
0038 | The Church [34 character, 23 emotion, 9 relationship] — Religious figures, including Friar Lawrence and Friar John, who offer spiritual guidance.
0042 | Other [16 character, 14 emotion, 2 relationship] — Non-character concepts, personifications, and minor external figures not affiliated with the main households.
```

The model never sees the source text. It gets one prompt listing the 51 leaves
by title, extraction counts and summary, and replies with titled groups, each
with a one-sentence summary. An agent looking for the Capulets now reads these
5 lines, opens node `0008` to see its 19 character leaves, and then opens only
the leaves it needs. If a reply leaves a node out, that node goes under a group
named `Other`, so every extraction stays in the tree.

Model output varies from build to build, so the output above is one build. In
3 builds with `gemini-3.5-flash` at temperature 0 on these 51 leaves:

- **Cost.** Each build made 1 model call with a 2,512-token prompt. Replies
  were 437–481 output tokens plus 4,435–7,691 thinking tokens, and a build
  took 16–40 seconds (median 22).
- **Accuracy.** For 25 of the named characters, the play leaves no doubt
  about the household. The builds placed 23, 25 and 25 of them under the right
  root; the first build put `ROSALINE` and `SAMPSON AND GREGORY` under
  `Other`.
- **Shape.** Two builds returned the 5 requested roots. The third split
  `Other` into characters and non-characters, for 6. No build left a node out
  of its reply, and all 1,048 extractions were present in every tree.

## Notes

- **Every extraction is kept.** Each input extraction is in exactly one leaf,
  and `index.all_extractions()` returns all of them. `find(node_id)` returns a
  node, and `path(node)` gives its titles from the root down.
- **Parent-level key functions read each node's first extraction.** Above the
  leaves, a key function is called on the first extraction in each node. Choose
  parent keys that every extraction in a node shares, as `http_class` does for
  leaves built with `by_text`.
- **Key-function levels are deterministic.** The same extractions and key
  functions always produce the same tree and node ids. Groups are ordered by
  their first extraction. Model levels are not deterministic; their titles,
  summaries and group boundaries can change between builds.
- **Model levels are batched.** A level with more nodes than `batch_size`
  (default 100) is sent in several prompts, and groups from different batches
  are not merged. To cap the number of roots, set `max_roots`, which reapplies
  the last rule until the top level is small enough.
