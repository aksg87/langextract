# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hierarchical section index for long documents and their extractions.

`lx.extract` returns a flat list of grounded extractions. On a long
structured document (a regulation, specification, filing or manual) that list
can run to thousands of items, which an agent consuming the results has to
read in full or search blindly. Following the approach popularized by
PageIndex (https://github.com/VectifyAI/PageIndex), this module builds a tree
of sections from the document's headings so the results can be navigated the
way a person reads a long report: table of contents first, then the relevant
section.

After extraction, no model calls needed:

    index = lx.document_index.build_index(result.text)
    index.attach_extractions(result)
    print(index.to_toc())          # sections with extraction counts
    index.section_view("0180")     # one section's extractions, as JSON data

Before extraction, the same tree can also decide where to extract. A model
reads the table of contents, picks the sections relevant to the task, and
only chunks overlapping them are sent for extraction:

    navigator = lx.factory.create_model_from_id("gemini-3.5-flash")
    sections = lx.document_index.select_sections(index, prompt, navigator)
    result = lx.extract(
        text,
        prompt_description=prompt,
        examples=examples,
        chunk_filter=lx.document_index.section_chunk_filter(sections),
    )

Section intervals are character offsets into the original text, so every
extraction keeps its exact source grounding either way.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterator, Sequence
import dataclasses
import json
import re
import textwrap
from typing import Any

from absl import logging
import more_itertools

from langextract import chunking
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions

__all__ = [
    "DocumentIndexError",
    "HeadingPattern",
    "MARKDOWN_HEADING",
    "NUMBERED_HEADING",
    "DEFAULT_HEADING_PATTERNS",
    "SectionNode",
    "DocumentIndex",
    "build_index",
    "summarize_index",
    "select_sections",
    "section_chunk_filter",
]

_PREAMBLE_TITLE = "Preamble"
_NODE_ID_WIDTH = 4


class DocumentIndexError(exceptions.LangExtractError):
  """Raised when an index cannot be summarized or navigated."""


@dataclasses.dataclass(frozen=True)
class HeadingPattern:
  """Recognizes one style of heading line.

  Attributes:
    regex: Pattern matched against a single line. It must define the named
      groups `marker` (the heading syntax, e.g. `##` or `15.3.1.`) and `title`.
    level: Maps the matched marker to a nesting depth, where 1 is outermost.
  """

  regex: re.Pattern[str]
  level: Callable[[str], int]


MARKDOWN_HEADING = HeadingPattern(
    regex=re.compile(r"^(?P<marker>#{1,6})\s+(?P<title>\S.*?)\s*#*\s*$"),
    level=len,
)

# Matches "7. TITLE", "1.2 Title" and "15.3.1.  200 OK". A bare number such
# as "2024 was" never matches, and a dotted number without a trailing dot
# must be followed by a capitalized title so that prose such as "3.14 is pi"
# is not mistaken for a heading.
NUMBERED_HEADING = HeadingPattern(
    regex=re.compile(
        r"^(?=\d+\.)(?P<marker>\d+(?:\.\d+)*(?P<dot>\.)?)\s+"
        r"(?P<title>(?(dot)\S|[^\sa-z]).*?)\s*$"
    ),
    level=lambda marker: len(marker.strip(".").split(".")),
)

DEFAULT_HEADING_PATTERNS: tuple[HeadingPattern, ...] = (
    MARKDOWN_HEADING,
    NUMBERED_HEADING,
)


@dataclasses.dataclass
class SectionNode:
  """A section of the document and the subsections nested under it.

  Attributes:
    node_id: Zero-padded pre-order position, stable for a given text.
    title: Heading text without its marker.
    level: Nesting depth; 1 is outermost.
    char_interval: Span in the source text from the start of the heading line
      to the start of the next heading at the same or a shallower level, so a
      parent's interval covers all of its descendants.
    summary: Short description set by `summarize_index`, or None.
    children: Subsections in document order.
    extractions: Extractions whose start falls in this section's own text,
      not in a subsection. Set by `DocumentIndex.attach_extractions`.
  """

  node_id: str
  title: str
  level: int
  char_interval: data.CharInterval
  summary: str | None = None
  children: list[SectionNode] = dataclasses.field(default_factory=list)
  extractions: list[data.Extraction] = dataclasses.field(
      default_factory=list, repr=False
  )

  def all_extractions(self) -> list[data.Extraction]:
    """Returns extractions in this section and its subsections, in order."""
    return [e for node in self.iter_nodes() for e in node.extractions]

  def iter_nodes(self) -> Iterator[SectionNode]:
    """Yields this node and its descendants in document order."""
    yield self
    for child in self.children:
      yield from child.iter_nodes()


@dataclasses.dataclass
class DocumentIndex:
  """Tree of sections built from a document's headings.

  Attributes:
    text: The indexed source text.
    roots: Top-level sections in document order.
    unplaced: Attached extractions without a character interval, which
      cannot be assigned to a section.
  """

  text: str
  roots: list[SectionNode]
  unplaced: list[data.Extraction] = dataclasses.field(
      default_factory=list, repr=False
  )

  def iter_nodes(self) -> Iterator[SectionNode]:
    """Yields every section in document order."""
    for root in self.roots:
      yield from root.iter_nodes()

  def find(self, node_id: str) -> SectionNode:
    """Returns the section with the given id.

    Raises:
      KeyError: If no section has that id.
    """
    for node in self.iter_nodes():
      if node.node_id == node_id:
        return node
    raise KeyError(node_id)

  def attach_extractions(
      self, document: data.AnnotatedDocument
  ) -> DocumentIndex:
    """Files each extraction of `document` under the section containing it.

    This turns a flat extraction list into a tree an agent can browse: see
    `to_toc` for an overview with counts and `section_view` for one
    section's extractions. Attaching again replaces earlier extractions.

    Args:
      document: Result of `lx.extract` over the same text as this index.

    Returns:
      This index, for chaining.

    Raises:
      DocumentIndexError: If the document's text differs from the index's.
    """
    if document.text != self.text:
      raise DocumentIndexError(
          "Extractions must come from the same text the index was built on."
      )
    for node in self.iter_nodes():
      node.extractions = []
    self.unplaced = []
    for extraction in document.extractions or []:
      interval = extraction.char_interval
      if interval is None or interval.start_pos is None:
        self.unplaced.append(extraction)
      else:
        self.section_at(interval.start_pos).extractions.append(extraction)
    return self

  def section_view(
      self, node_id: str, include_text: bool = False
  ) -> dict[str, Any]:
    """Returns one section's extractions for an agent, as plain JSON data.

    This is the drill-down step after reading `to_toc`: the agent asks for a
    section by id and receives only the extractions inside it, each with
    character offsets into the source text, plus the subsections it can
    open next.

    Args:
      node_id: Section id from `to_toc` or `to_dict`.
      include_text: Also return the section's source text.

    Raises:
      KeyError: If no section has that id.
    """
    node = self.find(node_id)
    view: dict[str, Any] = {
        "node_id": node.node_id,
        "path": self.heading_path(node),
        "start_index": node.char_interval.start_pos,
        "end_index": node.char_interval.end_pos,
    }
    if node.summary:
      view["summary"] = node.summary
    view["extractions"] = [
        _extraction_to_dict(e) for e in node.all_extractions()
    ]
    view["subsections"] = [
        {
            "node_id": child.node_id,
            "title": child.title,
            "extraction_count": len(child.all_extractions()),
        }
        for child in node.children
    ]
    if include_text:
      view["text"] = self.text[
          node.char_interval.start_pos : node.char_interval.end_pos
      ]
    return view

  def to_dict(self, include_extractions: bool = False) -> list[dict[str, Any]]:
    """Serializes the tree with PageIndex-style keys.

    `start_index` and `end_index` are character offsets into `text` rather
    than page numbers. Once extractions are attached, every node also has
    an `extraction_count` covering its subsections.

    Args:
      include_extractions: Also list each node's own extractions.
    """
    attached = self._has_extractions()
    return [
        _node_to_dict(root, attached, include_extractions)
        for root in self.roots
    ]

  def _has_extractions(self) -> bool:
    return bool(self.unplaced) or any(
        node.extractions for node in self.iter_nodes()
    )

  def section_at(self, pos: int) -> SectionNode:
    """Returns the deepest section containing character position `pos`.

    Useful after extraction to attach each extraction's heading context from
    its `char_interval`, with no model calls.

    Raises:
      IndexError: If `pos` is outside the indexed text.
    """
    node = next((root for root in self.roots if _contains(root, pos)), None)
    if node is None:
      raise IndexError(f"Position {pos} is outside the indexed text.")
    while True:
      child = next((c for c in node.children if _contains(c, pos)), None)
      if child is None:
        return node
      node = child

  def heading_path(self, node: SectionNode) -> list[str]:
    """Returns the titles from the top-level section down to `node`.

    Raises:
      KeyError: If `node` does not belong to this index.
    """
    for root in self.roots:
      path = _path_to(root, node)
      if path:
        return [section.title for section in path]
    raise KeyError(f"Section {node.node_id!r} is not part of this index.")

  def to_toc(
      self, include_summaries: bool = True, hide_empty: bool = False
  ) -> str:
    """Renders an indented table of contents, one section per line.

    Once extractions are attached, each line that contains any also shows
    counts per extraction class, including subsections, for example
    `0180 | Status Codes [46 status_code]`. An agent can read this overview
    first and open only the sections it needs with `section_view`.

    Args:
      include_summaries: Append each section's summary when set.
      hide_empty: Leave out sections with no extractions in their subtree,
        which keeps the overview short on long documents.
    """
    lines = []
    for node in self.iter_nodes():
      counts = collections.Counter(
          e.extraction_class for e in node.all_extractions()
      )
      if hide_empty and not counts:
        continue
      line = f"{'  ' * (node.level - 1)}{node.node_id} | {node.title}"
      if counts:
        line += (
            " ["
            + ", ".join(f"{n} {cls}" for cls, n in sorted(counts.items()))
            + "]"
        )
      if include_summaries and node.summary:
        line += f" — {node.summary}"
      lines.append(line)
    return "\n".join(lines)


def _contains(node: SectionNode, pos: int) -> bool:
  return node.char_interval.start_pos <= pos < node.char_interval.end_pos


def _path_to(current: SectionNode, target: SectionNode) -> list[SectionNode]:
  if current is target:
    return [current]
  for child in current.children:
    path = _path_to(child, target)
    if path:
      return [current] + path
  return []


def _extraction_to_dict(extraction: data.Extraction) -> dict[str, Any]:
  return {
      "extraction_class": extraction.extraction_class,
      "extraction_text": extraction.extraction_text,
      "attributes": extraction.attributes or {},
      "start_index": extraction.char_interval.start_pos,
      "end_index": extraction.char_interval.end_pos,
  }


def _node_to_dict(
    node: SectionNode, attached: bool, include_extractions: bool
) -> dict[str, Any]:
  result: dict[str, Any] = {
      "title": node.title,
      "node_id": node.node_id,
      "start_index": node.char_interval.start_pos,
      "end_index": node.char_interval.end_pos,
  }
  if node.summary is not None:
    result["summary"] = node.summary
  if attached:
    result["extraction_count"] = len(node.all_extractions())
  if include_extractions and node.extractions:
    result["extractions"] = [_extraction_to_dict(e) for e in node.extractions]
  if node.children:
    result["nodes"] = [
        _node_to_dict(child, attached, include_extractions)
        for child in node.children
    ]
  return result


@dataclasses.dataclass
class _Heading:
  title: str
  level: int
  start_pos: int


def _find_headings(
    text: str,
    heading_patterns: Sequence[HeadingPattern],
    max_heading_chars: int,
) -> list[_Heading]:
  """Returns headings of the first style in `heading_patterns` found in text.

  Styles are not mixed: a Markdown document's numbered list items would
  otherwise become level-1 sections that cut across the `#` hierarchy.
  """
  by_pattern: dict[int, list[_Heading]] = {
      i: [] for i in range(len(heading_patterns))
  }
  in_code_fence = False
  offset = 0
  for line in text.splitlines(keepends=True):
    stripped = line.strip()
    if stripped.startswith("```"):
      in_code_fence = not in_code_fence
    elif not in_code_fence and len(stripped) <= max_heading_chars:
      for i, pattern in enumerate(heading_patterns):
        match = pattern.regex.match(line.rstrip("\r\n"))
        if match:
          by_pattern[i].append(
              _Heading(
                  title=match.group("title"),
                  level=pattern.level(match.group("marker")),
                  start_pos=offset,
              )
          )
    offset += len(line)
  return next((h for h in by_pattern.values() if h), [])


def build_index(
    text: str,
    heading_patterns: Sequence[HeadingPattern] = DEFAULT_HEADING_PATTERNS,
    max_heading_chars: int = 120,
) -> DocumentIndex:
  """Builds a section tree from heading lines, without any model calls.

  Args:
    text: Source document. Headings must start at column 0; lines inside
      ``` code fences are ignored.
    heading_patterns: Heading styles to try, in order of preference. The
      first style that matches anywhere in the text is used for the whole
      document. Defaults to Markdown `#` headings, then dotted numbered
      headings such as `15.3.1.` for plain-text specifications and
      regulations.
    max_heading_chars: Lines longer than this are never headings, which keeps
      prose that happens to start with a number out of the tree.

  Returns:
    A DocumentIndex whose root intervals partition the whole text. Text before
    the first heading becomes a root titled "Preamble"; a text with no
    headings is a single such root.
  """
  headings = _find_headings(text, heading_patterns, max_heading_chars)

  roots: list[SectionNode] = []
  if not headings or headings[0].start_pos > 0:
    preamble_end = headings[0].start_pos if headings else len(text)
    roots.append(
        SectionNode(
            node_id="",
            title=_PREAMBLE_TITLE,
            level=1,
            char_interval=data.CharInterval(start_pos=0, end_pos=preamble_end),
        )
    )

  # A section runs until the next heading at the same or a shallower level,
  # so parents cover their descendants. Open sections live on the stack and
  # are closed, with their end position set, when such a heading arrives.
  stack: list[SectionNode] = []
  for heading in headings:
    node = SectionNode(
        node_id="",
        title=heading.title,
        level=heading.level,
        char_interval=data.CharInterval(
            start_pos=heading.start_pos, end_pos=len(text)
        ),
    )
    while stack and stack[-1].level >= heading.level:
      stack.pop().char_interval.end_pos = heading.start_pos
    if stack:
      stack[-1].children.append(node)
    else:
      roots.append(node)
    stack.append(node)

  index = DocumentIndex(text=text, roots=roots)
  for position, node in enumerate(index.iter_nodes(), start=1):
    node.node_id = str(position).zfill(_NODE_ID_WIDTH)
  return index


def _own_text(index: DocumentIndex, node: SectionNode) -> str:
  """Returns the node's text excluding its subsections."""
  end_pos = (
      node.children[0].char_interval.start_pos
      if node.children
      else node.char_interval.end_pos
  )
  return index.text[node.char_interval.start_pos : end_pos]


def _require_plain_model(model: base_model.BaseLanguageModel) -> None:
  if getattr(model, "schema", None) is not None:
    raise DocumentIndexError(
        "Document navigation needs a model without an extraction schema."
        " Create one with lx.factory.create_model_from_id(model_id) instead"
        " of reusing the schema-constrained extraction model."
    )


def _infer_one(model: base_model.BaseLanguageModel, prompt: str) -> str:
  return _infer_many(model, [prompt])[0]


def _infer_many(
    model: base_model.BaseLanguageModel, prompts: Sequence[str]
) -> list[str]:
  outputs = []
  for scored in model.infer(batch_prompts=prompts):
    scored = list(scored)
    if not scored or scored[0].output is None:
      raise DocumentIndexError("Language model returned no output.")
    outputs.append(scored[0].output)
  return outputs


_SUMMARY_PROMPT = textwrap.dedent("""\
    Describe in one or two sentences what information the following section
    of a document contains. Reply with the description only.

    Section:
    {section}
    """)


def summarize_index(
    index: DocumentIndex,
    model: base_model.BaseLanguageModel,
    *,
    max_verbatim_chars: int = 600,
    max_summary_input_chars: int = 12000,
    batch_length: int = 10,
) -> DocumentIndex:
  """Sets a summary on every section so a navigator can judge relevance.

  Mirrors PageIndex: short sections are described by their own text, longer
  ones by one language model call each. Only a section's own text (excluding
  its subsections) is summarized, since subsections carry their own summary.

  Args:
    index: Index to annotate in place.
    model: Model used for sections longer than `max_verbatim_chars`. It must
      not have an extraction schema applied.
    max_verbatim_chars: Sections up to this length use their whitespace-
      collapsed text as the summary, costing no model calls.
    max_summary_input_chars: Longer sections are truncated to this many
      characters before being summarized.
    batch_length: Number of summary prompts per `model.infer` call.

  Returns:
    The same index, for chaining.

  Raises:
    DocumentIndexError: If the model has a schema applied or returns no output.
  """
  _require_plain_model(model)

  pending: list[tuple[SectionNode, str]] = []
  for node in index.iter_nodes():
    own_text = _own_text(index, node)
    if node.title != _PREAMBLE_TITLE:
      # The heading line is already shown as the node's title.
      own_text = own_text.partition("\n")[2]
    collapsed = re.sub(r"\s+", " ", own_text).strip()
    if len(collapsed) <= max_verbatim_chars:
      node.summary = collapsed
    else:
      pending.append((node, own_text[:max_summary_input_chars]))

  for batch in more_itertools.batched(pending, batch_length):
    prompts = [_SUMMARY_PROMPT.format(section=text) for _, text in batch]
    for (node, _), output in zip(batch, _infer_many(model, prompts)):
      node.summary = output.strip()
  return index


_SELECTION_PROMPT = textwrap.dedent("""\
    You are planning an information extraction task over a long document.
    Below is the document's table of contents. Each line gives a section id,
    its title and, when available, a short description. Selecting a section
    also covers everything nested under it.

    Extraction task:
    {task}
    {classes}
    Table of contents:
    {toc}

    Select every section that may contain information for the task. Prefer
    including a section when unsure; sections you leave out are never read.
    Respond with JSON only, in this exact form:
    {{"node_ids": ["<id>", ...], "reasoning": "<one sentence>"}}
    """)


def select_sections(
    index: DocumentIndex,
    prompt_description: str,
    model: base_model.BaseLanguageModel,
    *,
    examples: Sequence[data.ExampleData] | None = None,
) -> list[SectionNode]:
  """Asks a model which sections are relevant to an extraction task.

  This is the reasoning step of PageIndex-style retrieval: the model sees the
  table of contents, not the body, and decides where to look.

  Args:
    index: Index to navigate, ideally after `summarize_index`.
    prompt_description: The extraction task, as passed to `lx.extract`.
    model: Model used for the single navigation call. It must not have an
      extraction schema applied.
    examples: Optional few-shot examples; their extraction classes are listed
      in the prompt to sharpen the selection.

  Returns:
    Selected sections in document order, without duplicates. An empty list
    means the model found nothing relevant; a warning is logged because the
    resulting extraction would be empty.

  Raises:
    DocumentIndexError: If the model has a schema applied or its reply is not
      the expected JSON object.
  """
  _require_plain_model(model)

  classes = ""
  if examples:
    class_names = sorted({
        extraction.extraction_class
        for example in examples
        for extraction in example.extractions
    })
    if class_names:
      classes = f"\nExtraction classes: {', '.join(class_names)}\n"

  prompt = _SELECTION_PROMPT.format(
      task=prompt_description, classes=classes, toc=index.to_toc()
  )
  output = _infer_one(model, prompt)
  node_ids = _parse_node_ids(output)

  requested = set(node_ids)
  selected = [node for node in index.iter_nodes() if node.node_id in requested]
  unknown = requested - {node.node_id for node in selected}
  if unknown:
    logging.warning(
        "Section selection returned unknown node ids, ignoring: %s",
        sorted(unknown),
    )
  if not selected:
    logging.warning(
        "Section selection chose no sections for task %r; extraction with"
        " this selection will produce no extractions.",
        prompt_description,
    )
  return selected


def _parse_node_ids(output: str) -> list[str]:
  """Reads the `node_ids` list from a reply that may carry fences or prose."""
  start, end = output.find("{"), output.rfind("}")
  if start == -1 or end == -1:
    raise DocumentIndexError(
        f"Section selection reply contains no JSON object: {output!r}"
    )
  try:
    parsed = json.loads(output[start : end + 1])
  except json.JSONDecodeError as e:
    raise DocumentIndexError(
        f"Section selection reply is not valid JSON: {output!r}"
    ) from e
  if not isinstance(parsed, dict) or not isinstance(
      parsed.get("node_ids"), list
  ):
    raise DocumentIndexError(
        "Section selection reply must be a JSON object with a 'node_ids'"
        f" list: {output!r}"
    )
  node_ids = []
  for node_id in parsed["node_ids"]:
    node_id = str(node_id).strip()
    if node_id.isdigit():
      node_id = node_id.zfill(_NODE_ID_WIDTH)
    node_ids.append(node_id)
  return node_ids


def section_chunk_filter(
    sections: Sequence[SectionNode],
) -> Callable[[chunking.TextChunk], bool]:
  """Returns a `chunk_filter` for `lx.extract` keeping only selected text.

  A chunk is kept when its character interval overlaps any selected section,
  so chunks straddling a section boundary are still processed.
  """
  intervals = [
      (node.char_interval.start_pos, node.char_interval.end_pos)
      for node in sections
  ]

  def keep(chunk: chunking.TextChunk) -> bool:
    chunk_start = chunk.char_interval.start_pos
    chunk_end = chunk.char_interval.end_pos
    return any(
        chunk_start < end and start < chunk_end for start, end in intervals
    )

  return keep
