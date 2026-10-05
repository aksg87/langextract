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

"""Hierarchical index over a document's extractions.

`lx.extract` returns a flat list of grounded extractions. On a long document
that list can run to hundreds or thousands of items, which a reader or a
downstream agent has to scan in full. This module groups the extractions from
the leaves upward into a tree, so the reader can scan a compact table of
contents first and then open only the nodes it needs.

Each level of the tree is defined by a key function that maps an extraction to
a group name. `by_text`, `by_class` and `by_attribute` cover common cases, and
any callable with the same signature works:

    index = lx.document_index.build_index(
        result,
        levels=[lx.document_index.by_text, lx.document_index.by_class],
    )
    print(index.to_toc())
    index.node_view("0002", context_chars=80)

Every leaf keeps the original `lx.data.Extraction` objects, including their
character intervals in the source text.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterator, Sequence
import dataclasses
from typing import Any

from langextract.core import data

__all__ = [
    "IndexNode",
    "DocumentIndex",
    "by_text",
    "by_class",
    "by_attribute",
    "build_index",
]

_NODE_ID_WIDTH = 4
_MAX_SUMMARY_CHARS = 200

KeyFn = Callable[[data.Extraction], str]


def _clean_line(text: str) -> str:
  return " ".join(text.split())


def by_text(extraction: data.Extraction) -> str:
  """Groups mentions that share the same normalized `extraction_text`."""
  return _clean_line(extraction.extraction_text) or _clean_line(
      extraction.extraction_class
  )


def by_class(extraction: data.Extraction) -> str:
  """Groups extractions by `extraction_class`."""
  return _clean_line(extraction.extraction_class) or "Other"


def by_attribute(name: str, default: str = "Other") -> KeyFn:
  """Returns a key function grouping extractions by `attributes[name]`."""

  def key(extraction: data.Extraction) -> str:
    attrs = extraction.attributes or {}
    value = attrs.get(name)
    if value is None:
      return default
    if isinstance(value, list):
      parts = [_clean_line(str(v)) for v in value]
      text = ", ".join(p for p in parts if p)
    else:
      text = _clean_line(str(value))
    return text or default

  return key


@dataclasses.dataclass
class IndexNode:
  """A group of extractions, or a group of child nodes, in the index.

  Leaf nodes hold the grounded `Extraction` objects in `extractions`; parent
  nodes hold child `IndexNode`s in `children`.

  Attributes:
    node_id: Zero-padded pre-order identifier, such as `"0001"`.
    title: Group name returned by the level's key function.
    summary: Short description: the distinct attribute values of a leaf's
      extractions, or the titles of a parent's children.
    children: Child nodes, in order of first appearance.
    extractions: Grounded extractions attached directly to this node.
  """

  node_id: str
  title: str
  summary: str | None = None
  children: list[IndexNode] = dataclasses.field(default_factory=list)
  extractions: list[data.Extraction] = dataclasses.field(
      default_factory=list, repr=False
  )

  def all_extractions(self) -> list[data.Extraction]:
    """Returns extractions in this node and its descendants, in tree order."""
    return [e for node in self.iter_nodes() for e in node.extractions]

  def iter_nodes(self) -> Iterator[IndexNode]:
    """Yields this node and its descendants in pre-order."""
    yield self
    for child in self.children:
      yield from child.iter_nodes()


@dataclasses.dataclass
class DocumentIndex:
  """Hierarchical tree built bottom-up from a document's extractions.

  Attributes:
    roots: Top-level nodes of the tree.
    text: Source text the extractions are grounded in, when known.
  """

  roots: list[IndexNode]
  text: str | None = dataclasses.field(default=None, repr=False)

  def iter_nodes(self) -> Iterator[IndexNode]:
    """Yields every node in pre-order."""
    for root in self.roots:
      yield from root.iter_nodes()

  def all_extractions(self) -> list[data.Extraction]:
    """Returns all extractions in the tree, in pre-order."""
    return [e for node in self.iter_nodes() for e in node.extractions]

  def find(self, node_id: str) -> IndexNode:
    """Returns the node with the given id.

    Args:
      node_id: A node id from `to_toc`, such as `"0007"`. Leading zeros may be
        omitted.

    Raises:
      KeyError: If no node has that id.
    """
    normalized = _normalize_node_id(node_id)
    for node in self.iter_nodes():
      if node.node_id in (node_id, normalized):
        return node
    raise KeyError(node_id)

  def path(self, node: IndexNode) -> list[str]:
    """Returns the titles from the top-level root down to `node`.

    Raises:
      KeyError: If `node` does not belong to this index.
    """
    for root in self.roots:
      chain = _path_to(root, node)
      if chain:
        return [item.title for item in chain]
    raise KeyError(f"Node {node.node_id!r} is not part of this index.")

  def to_toc(
      self,
      include_summaries: bool = True,
      max_depth: int | None = None,
  ) -> str:
    """Renders the tree as a table of contents indented by depth.

    Each line shows the node id, title, counts per extraction class across its
    subtree, and optional summary, for example
    `0001 | 4xx [2 status_code] — Includes 401, 404.`.

    Args:
      include_summaries: Append each node's summary when set.
      max_depth: Optional maximum depth to render, where 0 is `roots` only.
    """
    lines = []
    for node, depth in _walk(self.roots):
      if max_depth is not None and depth > max_depth:
        continue
      line = f"{'  ' * depth}{node.node_id} | {node.title}"
      counts = _format_counts(node.all_extractions())
      if counts:
        line += f" [{counts}]"
      if include_summaries and node.summary:
        line += f" — {node.summary}"
      lines.append(line)
    return "\n".join(lines)

  def node_view(
      self,
      node_id: str,
      *,
      include_descendant_extractions: bool = False,
      context_chars: int = 0,
  ) -> dict[str, Any]:
    """Returns one node's children and extractions as plain JSON data.

    Opening a parent node returns its children, so an agent can drill down one
    level at a time. Opening a leaf returns its grounded extractions with
    `start_index` and `end_index` into the source text.

    Args:
      node_id: A node id from `to_toc`.
      include_descendant_extractions: Also return every extraction under a
        parent node.
      context_chars: When positive and `text` is set, add a `context` snippet
        with up to this many characters on each side of each extraction.

    Raises:
      KeyError: If no node has that id.
    """
    node = self.find(node_id)
    extractions = (
        node.all_extractions()
        if include_descendant_extractions or not node.children
        else node.extractions
    )
    view: dict[str, Any] = {
        "node_id": node.node_id,
        "path": self.path(node),
        "extraction_count": len(node.all_extractions()),
    }
    if node.summary:
      view["summary"] = node.summary
    view["children"] = [
        _child_summary_to_dict(child) for child in node.children
    ]
    view["extractions"] = [
        _extraction_to_dict(e, self.text, context_chars) for e in extractions
    ]
    return view


def _normalize_node_id(node_id: str) -> str:
  cleaned = str(node_id).strip()
  if cleaned.isdigit():
    return cleaned.zfill(_NODE_ID_WIDTH)
  return cleaned


def _walk(
    nodes: Sequence[IndexNode], depth: int = 0
) -> Iterator[tuple[IndexNode, int]]:
  """Yields nodes and their descendants in pre-order with tree depth."""
  for node in nodes:
    yield node, depth
    yield from _walk(node.children, depth + 1)


def _path_to(current: IndexNode, target: IndexNode) -> list[IndexNode]:
  if current is target:
    return [current]
  for child in current.children:
    chain = _path_to(child, target)
    if chain:
      return [current] + chain
  return []


def _format_counts(extractions: Sequence[data.Extraction]) -> str:
  counts = collections.Counter(e.extraction_class for e in extractions)
  return ", ".join(f"{n} {cls}" for cls, n in sorted(counts.items()))


def _child_summary_to_dict(node: IndexNode) -> dict[str, Any]:
  entry: dict[str, Any] = {
      "node_id": node.node_id,
      "title": node.title,
      "extraction_count": len(node.all_extractions()),
  }
  if node.summary:
    entry["summary"] = node.summary
  return entry


def _extraction_to_dict(
    extraction: data.Extraction,
    text: str | None = None,
    context_chars: int = 0,
) -> dict[str, Any]:
  interval = extraction.char_interval
  start = interval.start_pos if interval else None
  end = interval.end_pos if interval else None
  result: dict[str, Any] = {
      "extraction_class": extraction.extraction_class,
      "extraction_text": extraction.extraction_text,
      "attributes": extraction.attributes or {},
      "start_index": start,
      "end_index": end,
  }
  if (
      context_chars > 0
      and text is not None
      and start is not None
      and end is not None
  ):
    lo = max(0, start - context_chars)
    hi = min(len(text), end + context_chars)
    result["context"] = text[lo:hi]
  return result


def _truncate(text: str, limit: int) -> str:
  collapsed = " ".join(text.split())
  if len(collapsed) <= limit:
    return collapsed
  return collapsed[: limit - 3].rstrip() + "..."


def _summarize_extractions(extractions: Sequence[data.Extraction]) -> str:
  """Synthesizes a short deterministic description from attributes or texts."""
  pairs: list[str] = []
  seen_pairs: set[str] = set()
  for extraction in extractions:
    for key, value in (extraction.attributes or {}).items():
      if value is None:
        continue
      val_str = (
          ", ".join(str(v).strip() for v in value if str(v).strip())
          if isinstance(value, list)
          else str(value).strip()
      )
      if not val_str:
        continue
      pair = f"{key}: {val_str}"
      folded = pair.casefold()
      if folded not in seen_pairs:
        seen_pairs.add(folded)
        pairs.append(pair)
  if pairs:
    return _truncate("; ".join(pairs), _MAX_SUMMARY_CHARS)

  texts: list[str] = []
  seen_texts: set[str] = set()
  for extraction in extractions:
    cleaned = extraction.extraction_text.strip()
    folded = cleaned.casefold()
    if cleaned and folded not in seen_texts:
      seen_texts.add(folded)
      texts.append(cleaned)
  return _truncate(", ".join(texts), _MAX_SUMMARY_CHARS)


def _summarize_child_nodes(children: Sequence[IndexNode]) -> str:
  titles = [c.title for c in children if c.title]
  return _truncate("Includes " + ", ".join(titles) + ".", _MAX_SUMMARY_CHARS)


def _group_extractions_by_key(
    extractions: Sequence[data.Extraction], key_fn: KeyFn
) -> list[IndexNode]:
  """Groups extractions into leaf `IndexNode`s by `key_fn`."""
  groups: dict[str, list[data.Extraction]] = collections.defaultdict(list)
  titles: dict[str, str] = {}
  for extraction in extractions:
    raw = _clean_line(str(key_fn(extraction) or "")) or "Other"
    folded = raw.casefold()
    if folded not in titles:
      titles[folded] = raw
    groups[folded].append(extraction)
  return [
      IndexNode(
          node_id="",
          title=titles[folded],
          summary=_summarize_extractions(members),
          extractions=members,
      )
      for folded, members in groups.items()
  ]


def _group_nodes_by_key(
    nodes: Sequence[IndexNode], key_fn: KeyFn
) -> list[IndexNode]:
  """Groups nodes into parents by `key_fn` of each node's first extraction."""
  groups: dict[str, list[IndexNode]] = collections.defaultdict(list)
  titles: dict[str, str] = {}
  for node in nodes:
    leaf_extractions = node.all_extractions()
    raw = (
        _clean_line(str(key_fn(leaf_extractions[0]) or ""))
        if leaf_extractions
        else ""
    ) or "Other"
    folded = raw.casefold()
    if folded not in titles:
      titles[folded] = raw
    groups[folded].append(node)
  return [
      IndexNode(
          node_id="",
          title=titles[folded],
          summary=_summarize_child_nodes(children),
          children=children,
      )
      for folded, children in groups.items()
  ]


def build_index(
    source: data.AnnotatedDocument | Sequence[data.Extraction],
    *,
    levels: KeyFn | Sequence[KeyFn],
) -> DocumentIndex:
  """Builds a hierarchical index bottom-up from a document's extractions.

  The first key function groups the extractions into leaf nodes. Each later
  key function groups the nodes of the level below into parent nodes. To place
  a node, the key function is called on the node's first extraction in tree
  order, so every extraction under a node should share the parent-level key.

  Group names are compared case-insensitively, and an empty name becomes
  `"Other"`. Groups keep the order in which their first extraction appears in
  `source`.

  Args:
    source: An `AnnotatedDocument` returned by `lx.extract`, or a sequence of
      `Extraction` objects.
    levels: One key function, or a non-empty sequence of key functions ordered
      from the leaves up to the roots. A key function maps an extraction to a
      group name; see `by_text`, `by_class` and `by_attribute`.

  Returns:
    A `DocumentIndex` with zero-padded `node_id`s assigned in pre-order.

  Raises:
    TypeError: If a level is not callable.
    ValueError: If `levels` is empty.
  """
  if callable(levels) or not isinstance(levels, Sequence):
    key_fns = [levels]
  else:
    key_fns = list(levels)
  if not key_fns:
    raise ValueError("levels must not be empty.")
  for key_fn in key_fns:
    if not callable(key_fn):
      raise TypeError(
          f"Each level must be a key function, got {type(key_fn).__name__}."
      )

  if isinstance(source, data.AnnotatedDocument):
    extractions = list(source.extractions or [])
    text = source.text
  else:
    extractions = list(source)
    text = None

  nodes = _group_extractions_by_key(extractions, key_fns[0])
  for key_fn in key_fns[1:]:
    nodes = _group_nodes_by_key(nodes, key_fn)

  index = DocumentIndex(roots=nodes, text=text)
  for position, node in enumerate(index.iter_nodes(), start=1):
    node.node_id = str(position).zfill(_NODE_ID_WIDTH)
  return index
