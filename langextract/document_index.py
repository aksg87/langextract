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
downstream agent has to scan in full. This module builds a tree from the
extractions upward, so the reader can scan a compact table of contents first
and then open only the nodes it needs.

The extractions are the leaves. Each level above groups the nodes of the level
below, either with a key function such as `by_text`, `by_class` or
`by_attribute`, or with a language model that follows a natural-language rule:

    index = lx.document_index.build_index(
        result,
        levels=[
            lx.document_index.by_text,
            "Group the status codes by what the client should do next.",
            "Group these groups into the five HTTP status classes.",
        ],
        model_id="gemini-3.5-flash",
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
import json
import re
import textwrap
from typing import Any

from absl import logging

from langextract import factory
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions

__all__ = [
    "DocumentIndexError",
    "LevelSpec",
    "IndexNode",
    "DocumentIndex",
    "by_text",
    "by_class",
    "by_attribute",
    "build_index",
]

_NODE_ID_WIDTH = 4
_MAX_SUMMARY_CHARS = 200
_DEFAULT_ROLLUP_INSTRUCTION = (
    "Group related nodes into broader parent categories."
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class DocumentIndexError(exceptions.LangExtractError):
  """Raised when a model-organized index level cannot be built."""


KeyFn = Callable[[data.Extraction], str]


@dataclasses.dataclass(frozen=True)
class LevelSpec:
  """A natural-language rule for one model-organized level of the index.

  Attributes:
    instruction: How the model should group the nodes at this level, for
      example "Group the characters by household".
    max_groups: Most groups the model should form per batch. Defaults to
      `build_index`'s `max_groups`.
  """

  instruction: str
  max_groups: int | None = None


LevelRule = KeyFn | str | LevelSpec


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
    title: Group name from the level's key function, or written by the model.
    summary: Short description. The model writes it for natural-language
      levels; otherwise it lists the distinct attribute values of a leaf's
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


_CLUSTER_PROMPT = textwrap.dedent("""\
    You are building one level of a hierarchical tree index over structured
    extractions from a document.
    Group the items below into at most {max_groups} coherent clusters following
    the organization rule. Every item id from 0 to {last_id} must appear in
    exactly one group.

    Organization rule:
    {instruction}

    Items:
    {items}

    Respond with JSON only, in this exact form:
    {{"groups": [{{"title": "<short name>", "summary": "<one sentence describing what the group covers>", "item_ids": [0, ...]}}]}}
    """)


def _resolve_model(
    model: base_model.BaseLanguageModel | None,
    model_id: str | None = None,
    api_key: str | None = None,
) -> base_model.BaseLanguageModel:
  """Returns a plain language model for natural-language index levels."""
  if model is None and model_id is not None:
    kwargs = {"api_key": api_key} if api_key is not None else {}
    model = factory.create_model_from_id(model_id, **kwargs)
  if model is None:
    raise DocumentIndexError(
        "A language model or model_id is required for natural-language index"
        " levels."
    )
  if getattr(model, "schema", None) is not None:
    raise DocumentIndexError(
        "Document indexing needs a model without an extraction schema."
        " Pass model_id=... or create one with"
        " lx.factory.create_model_from_id(model_id) instead of reusing the"
        " schema-constrained extraction model."
    )
  return model


def _infer_many(
    model: base_model.BaseLanguageModel, prompts: Sequence[str]
) -> list[str]:
  outputs = []
  for scored in model.infer(batch_prompts=prompts):
    scored = list(scored)
    if not scored or scored[0].output is None:
      raise DocumentIndexError("Language model returned no output.")
    outputs.append(scored[0].output)
  if len(outputs) != len(prompts):
    raise DocumentIndexError(
        f"Language model returned {len(outputs)} outputs for"
        f" {len(prompts)} prompts."
    )
  return outputs


def _format_node_item(index: int, node: IndexNode) -> str:
  counts = _format_counts(node.all_extractions())
  count_part = f" [{counts}]" if counts else ""
  summary_part = f" — {node.summary}" if node.summary else ""
  return f"[{index}] {node.title}{count_part}{summary_part}"


def _cluster_prompt(
    batch: Sequence[IndexNode], instruction: str, max_groups: int
) -> str:
  return _CLUSTER_PROMPT.format(
      max_groups=max_groups,
      last_id=len(batch) - 1,
      instruction=instruction,
      items="\n".join(_format_node_item(i, n) for i, n in enumerate(batch)),
  )


def _parse_item_id(raw_id: Any) -> int | None:
  if isinstance(raw_id, bool):
    return None
  if isinstance(raw_id, int):
    return raw_id
  text = str(raw_id).strip().strip("[]")
  if text.isdigit():
    return int(text)
  return None


def _load_group_entries(output: str) -> list[dict[str, Any]]:
  """Returns the `groups` entries from a clustering reply's JSON object."""
  fenced = _FENCED_JSON_RE.search(output)
  if fenced:
    candidate = fenced.group(1)
  else:
    start, end = output.find("{"), output.rfind("}")
    if start == -1 or end == -1 or start >= end:
      raise DocumentIndexError(
          f"Clustering reply contains no JSON object: {output!r}"
      )
    candidate = output[start : end + 1]
  try:
    parsed = json.loads(candidate)
  except json.JSONDecodeError as e:
    raise DocumentIndexError(
        f"Clustering reply is not valid JSON: {output!r}"
    ) from e
  entries = parsed.get("groups") if isinstance(parsed, dict) else None
  if not isinstance(entries, list) or not all(
      isinstance(e, dict) and isinstance(e.get("item_ids"), list)
      for e in entries
  ):
    raise DocumentIndexError(
        "Clustering reply must be a JSON object with a 'groups' list whose"
        f" entries each have an 'item_ids' list: {output!r}"
    )
  return entries


def _parse_groups(
    output: str, num_items: int
) -> list[tuple[str, str | None, list[int]]]:
  """Parses a clustering reply and places every item id in exactly one group.

  Repeated and unknown ids are dropped. Ids that the reply leaves out join
  the reply's "Other" group, or a new one at the end, so no item is lost.
  """
  assigned: set[int] = set()
  unknown: list[Any] = []
  groups: list[tuple[str, str | None, list[int]]] = []
  for entry in _load_group_entries(output):
    item_ids: list[int] = []
    for raw_id in entry["item_ids"]:
      idx = _parse_item_id(raw_id)
      if idx is None or not 0 <= idx < num_items:
        unknown.append(raw_id)
      elif idx not in assigned:
        assigned.add(idx)
        item_ids.append(idx)
    if item_ids:
      title = _clean_line(str(entry.get("title") or "")) or "Untitled"
      summary = _clean_line(str(entry.get("summary") or "")) or None
      groups.append((title, summary, item_ids))

  if unknown:
    logging.warning(
        "Clustering reply contained unknown item ids, ignoring: %s", unknown
    )
  if not groups:
    raise DocumentIndexError(
        f"Clustering reply assigned no valid item ids: {output!r}"
    )
  missing = [i for i in range(num_items) if i not in assigned]
  if missing:
    logging.warning(
        "Clustering reply omitted %d of %d items; placing them under 'Other'.",
        len(missing),
        num_items,
    )
    other = next((g for g in groups if g[0].casefold() == "other"), None)
    if other is None:
      groups.append(("Other", None, missing))
    else:
      other[2].extend(missing)
  return groups


def _make_parent(
    title: str, summary: str | None, children: list[IndexNode]
) -> IndexNode:
  if len(children) == 1 and title.casefold() == children[0].title.casefold():
    return children[0]
  return IndexNode(
      node_id="",
      title=title,
      summary=summary or _summarize_child_nodes(children),
      children=children,
  )


def _cluster_nodes_with_llm(
    nodes: Sequence[IndexNode],
    model: base_model.BaseLanguageModel,
    instruction: str,
    max_groups: int,
    batch_size: int,
) -> list[IndexNode]:
  """Asks `model` to group `nodes` into parents, one prompt per batch."""
  batches = [
      list(nodes[i : i + batch_size]) for i in range(0, len(nodes), batch_size)
  ]
  prompts = [
      _cluster_prompt(batch, instruction, max_groups)
      for batch in batches
      if len(batch) > 1
  ]
  replies = iter(_infer_many(model, prompts) if prompts else [])
  parents: list[IndexNode] = []
  for batch in batches:
    if len(batch) == 1:
      parents.extend(batch)
      continue
    for title, summary, item_ids in _parse_groups(next(replies), len(batch)):
      parents.append(_make_parent(title, summary, [batch[i] for i in item_ids]))
  return parents


def _apply_node_rule(
    nodes: Sequence[IndexNode],
    rule: LevelRule,
    model: base_model.BaseLanguageModel | None,
    default_max_groups: int,
    batch_size: int,
) -> list[IndexNode]:
  if callable(rule):
    return _group_nodes_by_key(nodes, rule)
  if len(nodes) <= 1:
    return list(nodes)
  spec = rule if isinstance(rule, LevelSpec) else LevelSpec(rule)
  return _cluster_nodes_with_llm(
      nodes,
      _resolve_model(model),
      spec.instruction.strip(),
      spec.max_groups or default_max_groups,
      batch_size,
  )


def _roll_up(
    nodes: list[IndexNode],
    model: base_model.BaseLanguageModel | None,
    rules: Sequence[LevelRule],
    max_roots: int,
    default_max_groups: int,
    batch_size: int,
) -> list[IndexNode]:
  """Reapplies the last natural-language rule until `max_roots` is met."""
  if len(nodes) <= max_roots:
    return nodes
  resolved = _resolve_model(model)
  rule = next(
      (r for r in reversed(rules) if not callable(r)),
      _DEFAULT_ROLLUP_INSTRUCTION,
  )
  spec = rule if isinstance(rule, LevelSpec) else LevelSpec(rule)
  step_max = spec.max_groups or default_max_groups
  instruction = spec.instruction.strip()
  while len(nodes) > max_roots:
    limit = min(step_max, max_roots) if len(nodes) <= batch_size else step_max
    merged = _cluster_nodes_with_llm(
        nodes, resolved, instruction, limit, batch_size
    )
    if len(merged) >= len(nodes):
      break
    nodes = merged
  return nodes


def _check_levels(levels: KeyFn | Sequence[LevelRule]) -> list[LevelRule]:
  if (
      callable(levels)
      or isinstance(levels, (str, LevelSpec))
      or not isinstance(levels, Sequence)
  ):
    rules = [levels]
  else:
    rules = list(levels)
  if not rules:
    raise ValueError("levels must not be empty.")
  if not callable(rules[0]):
    raise TypeError(
        "The first level must be a key function that groups extractions into"
        f" leaves, such as by_text; got {type(rules[0]).__name__}."
    )
  for rule in rules[1:]:
    if not (callable(rule) or isinstance(rule, (str, LevelSpec))):
      raise TypeError(
          "Each level must be a key function, a string or a LevelSpec; got"
          f" {type(rule).__name__}."
      )
    if isinstance(rule, str) and not rule.strip():
      raise ValueError("Level instruction must not be empty.")
    if isinstance(rule, LevelSpec):
      if not rule.instruction.strip():
        raise ValueError("LevelSpec.instruction must not be empty.")
      if rule.max_groups is not None and rule.max_groups < 2:
        raise ValueError("LevelSpec.max_groups must be at least 2.")
  return rules


def build_index(
    source: data.AnnotatedDocument | Sequence[data.Extraction],
    *,
    levels: KeyFn | Sequence[LevelRule],
    model: base_model.BaseLanguageModel | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    max_groups: int = 8,
    max_roots: int | None = None,
    batch_size: int = 100,
) -> DocumentIndex:
  """Builds a hierarchical index bottom-up from a document's extractions.

  The first level must be a key function; it groups the extractions into leaf
  nodes. Each later level groups the nodes of the level below into parents:

  - A key function is called on each node's first extraction in tree order,
    so every extraction under a node should share the parent-level key. Group
    names are compared case-insensitively, and an empty name becomes "Other".
  - A string or `LevelSpec` is a rule for a language model. The model sees
    each node's title, extraction counts and summary, in batches of up to
    `batch_size` nodes, and returns titled groups with a summary. Groups from
    different batches are not merged. Nodes that a reply leaves out go to its
    "Other" group, or to a new one, so no extraction is lost.

  Args:
    source: An `AnnotatedDocument` returned by `lx.extract`, or a sequence of
      `Extraction` objects.
    levels: One key function, or a sequence of levels ordered from the leaves
      up to the roots. The first level must be a key function; later levels
      are key functions, strings or `LevelSpec`s.
    model: Pre-created language model for natural-language levels. It must not
      have an extraction schema. If omitted, `model_id` is used to create one.
      Neither is needed when every level is a key function and `max_roots` is
      not exceeded.
    model_id: Model identifier (such as `"gemini-3.5-flash"`) used to create a
      plain language model when `model` is not provided.
    api_key: Optional API key forwarded when creating a model from `model_id`.
    max_groups: Most groups the model should form per batch, unless a
      `LevelSpec` sets its own `max_groups`. Must be at least 2.
    max_roots: If set and the top level has more nodes than this, the last
      natural-language rule, or a generic rule if there is none, is applied
      again until the top level has at most `max_roots` nodes or stops
      shrinking.
    batch_size: Most nodes sent to the model in one prompt. Must be at least 2.

  Returns:
    A `DocumentIndex` with zero-padded `node_id`s assigned in pre-order.

  Raises:
    TypeError: If the first level is not a key function, or a later level is
      not a key function, string or `LevelSpec`.
    ValueError: If `levels` or a level's instruction is empty, or a numeric
      limit is too small.
    DocumentIndexError: If a model is needed but neither `model` nor
      `model_id` is provided, if `model` has an extraction schema, or if a
      model reply cannot be parsed.
  """
  if max_groups < 2:
    raise ValueError("max_groups must be at least 2.")
  if max_roots is not None and max_roots < 1:
    raise ValueError("max_roots must be at least 1.")
  if batch_size < 2:
    raise ValueError("batch_size must be at least 2.")
  rules = _check_levels(levels)
  if any(not callable(r) for r in rules[1:]):
    model = _resolve_model(model, model_id, api_key)

  if isinstance(source, data.AnnotatedDocument):
    extractions = list(source.extractions or [])
    text = source.text
  else:
    extractions = list(source)
    text = None

  nodes = _group_extractions_by_key(extractions, rules[0])
  for rule in rules[1:]:
    nodes = _apply_node_rule(nodes, rule, model, max_groups, batch_size)
  if max_roots is not None and len(nodes) > max_roots:
    model = _resolve_model(model, model_id, api_key)
    nodes = _roll_up(nodes, model, rules, max_roots, max_groups, batch_size)

  index = DocumentIndex(roots=nodes, text=text)
  for position, node in enumerate(index.iter_nodes(), start=1):
    node.node_id = str(position).zfill(_NODE_ID_WIDTH)
  return index
