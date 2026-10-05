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

"""Tests for langextract.document_index."""

from collections.abc import Iterator, Sequence
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import document_index
import langextract as lx
from langextract.core import base_model
from langextract.core import data
from langextract.core import types


def _extraction(
    extraction_class: str,
    extraction_text: str,
    start: int | None = None,
    end: int | None = None,
    **attributes,
) -> data.Extraction:
  interval = (
      data.CharInterval(start_pos=start, end_pos=end)
      if start is not None and end is not None
      else None
  )
  return data.Extraction(
      extraction_class=extraction_class,
      extraction_text=extraction_text,
      char_interval=interval,
      attributes=attributes or None,
  )


def _reply(*groups: tuple[str, list[int]]) -> str:
  """Returns a clustering reply with one group per `(title, item_ids)`."""
  return json.dumps({
      "groups": [
          {"title": title, "summary": f"{title} summary.", "item_ids": ids}
          for title, ids in groups
      ]
  })


class FakeLanguageModel(base_model.BaseLanguageModel):
  """Records prompts and returns canned replies in order."""

  def __init__(self, replies: Sequence[str], schema=None):
    super().__init__()
    self._replies = list(replies)
    self.prompts: list[str] = []
    self.apply_schema(schema)

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[types.ScoredOutput]]:
    for prompt in batch_prompts:
      self.prompts.append(prompt)
      reply = self._replies.pop(0) if self._replies else ""
      yield [types.ScoredOutput(score=1.0, output=reply)]


class BuildIndexTest(parameterized.TestCase):

  def test_lazy_import_from_top_level_package(self):
    self.assertIs(lx.document_index, document_index)

  def test_empty_source_returns_empty_index(self):
    doc = data.AnnotatedDocument(text="Nothing here.", extractions=[])
    index = document_index.build_index(doc, levels=document_index.by_text)
    self.assertEmpty(index.roots)
    self.assertEmpty(index.all_extractions())
    self.assertEqual(index.text, "Nothing here.")
    self.assertEqual(index.to_toc(), "")

  def test_by_text_groups_mentions_case_insensitively_and_keeps_spans(self):
    e1 = _extraction("character", "Juliet", 0, 6, household="Capulet")
    e2 = _extraction("character", "juliet", 20, 26, role="protagonist")
    e3 = _extraction("character", "Romeo", 40, 45, household="Montague")
    e4 = _extraction("character", "friar\n  lawrence", 60, 76)
    e5 = _extraction("character", "Friar Lawrence", 80, 94)

    index = document_index.build_index(
        [e1, e2, e3, e4, e5], levels=document_index.by_text
    )

    self.assertEqual([r.node_id for r in index.roots], ["0001", "0002", "0003"])
    self.assertEqual(
        [r.title for r in index.roots], ["Juliet", "Romeo", "friar lawrence"]
    )
    self.assertEqual(index.roots[0].extractions, [e1, e2])
    self.assertEqual(index.roots[2].extractions, [e4, e5])
    self.assertIn("household: Capulet", index.roots[0].summary)
    self.assertIn("role: protagonist", index.roots[0].summary)

  def test_key_functions_build_multi_level_tree(self):
    extractions = [
        _extraction(
            "status_code", "200", 10, 13, family="2xx", reason_phrase="OK"
        ),
        _extraction(
            "status_code", "200", 50, 53, family="2xx", reason_phrase="OK"
        ),
        _extraction(
            "status_code", "201", 80, 83, family="2xx", reason_phrase="Created"
        ),
        _extraction(
            "status_code",
            "404",
            120,
            123,
            family="4xx",
            reason_phrase="Not Found",
        ),
        _extraction("status_code", "499", 150, 153),
    ]

    index = document_index.build_index(
        extractions,
        levels=[
            document_index.by_text,
            document_index.by_attribute("family", default="Unclassified"),
            document_index.by_class,
        ],
    )

    self.assertLen(index.roots, 1)
    root = index.roots[0]
    self.assertEqual(root.title, "status_code")
    self.assertEqual(
        [c.title for c in root.children], ["2xx", "4xx", "Unclassified"]
    )
    self.assertEqual(
        [leaf.title for leaf in root.children[0].children], ["200", "201"]
    )
    self.assertEqual(index.all_extractions(), extractions)

  def test_by_attribute_joins_list_values(self):
    e = _extraction("entity", "aspirin", tags=["nsaid", "analgesic"])
    key_fn = document_index.by_attribute("tags")
    self.assertEqual(key_fn(e), "nsaid, analgesic")

  def test_groups_follow_first_appearance_and_rebuild_identically(self):
    extractions = [
        _extraction("character", "Romeo", 0, 5),
        _extraction("character", "Juliet", 10, 16),
        _extraction("character", "romeo", 20, 25),
    ]

    first = document_index.build_index(
        extractions, levels=document_index.by_text
    )
    second = document_index.build_index(
        extractions, levels=document_index.by_text
    )

    self.assertEqual([r.title for r in first.roots], ["Romeo", "Juliet"])
    self.assertEqual([r.node_id for r in first.roots], ["0001", "0002"])
    self.assertEqual(first, second)

  def test_parent_level_keys_each_node_by_its_first_extraction(self):
    character = _extraction("character", "Romeo", 0, 5)
    emotion = _extraction("emotion", "romeo", 10, 15)

    index = document_index.build_index(
        [character, emotion],
        levels=[document_index.by_text, document_index.by_class],
    )

    self.assertEqual([r.title for r in index.roots], ["character"])
    self.assertEqual(
        index.roots[0].children[0].extractions, [character, emotion]
    )

  @parameterized.named_parameters(
      {"testcase_name": "empty", "key": ""},
      {"testcase_name": "whitespace", "key": "  "},
  )
  def test_blank_keys_are_grouped_under_other(self, key):
    extractions = [_extraction("item", "alpha"), _extraction("item", "beta")]

    index = document_index.build_index(extractions, levels=lambda _: key)

    self.assertEqual([r.title for r in index.roots], ["Other"])
    self.assertEqual(index.all_extractions(), extractions)

  @parameterized.named_parameters(
      {"testcase_name": "none", "levels": None},
      {"testcase_name": "string", "levels": "Group by topic"},
      {
          "testcase_name": "string_first",
          "levels": ["Group by topic", document_index.by_text],
      },
      {"testcase_name": "number", "levels": [document_index.by_text, 3]},
  )
  def test_invalid_levels_raise_type_error(self, levels):
    with self.assertRaises(TypeError):
      document_index.build_index([_extraction("item", "alpha")], levels=levels)

  def test_empty_levels_raise_value_error(self):
    with self.assertRaises(ValueError):
      document_index.build_index([_extraction("item", "alpha")], levels=[])


class ModelLevelsTest(parameterized.TestCase):

  def test_rule_levels_group_nodes_from_leaves_upward(self):
    extractions = [
        _extraction("status_code", "200", 10, 13, reason="OK"),
        _extraction("status_code", "204", 20, 23, reason="No Content"),
        _extraction("status_code", "401", 30, 33, reason="Unauthorized"),
        _extraction("status_code", "403", 40, 43, reason="Forbidden"),
    ]
    model = FakeLanguageModel([
        _reply(("Success", [0, 1]), ("Auth Errors", [2, 3])),
        _reply(("HTTP Status Codes", [0, 1])),
    ])

    index = document_index.build_index(
        extractions,
        model=model,
        levels=[
            document_index.by_text,
            "Group status codes by meaning",
            "Combine into top-level categories",
        ],
    )

    self.assertLen(model.prompts, 2)
    self.assertIn("Group status codes by meaning", model.prompts[0])
    self.assertIn(
        "[2] 401 [1 status_code] — reason: Unauthorized", model.prompts[0]
    )
    self.assertIn("Combine into top-level categories", model.prompts[1])
    self.assertIn(
        "[1] Auth Errors [2 status_code] — Auth Errors summary.",
        model.prompts[1],
    )
    root = index.roots[0]
    self.assertEqual((root.node_id, root.title), ("0001", "HTTP Status Codes"))
    self.assertEqual(
        [(c.node_id, c.title) for c in root.children],
        [("0002", "Success"), ("0005", "Auth Errors")],
    )
    self.assertEqual(index.all_extractions(), extractions)

  def test_model_id_creates_model_via_factory(self):
    extractions = [_extraction("code", "200"), _extraction("code", "404")]
    model = FakeLanguageModel([_reply(("All", [0, 1]))])
    with mock.patch.object(
        document_index.factory, "create_model_from_id", return_value=model
    ) as create_mock:
      index = document_index.build_index(
          extractions,
          levels=[document_index.by_text, "Group codes"],
          model_id="gemini-3.5-flash",
          api_key="test-key",
      )

    create_mock.assert_called_once_with("gemini-3.5-flash", api_key="test-key")
    self.assertEqual([r.title for r in index.roots], ["All"])

  def test_key_function_level_can_follow_rule_levels(self):
    extractions = [
        _extraction("status_code", "401"),
        _extraction("status_code", "403"),
        _extraction("status_code", "500"),
    ]
    model = FakeLanguageModel(
        [_reply(("Auth", [0, 1]), ("Server Errors", [2]))]
    )

    index = document_index.build_index(
        extractions,
        model=model,
        levels=[
            document_index.by_text,
            "Group status codes by meaning",
            lambda e: e.extraction_text[:1] + "xx",
        ],
    )

    self.assertLen(model.prompts, 1)
    self.assertEqual([r.title for r in index.roots], ["4xx", "5xx"])
    self.assertEqual([c.title for c in index.roots[0].children], ["Auth"])

  def test_max_roots_reapplies_last_rule_until_met(self):
    extractions = [_extraction("code", str(i)) for i in range(6)]
    model = FakeLanguageModel([
        _reply(("G1", [0, 1]), ("G2", [2, 3]), ("G3", [4, 5])),
        _reply(("Top A", [0, 1]), ("Top B", [2])),
    ])

    index = document_index.build_index(
        extractions,
        model=model,
        levels=[document_index.by_text, "Group codes by range"],
        max_groups=3,
        max_roots=2,
    )

    self.assertLen(model.prompts, 2)
    self.assertIn("at most 3 coherent clusters", model.prompts[0])
    self.assertIn("Group codes by range", model.prompts[1])
    self.assertIn("at most 2 coherent clusters", model.prompts[1])
    self.assertEqual([r.title for r in index.roots], ["Top A", "Top B"])
    self.assertEqual([c.title for c in index.roots[0].children], ["G1", "G2"])
    self.assertEqual(index.all_extractions(), extractions)

  def test_max_roots_without_a_rule_uses_a_generic_rule(self):
    extractions = [_extraction("code", "1"), _extraction("code", "2")]
    model = FakeLanguageModel([_reply(("All codes", [0, 1]))])

    index = document_index.build_index(
        extractions, model=model, levels=document_index.by_text, max_roots=1
    )

    self.assertIn(
        "Group related nodes into broader parent categories.", model.prompts[0]
    )
    self.assertEqual([r.title for r in index.roots], ["All codes"])

  def test_roll_up_stops_when_a_pass_does_not_shrink(self):
    extractions = [_extraction("code", "1"), _extraction("code", "2")]
    model = FakeLanguageModel([_reply(("A", [0]), ("B", [1]))] * 2)

    index = document_index.build_index(
        extractions, model=model, levels=document_index.by_text, max_roots=1
    )

    self.assertLen(model.prompts, 1)
    self.assertEqual([r.title for r in index.roots], ["1", "2"])
    self.assertEqual(index.all_extractions(), extractions)

  def test_level_spec_sets_max_groups_and_large_levels_are_batched(self):
    extractions = [_extraction("code", str(i)) for i in range(5)]
    model = FakeLanguageModel(
        [_reply(("Part 1", [0, 1])), _reply(("Part 2", [0, 1]))]
    )

    index = document_index.build_index(
        extractions,
        model=model,
        levels=[
            document_index.by_text,
            document_index.LevelSpec("Cluster pairs", max_groups=2),
        ],
        batch_size=2,
    )

    # Batches of 2, 2 and 1 nodes; the single node needs no model call.
    self.assertLen(model.prompts, 2)
    self.assertIn("at most 2 coherent clusters", model.prompts[0])
    self.assertEqual([r.title for r in index.roots], ["Part 1", "Part 2", "4"])
    self.assertEqual(index.all_extractions(), extractions)

  def test_single_child_group_with_the_same_title_is_not_wrapped(self):
    extractions = [_extraction("code", "200"), _extraction("code", "404")]
    model = FakeLanguageModel([_reply(("200", [0]), ("Client Errors", [1]))])

    index = document_index.build_index(
        extractions, model=model, levels=[document_index.by_text, "Group"]
    )

    self.assertEqual([r.title for r in index.roots], ["200", "Client Errors"])
    self.assertEmpty(index.roots[0].children)
    self.assertLen(index.roots[1].children, 1)

  def test_omitted_repeated_and_unknown_ids_are_repaired(self):
    extractions = [
        _extraction("code", "200"),
        _extraction("code", "201"),
        _extraction("code", "404"),
    ]
    reply = """Note {item_ids} format:
    ```json
    {"groups": [{"title": "Success\\n codes", "summary": " OK\\nreplies ",
                 "item_ids": ["[0]", 0, 99, "bad"]}]}
    ```
    Done {ok}."""
    model = FakeLanguageModel([reply])

    index = document_index.build_index(
        extractions, model=model, levels=[document_index.by_text, "Group"]
    )

    self.assertEqual(
        [(r.title, r.summary) for r in index.roots],
        [("Success codes", "OK replies"), ("Other", "Includes 201, 404.")],
    )
    self.assertEqual([c.title for c in index.roots[1].children], ["201", "404"])
    self.assertEqual(index.all_extractions(), extractions)

  def test_omitted_ids_join_an_existing_other_group(self):
    extractions = [
        _extraction("code", "200"),
        _extraction("code", "201"),
        _extraction("code", "404"),
    ]
    model = FakeLanguageModel([_reply(("Success", [0]), ("Other", [1]))])

    index = document_index.build_index(
        extractions, model=model, levels=[document_index.by_text, "Group"]
    )

    self.assertEqual([r.title for r in index.roots], ["Success", "Other"])
    self.assertEqual([c.title for c in index.roots[1].children], ["201", "404"])
    self.assertEqual(index.all_extractions(), extractions)

  @parameterized.named_parameters(
      ("no_json", "No braces here."),
      ("reversed_braces", "} before {"),
      ("invalid_json", "{not valid json}"),
      ("missing_groups_key", '{"node_ids": [0]}'),
      ("group_missing_item_ids", '{"groups": [{}]}'),
      ("no_valid_item_ids", '{"groups": [{"title": "X", "item_ids": [99]}]}'),
  )
  def test_malformed_reply_raises(self, reply):
    extractions = [_extraction("code", "200"), _extraction("code", "404")]
    model = FakeLanguageModel([reply])
    with self.assertRaises(document_index.DocumentIndexError):
      document_index.build_index(
          extractions, model=model, levels=[document_index.by_text, "Group"]
      )

  def test_missing_or_schema_constrained_model_raises(self):
    single = [_extraction("code", "200")]
    levels = [document_index.by_text, "Group"]
    with self.assertRaises(document_index.DocumentIndexError):
      document_index.build_index(single, levels=levels)

    constrained = FakeLanguageModel([], schema=object())
    with self.assertRaises(document_index.DocumentIndexError):
      document_index.build_index(single, model=constrained, levels=levels)

  @parameterized.named_parameters(
      {"testcase_name": "max_groups", "max_groups": 1},
      {"testcase_name": "max_roots", "max_roots": 0},
      {"testcase_name": "batch_size", "batch_size": 1},
      {
          "testcase_name": "blank_rule",
          "levels": [document_index.by_text, "   "],
      },
      {
          "testcase_name": "blank_level_spec",
          "levels": [
              document_index.by_text,
              document_index.LevelSpec("   "),
          ],
      },
      {
          "testcase_name": "level_spec_max_groups_below_min",
          "levels": [
              document_index.by_text,
              document_index.LevelSpec("Group", max_groups=1),
          ],
      },
  )
  def test_invalid_limits_or_rules_raise_value_error(
      self, levels=document_index.by_text, **limits
  ):
    with self.assertRaises(ValueError):
      document_index.build_index(
          [_extraction("code", "200")], levels=levels, **limits
      )


class DocumentIndexViewsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.e200 = _extraction("status_code", "200", 5, 8, reason="OK")
    self.e401 = _extraction("status_code", "401", 32, 35, reason="Unauthorized")
    self.index = document_index.build_index(
        data.AnnotatedDocument(
            text="HTTP 200 OK means success. HTTP 401 Unauthorized needs auth.",
            extractions=[self.e200, self.e401],
        ),
        levels=[
            document_index.by_text,
            document_index.by_class,
        ],
    )

  def test_find_and_path(self):
    leaf = self.index.find("2")
    self.assertEqual(leaf.node_id, "0002")
    self.assertEqual(leaf.extractions, [self.e200])
    self.assertEqual(self.index.path(leaf), ["status_code", "200"])

    with self.assertRaises(KeyError):
      self.index.find("9999")
    foreign = document_index.IndexNode(node_id="0001", title="Foreign")
    with self.assertRaises(KeyError):
      self.index.path(foreign)

  def test_to_toc_respects_max_depth_and_include_summaries(self):
    full_toc = self.index.to_toc()
    self.assertIn("0001 | status_code [2 status_code]", full_toc)
    self.assertIn("  0002 | 200 [1 status_code] — reason: OK", full_toc)

    shallow = self.index.to_toc(include_summaries=False, max_depth=0)
    self.assertEqual(shallow, "0001 | status_code [2 status_code]")

  def test_node_view_on_parent_and_leaf_nodes(self):
    root_view = self.index.node_view("0001")
    self.assertEqual(root_view["extraction_count"], 2)
    self.assertLen(root_view["children"], 2)
    self.assertEmpty(root_view["extractions"])

    root_with_all = self.index.node_view(
        "0001", include_descendant_extractions=True
    )
    self.assertLen(root_with_all["extractions"], 2)

    leaf_view = self.index.node_view("0002", context_chars=5)
    self.assertEqual(leaf_view["path"], ["status_code", "200"])
    self.assertEqual(leaf_view["extractions"][0]["start_index"], 5)
    self.assertEqual(leaf_view["extractions"][0]["context"], "HTTP 200 OK m")


if __name__ == "__main__":
  absltest.main()
