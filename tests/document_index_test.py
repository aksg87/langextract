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

from absl.testing import absltest
from absl.testing import parameterized

from langextract import document_index
import langextract as lx
from langextract.core import data


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
    e1 = _extraction("character", "Friar Lawrence", 0, 14, household="Church")
    e2 = _extraction("character", "friar\n  lawrence", 20, 36, role="confessor")
    e3 = _extraction("character", "Romeo", 40, 45, household="Montague")

    index = document_index.build_index(
        [e1, e2, e3], levels=document_index.by_text
    )

    self.assertEqual([r.node_id for r in index.roots], ["0001", "0002"])
    self.assertEqual(
        [r.title for r in index.roots], ["Friar Lawrence", "Romeo"]
    )
    self.assertEqual(index.roots[0].extractions, [e1, e2])
    self.assertIn("household: Church", index.roots[0].summary)
    self.assertIn("role: confessor", index.roots[0].summary)

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
