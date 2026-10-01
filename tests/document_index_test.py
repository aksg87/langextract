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

import textwrap
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import chunking
from langextract import document_index
import langextract as lx
from langextract.core import base_model
from langextract.core import data
from langextract.core import tokenizer
from langextract.core import types

MARKDOWN_DOC = textwrap.dedent("""\
    Intro paragraph before any heading.

    # Report

    Opening remarks.

    ## Finances

    Revenue grew 10%.

    ### Fees

    The management fee is 2%.

    ## Governance

    The board met twice.
    """)

NUMBERED_DOC = textwrap.dedent("""\
    1.  Introduction

       Scope of this specification.

    1.1.  Purpose

       Why it exists.

    15.  Status Codes

    15.3.1.  200 OK

       The 200 (OK) status code indicates success.

    16.  Extending HTTP

       Registries.
    """)


class FakeLanguageModel(base_model.BaseLanguageModel):
  """Returns canned outputs in order and records the prompts it received."""

  def __init__(self, outputs):
    super().__init__()
    self._outputs = list(outputs)
    self.prompts: list[str] = []

  def infer(self, batch_prompts, **kwargs):
    for prompt in batch_prompts:
      self.prompts.append(prompt)
      yield [types.ScoredOutput(score=1.0, output=self._outputs.pop(0))]


def _node_text(index, node):
  return index.text[node.char_interval.start_pos : node.char_interval.end_pos]


class BuildIndexTest(parameterized.TestCase):

  def test_markdown_headings_build_nested_tree(self):
    index = document_index.build_index(MARKDOWN_DOC)

    self.assertEqual([n.title for n in index.roots], ["Preamble", "Report"])
    report = index.roots[1]
    self.assertEqual(report.level, 1)
    self.assertEqual(
        [c.title for c in report.children], ["Finances", "Governance"]
    )
    finances = report.children[0]
    self.assertEqual([c.title for c in finances.children], ["Fees"])
    self.assertEqual(finances.children[0].level, 3)

  def test_char_intervals_slice_back_to_section_text(self):
    index = document_index.build_index(MARKDOWN_DOC)
    fees = index.find("0004")

    self.assertEqual(fees.title, "Fees")
    self.assertEqual(
        _node_text(index, fees), "### Fees\n\nThe management fee is 2%.\n\n"
    )
    finances = index.find("0003")
    self.assertTrue(_node_text(index, finances).startswith("## Finances"))
    self.assertIn("The management fee is 2%.", _node_text(index, finances))
    self.assertNotIn("The board met twice.", _node_text(index, finances))

  def test_intervals_partition_the_whole_text(self):
    index = document_index.build_index(MARKDOWN_DOC)
    roots = index.roots

    self.assertEqual(roots[0].char_interval.start_pos, 0)
    for prev, nxt in zip(roots, roots[1:]):
      self.assertEqual(prev.char_interval.end_pos, nxt.char_interval.start_pos)
    self.assertEqual(roots[-1].char_interval.end_pos, len(MARKDOWN_DOC))

  def test_node_ids_are_preorder_and_zero_padded(self):
    index = document_index.build_index(MARKDOWN_DOC)

    self.assertEqual(
        [(n.node_id, n.title) for n in index.iter_nodes()],
        [
            ("0001", "Preamble"),
            ("0002", "Report"),
            ("0003", "Finances"),
            ("0004", "Fees"),
            ("0005", "Governance"),
        ],
    )

  def test_numbered_headings_use_dotted_depth_as_level(self):
    index = document_index.build_index(NUMBERED_DOC)

    self.assertEqual(
        [(n.title, n.level) for n in index.iter_nodes()],
        [
            ("Introduction", 1),
            ("Purpose", 2),
            ("Status Codes", 1),
            ("200 OK", 3),
            ("Extending HTTP", 1),
        ],
    )
    status_codes = index.find("0003")
    self.assertEqual([c.title for c in status_codes.children], ["200 OK"])
    self.assertIn("indicates success", _node_text(index, status_codes))

  def test_no_preamble_when_text_starts_with_heading(self):
    index = document_index.build_index("# Only\n\nbody\n")

    self.assertLen(index.roots, 1)
    self.assertEqual(index.roots[0].title, "Only")

  def test_text_without_headings_is_a_single_root(self):
    text = "Just a paragraph. Another sentence."
    index = document_index.build_index(text)

    self.assertLen(index.roots, 1)
    self.assertEqual(index.roots[0].title, "Preamble")
    self.assertEqual(_node_text(index, index.roots[0]), text)

  def test_headings_inside_code_fences_are_ignored(self):
    text = (
        "# Real\n\n```\n# not a heading\n1. not one either\n```\n\n# Also"
        " real\n"
    )
    index = document_index.build_index(text)

    self.assertEqual(
        [n.title for n in index.iter_nodes()], ["Real", "Also real"]
    )

  @parameterized.named_parameters(
      ("long_numbered_line", "1. " + "word " * 40),
      ("indented_toc_entry", "   15.  Status Codes ........ 90"),
      ("bare_year", "2024 was a good year"),
      ("decimal_number", "3.14 is pi"),
  )
  def test_lines_that_are_not_headings(self, line):
    index = document_index.build_index("# Top\n\n" + line + "\n")

    self.assertEqual([n.title for n in index.iter_nodes()], ["Top"])

  def test_numbered_list_items_under_markdown_headings_are_not_sections(
      self,
  ):
    text = "# Setup\n\n1. Install it.\n2. Run it.\n\n## Details\n\nMore.\n"
    index = document_index.build_index(text)

    self.assertEqual(
        [n.title for n in index.iter_nodes()], ["Setup", "Details"]
    )
    self.assertEqual(
        index.heading_path(index.find("0002")), ["Setup", "Details"]
    )

  def test_crlf_line_endings(self):
    text = "# A\r\n\r\nbody\r\n## B\r\nmore"
    index = document_index.build_index(text)

    self.assertEqual([n.title for n in index.iter_nodes()], ["A", "B"])
    self.assertEqual(_node_text(index, index.find("0002")), "## B\r\nmore")

  def test_heading_on_last_line_without_newline(self):
    text = "intro\n# A"
    index = document_index.build_index(text)

    self.assertEqual(_node_text(index, index.find("0002")), "# A")
    self.assertEqual(index.roots[-1].char_interval.end_pos, len(text))

  def test_heading_path_of_foreign_node_raises_key_error(self):
    index = document_index.build_index(MARKDOWN_DOC)
    other = document_index.build_index("# Elsewhere\n")

    with self.assertRaisesRegex(KeyError, "not part of this index"):
      index.heading_path(other.roots[0])

  def test_find_unknown_id_raises_key_error(self):
    index = document_index.build_index(MARKDOWN_DOC)

    with self.assertRaises(KeyError):
      index.find("9999")

  @parameterized.named_parameters(
      ("preamble_start", "Intro paragraph", "Preamble"),
      ("parent_own_text", "Opening remarks", "Report"),
      ("nested_leaf", "management fee", "Fees"),
      ("heading_line_itself", "## Governance", "Governance"),
  )
  def test_section_at_returns_deepest_containing_section(
      self, needle, expected_title
  ):
    index = document_index.build_index(MARKDOWN_DOC)

    section = index.section_at(MARKDOWN_DOC.index(needle))

    self.assertEqual(section.title, expected_title)

  @parameterized.named_parameters(
      ("past_end", len(MARKDOWN_DOC)), ("negative", -1)
  )
  def test_section_at_outside_text_raises(self, pos):
    index = document_index.build_index(MARKDOWN_DOC)

    with self.assertRaises(IndexError):
      index.section_at(pos)

  def test_heading_path_lists_titles_from_root(self):
    index = document_index.build_index(MARKDOWN_DOC)

    self.assertEqual(
        index.heading_path(index.find("0004")),
        ["Report", "Finances", "Fees"],
    )
    self.assertEqual(index.heading_path(index.find("0001")), ["Preamble"])

  def test_to_dict_uses_pageindex_style_keys(self):
    index = document_index.build_index("# A\n\n## B\n\nx\n")

    self.assertEqual(
        index.to_dict(),
        [{
            "title": "A",
            "node_id": "0001",
            "start_index": 0,
            "end_index": 13,
            "nodes": [{
                "title": "B",
                "node_id": "0002",
                "start_index": 5,
                "end_index": 13,
            }],
        }],
    )

  def test_to_toc_indents_by_depth_and_shows_summaries(self):
    index = document_index.build_index("# A\n\n## B\n\nx\n")
    index.find("0002").summary = "About B."

    self.assertEqual(
        index.to_toc(),
        "0001 | A\n  0002 | B — About B.",
    )
    self.assertEqual(
        index.to_toc(include_summaries=False), "0001 | A\n  0002 | B"
    )


class SummarizeIndexTest(absltest.TestCase):

  def test_short_sections_are_summarized_verbatim_without_model_calls(self):
    index = document_index.build_index("# A\n\nshort   text\n\n# B\n\nmore\n")
    model = FakeLanguageModel([])

    document_index.summarize_index(index, model)

    self.assertEmpty(model.prompts)
    self.assertEqual(index.find("0001").summary, "short text")
    self.assertEqual(index.find("0002").summary, "more")

  def test_long_sections_use_model_and_summary_excludes_children(self):
    long_body = "lorem ipsum " * 20
    text = f"# A\n\n{long_body}\n\n## Child\n\nchild body\n"
    index = document_index.build_index(text)
    model = FakeLanguageModel(["Summary of A."])

    document_index.summarize_index(index, model, max_verbatim_chars=50)

    self.assertLen(model.prompts, 1)
    self.assertIn(long_body.strip(), model.prompts[0])
    self.assertNotIn("child body", model.prompts[0])
    self.assertEqual(index.find("0001").summary, "Summary of A.")
    self.assertEqual(index.find("0002").summary, "child body")

  def test_summary_input_is_truncated(self):
    text = "# A\n\n" + "x" * 5000 + "\n"
    index = document_index.build_index(text)
    model = FakeLanguageModel(["S"])

    document_index.summarize_index(
        index, model, max_verbatim_chars=10, max_summary_input_chars=100
    )

    self.assertIn("x" * 95, model.prompts[0])
    self.assertNotIn("x" * 101, model.prompts[0])

  def test_model_calls_are_batched(self):
    text = "".join(f"# S{i}\n\n{'y' * 100}\n\n" for i in range(5))
    index = document_index.build_index(text)
    model = FakeLanguageModel([f"sum{i}" for i in range(5)])

    with mock.patch.object(
        model, "infer", wraps=model.infer, autospec=True
    ) as infer:
      document_index.summarize_index(
          index, model, max_verbatim_chars=10, batch_length=2
      )

    self.assertEqual(
        [len(c.kwargs["batch_prompts"]) for c in infer.call_args_list],
        [2, 2, 1],
    )
    self.assertEqual(
        [n.summary for n in index.iter_nodes()],
        ["sum0", "sum1", "sum2", "sum3", "sum4"],
    )

  def test_model_with_schema_is_rejected(self):
    index = document_index.build_index("# A\n\n" + "z" * 100)
    model = FakeLanguageModel(["S"])
    model.apply_schema(mock.Mock())

    with self.assertRaisesRegex(document_index.DocumentIndexError, "schema"):
      document_index.summarize_index(index, model, max_verbatim_chars=10)


class SelectSectionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.index = document_index.build_index(MARKDOWN_DOC)

  @parameterized.named_parameters(
      ("raw_json", '{"node_ids": ["0004", "0003"], "reasoning": "fees"}'),
      (
          "fenced_json",
          '```json\n{"node_ids": ["0004", "0003"], "reasoning": "fees"}\n```',
      ),
      ("integer_ids", '{"node_ids": [4, 3]}'),
      ("padded_strings", '{"node_ids": [" 0004 ", "3"]}'),
      (
          "surrounding_prose",
          'Sure, here you go:\n{"node_ids": ["0004", "0003"]}\nI chose fees.',
      ),
  )
  def test_returns_selected_nodes_in_document_order(self, output):
    model = FakeLanguageModel([output])

    selected = document_index.select_sections(
        self.index, "Extract fee percentages.", model
    )

    self.assertEqual([n.node_id for n in selected], ["0003", "0004"])

  def test_prompt_contains_task_toc_and_example_classes(self):
    model = FakeLanguageModel(['{"node_ids": ["0004"]}'])
    examples = [
        data.ExampleData(
            text="Fee is 1%.",
            extractions=[
                data.Extraction(extraction_class="fee", extraction_text="1%")
            ],
        )
    ]

    document_index.select_sections(
        self.index, "Extract fee percentages.", model, examples=examples
    )

    prompt = model.prompts[0]
    self.assertIn("Extract fee percentages.", prompt)
    self.assertIn("0004 | Fees", prompt)
    self.assertIn("Extraction classes: fee", prompt)

  def test_unknown_ids_are_dropped_and_duplicates_collapsed(self):
    model = FakeLanguageModel(['{"node_ids": ["0004", "9999", "0004"]}'])

    with self.assertLogs(level="WARNING") as logs:
      selected = document_index.select_sections(self.index, "task", model)

    self.assertEqual([n.node_id for n in selected], ["0004"])
    self.assertIn("9999", "\n".join(logs.output))

  def test_empty_selection_warns_and_returns_empty_list(self):
    model = FakeLanguageModel(['{"node_ids": []}'])

    with self.assertLogs(level="WARNING") as logs:
      selected = document_index.select_sections(self.index, "task", model)

    self.assertEmpty(selected)
    self.assertIn("no sections", "\n".join(logs.output).lower())

  @parameterized.named_parameters(
      ("not_json", "I think sections 3 and 4."),
      ("wrong_shape", '["0003", "0004"]'),
      ("missing_key", '{"sections": ["0003"]}'),
  )
  def test_unparseable_output_raises(self, output):
    model = FakeLanguageModel([output])

    with self.assertRaises(document_index.DocumentIndexError):
      document_index.select_sections(self.index, "task", model)

  def test_model_with_schema_is_rejected(self):
    model = FakeLanguageModel(['{"node_ids": ["0004"]}'])
    model.apply_schema(mock.Mock())

    with self.assertRaisesRegex(document_index.DocumentIndexError, "schema"):
      document_index.select_sections(self.index, "task", model)


class SectionChunkFilterTest(parameterized.TestCase):

  def _chunk(self, text, start, end):
    doc = data.Document(text=text)
    tokenized = tokenizer.tokenize(text)
    doc.tokenized_text = tokenized
    token_indices = [
        i
        for i, tok in enumerate(tokenized.tokens)
        if tok.char_interval.start_pos >= start
        and tok.char_interval.end_pos <= end
    ]
    return chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(
            start_index=token_indices[0], end_index=token_indices[-1] + 1
        ),
        document=doc,
    )

  def test_overlap_semantics(self):
    text = "aaa bbb ccc ddd eee fff"
    section = document_index.SectionNode(
        node_id="0001",
        title="ccc ddd",
        level=1,
        char_interval=data.CharInterval(start_pos=8, end_pos=15),
    )
    keep = document_index.section_chunk_filter([section])

    self.assertTrue(keep(self._chunk(text, 8, 11)))  # inside
    self.assertTrue(keep(self._chunk(text, 4, 11)))  # straddles start
    self.assertTrue(keep(self._chunk(text, 12, 19)))  # straddles end
    self.assertFalse(keep(self._chunk(text, 0, 7)))  # before
    self.assertFalse(keep(self._chunk(text, 16, 23)))  # after

  def test_empty_selection_filters_everything(self):
    text = "aaa bbb"
    keep = document_index.section_chunk_filter([])

    self.assertFalse(keep(self._chunk(text, 0, 7)))


class ExtractWithChunkFilterTest(absltest.TestCase):

  def test_only_selected_sections_reach_the_model(self):
    index = document_index.build_index(MARKDOWN_DOC)
    fees = index.find("0004")
    extraction_json = (
        '{"extractions": [{"fee": "2%", "fee_attributes": {"kind":'
        ' "management"}}]}'
    )
    # The 40-char buffer splits the Fees section into two chunks; the second
    # straddles into the next heading, which the overlap filter still keeps.
    model = FakeLanguageModel(['{"extractions": []}', extraction_json])
    examples = [
        data.ExampleData(
            text="Fee is 1%.",
            extractions=[
                data.Extraction(extraction_class="fee", extraction_text="1%")
            ],
        )
    ]

    result = lx.extract(
        MARKDOWN_DOC,
        prompt_description="Extract fee percentages.",
        examples=examples,
        model=model,
        fence_output=False,
        use_schema_constraints=False,
        max_char_buffer=40,
        chunk_filter=document_index.section_chunk_filter([fees]),
        show_progress=False,
    )

    self.assertLen(model.prompts, 2)
    sent = "\n".join(model.prompts)
    self.assertIn("management fee is 2%", sent)
    self.assertNotIn("Opening remarks", sent)
    self.assertNotIn("board met", sent)
    self.assertLen(result.extractions, 1)
    extraction = result.extractions[0]
    interval = extraction.char_interval
    self.assertEqual(MARKDOWN_DOC[interval.start_pos : interval.end_pos], "2%")
    self.assertGreaterEqual(interval.start_pos, fees.char_interval.start_pos)

  def test_filter_applies_per_document_and_keeps_document_order(self):
    docs = [
        data.Document(text="# Skip\n\nnothing here\n", document_id="a"),
        data.Document(text="# Keep\n\nfee is 3%\n", document_id="b"),
    ]
    keep_index = document_index.build_index(docs[1].text)
    model = FakeLanguageModel(['{"extractions": [{"fee": "3%"}]}'])
    examples = [
        data.ExampleData(
            text="Fee is 1%.",
            extractions=[
                data.Extraction(extraction_class="fee", extraction_text="1%")
            ],
        )
    ]

    def keep(chunk):
      return chunk.document_id == "b" and document_index.section_chunk_filter(
          keep_index.roots
      )(chunk)

    results = lx.extract(
        docs,
        prompt_description="Extract fees.",
        examples=examples,
        model=model,
        fence_output=False,
        use_schema_constraints=False,
        chunk_filter=keep,
        show_progress=False,
    )

    self.assertLen(model.prompts, 1)
    self.assertEqual([r.document_id for r in results], ["a", "b"])
    self.assertEmpty(results[0].extractions)
    self.assertEqual(results[1].extractions[0].extraction_text, "3%")


def _grounded(text, extraction_class, needle, **attributes):
  start = text.index(needle)
  return data.Extraction(
      extraction_class,
      needle,
      char_interval=data.CharInterval(
          start_pos=start, end_pos=start + len(needle)
      ),
      attributes=attributes or None,
  )


class AttachExtractionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.index = document_index.build_index(MARKDOWN_DOC)
    self.document = data.AnnotatedDocument(
        text=MARKDOWN_DOC,
        extractions=[
            _grounded(MARKDOWN_DOC, "metric", "10%"),
            _grounded(MARKDOWN_DOC, "fee", "2%", kind="management"),
            _grounded(MARKDOWN_DOC, "event", "met twice"),
            data.Extraction("fee", "unfound"),
        ],
    )

  def test_extractions_are_filed_under_the_deepest_section(self):
    self.index.attach_extractions(self.document)

    self.assertEqual(
        [e.extraction_text for e in self.index.find("0003").extractions],
        ["10%"],
    )
    self.assertEqual(
        [e.extraction_text for e in self.index.find("0004").extractions],
        ["2%"],
    )
    self.assertEqual(
        [e.extraction_text for e in self.index.find("0003").all_extractions()],
        ["10%", "2%"],
    )
    self.assertEqual(
        [e.extraction_text for e in self.index.unplaced], ["unfound"]
    )

  def test_attaching_again_replaces_previous_extractions(self):
    self.index.attach_extractions(self.document)
    self.index.attach_extractions(
        data.AnnotatedDocument(text=MARKDOWN_DOC, extractions=[])
    )

    self.assertEmpty(self.index.find("0004").extractions)
    self.assertEmpty(self.index.unplaced)

  def test_text_mismatch_raises(self):
    with self.assertRaisesRegex(document_index.DocumentIndexError, "same text"):
      self.index.attach_extractions(
          data.AnnotatedDocument(text="other", extractions=[])
      )

  def test_toc_shows_subtree_counts_per_class(self):
    self.index.attach_extractions(self.document)

    self.assertEqual(
        self.index.to_toc().splitlines(),
        [
            "0001 | Preamble",
            "0002 | Report [1 event, 1 fee, 1 metric]",
            "  0003 | Finances [1 fee, 1 metric]",
            "    0004 | Fees [1 fee]",
            "  0005 | Governance [1 event]",
        ],
    )

  def test_toc_can_hide_sections_without_extractions(self):
    self.index.attach_extractions(self.document)

    self.assertEqual(
        [
            line.strip()
            for line in self.index.to_toc(hide_empty=True).splitlines()
        ],
        [
            "0002 | Report [1 event, 1 fee, 1 metric]",
            "0003 | Finances [1 fee, 1 metric]",
            "0004 | Fees [1 fee]",
            "0005 | Governance [1 event]",
        ],
    )

  def test_section_view_returns_subtree_extractions_and_children(self):
    self.index.attach_extractions(self.document)

    view = self.index.section_view("0003")

    self.assertEqual(view["path"], ["Report", "Finances"])
    self.assertEqual(
        view["extractions"],
        [
            {
                "extraction_class": "metric",
                "extraction_text": "10%",
                "attributes": {},
                "start_index": MARKDOWN_DOC.index("10%"),
                "end_index": MARKDOWN_DOC.index("10%") + 3,
            },
            {
                "extraction_class": "fee",
                "extraction_text": "2%",
                "attributes": {"kind": "management"},
                "start_index": MARKDOWN_DOC.index("2%"),
                "end_index": MARKDOWN_DOC.index("2%") + 2,
            },
        ],
    )
    self.assertEqual(
        view["subsections"],
        [{"node_id": "0004", "title": "Fees", "extraction_count": 1}],
    )
    self.assertNotIn("text", view)

  def test_section_view_can_include_source_text(self):
    view = self.index.section_view("0004", include_text=True)

    self.assertEqual(view["text"], "### Fees\n\nThe management fee is 2%.\n\n")
    self.assertEmpty(view["extractions"])

  def test_to_dict_adds_counts_and_optional_extractions(self):
    self.index.attach_extractions(self.document)

    tree = self.index.to_dict(include_extractions=True)

    report = tree[1]
    self.assertEqual(report["extraction_count"], 3)
    self.assertNotIn("extractions", report)
    fees = report["nodes"][0]["nodes"][0]
    self.assertEqual(fees["extraction_count"], 1)
    self.assertEqual(fees["extractions"][0]["extraction_text"], "2%")

  def test_to_dict_omits_counts_before_attaching(self):
    self.assertNotIn("extraction_count", self.index.to_dict()[0])


if __name__ == "__main__":
  absltest.main()
