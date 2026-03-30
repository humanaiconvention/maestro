"""Unit tests for BenchmarkRegistry and NaturalDatasetLoader with mocked HF datasets."""

import pytest
from unittest.mock import patch, MagicMock
from semantic_grounding.evaluator.tasks import BenchmarkRegistry
from semantic_grounding.datasets.natural.loaders import NaturalDatasetLoader


# ---------------------------------------------------------------------------
# BenchmarkRegistry
# ---------------------------------------------------------------------------

class TestBenchmarkRegistry:

    @patch("semantic_grounding.evaluator.tasks.load_dataset")
    def test_load_arc_formats_correctly(self, mock_ld):
        """ARC items are formatted as QA prompts with the correct answer."""
        fake_item = {
            "question": "What is H2O?",
            "choices": {"label": ["A", "B", "C"], "text": ["Fire", "Water", "Earth"]},
            "answerKey": "B",
        }
        fake_ds = MagicMock()
        fake_ds.shuffle.return_value = fake_ds
        fake_ds.select.return_value = [fake_item]
        fake_ds.__len__ = lambda self: 1

        # load_dataset is called three times (_load_arc, _load_gsm8k, _load_wikitext)
        mock_ld.return_value = fake_ds

        registry = BenchmarkRegistry(eval_samples=1, seed=0)

        arc = registry.datasets["arc_easy"]
        assert len(arc) == 1
        assert arc[0]["prompt"].startswith("Question:")
        assert "Water" in arc[0]["expected"]

    @patch("semantic_grounding.evaluator.tasks.load_dataset")
    def test_load_wikitext_filters_short(self, mock_ld):
        """Wikitext items shorter than 50 chars are excluded."""
        short = {"text": "Hi"}
        long = {"text": "A" * 60}

        fake_ds = MagicMock()
        fake_ds.shuffle.return_value = fake_ds
        fake_ds.select.return_value = []
        fake_ds.__len__ = lambda self: 0
        fake_ds.__iter__ = lambda self: iter([short, long])

        mock_ld.return_value = fake_ds

        registry = BenchmarkRegistry(eval_samples=10, seed=0)
        wiki = registry.datasets["wikitext"]
        # Only the long item should survive
        assert len(wiki) == 1
        assert len(wiki[0]["expected"]) >= 50

    @patch("semantic_grounding.evaluator.tasks.load_dataset")
    def test_load_failure_returns_empty(self, mock_ld):
        """Network errors return empty lists, not exceptions."""
        mock_ld.side_effect = Exception("Network down")
        registry = BenchmarkRegistry(eval_samples=5, seed=0)
        assert registry.datasets["arc_easy"] == []
        assert registry.datasets["gsm8k"] == []
        assert registry.datasets["wikitext"] == []

    @patch("semantic_grounding.evaluator.tasks.load_dataset")
    def test_evaluate_model_returns_generation(self, mock_ld):
        """evaluate_model returns a dict with 'generation' key even when datasets empty."""
        mock_ld.side_effect = Exception("Offline")
        registry = BenchmarkRegistry(eval_samples=1, seed=0)

        mock_trainer = MagicMock()
        result = registry.evaluate_model(mock_trainer, generation_idx=3)
        assert result["generation"] == 3

    @patch("semantic_grounding.evaluator.tasks.load_dataset")
    def test_evaluate_model_calls_trainer(self, mock_ld):
        """evaluate_model calls trainer.calculate_perplexity and generate_synthetic_data."""
        # Provide minimal wikitext and arc data
        wiki_item = {"text": "A" * 60}
        arc_item = {
            "question": "Q?",
            "choices": {"label": ["A"], "text": ["Ans"]},
            "answerKey": "A",
        }

        def fake_load(name, *args, **kwargs):
            if "wikitext" in name:
                return MagicMock(__iter__=lambda s: iter([wiki_item]))
            ds = MagicMock()
            ds.shuffle.return_value = ds
            ds.select.return_value = [arc_item] if "arc" in name else []
            ds.__len__ = lambda s: 1
            return ds

        mock_ld.side_effect = fake_load

        registry = BenchmarkRegistry(eval_samples=1, seed=0)

        mock_trainer = MagicMock()
        mock_trainer.calculate_perplexity.return_value = 10.0
        mock_trainer.generate_synthetic_data.return_value = [
            {"prompt": "Q?", "completion": " Ans"}
        ]

        result = registry.evaluate_model(mock_trainer, generation_idx=0)
        assert "val_perplexity" in result
        assert mock_trainer.calculate_perplexity.called


# ---------------------------------------------------------------------------
# NaturalDatasetLoader
# ---------------------------------------------------------------------------

class TestNaturalDatasetLoader:

    @patch("semantic_grounding.datasets.natural.loaders.load_dataset")
    def test_load_arc_easy_train_eval(self, mock_ld):
        """load_arc_easy returns (train, eval) tuple with formatted dicts."""
        arc_item = {
            "question": "Color of sky?",
            "choices": {"label": ["A", "B"], "text": ["Red", "Blue"]},
            "answerKey": "B",
        }

        fake_split = MagicMock()
        fake_split.shuffle.return_value = fake_split
        fake_split.select.return_value = [arc_item]
        fake_split.__len__ = lambda self: 1

        mock_ld.return_value = {"train": fake_split, "test": fake_split}

        train, eval_set = NaturalDatasetLoader.load_arc_easy(num_train=1, num_eval=1)
        assert len(train) == 1
        assert len(eval_set) == 1
        assert "Blue" in train[0]["expected"]
        assert "completion" in train[0]  # training format includes completion

    @patch("semantic_grounding.datasets.natural.loaders.load_dataset")
    def test_load_hellaswag(self, mock_ld):
        """load_hellaswag returns formatted dicts with Context/Next step prompt."""
        item = {
            "ctx": "A person walks into a room.",
            "endings": ["sits down", "flies away", "vanishes", "cooks"],
            "label": "0",
        }
        fake_split = MagicMock()
        fake_split.shuffle.return_value = fake_split
        fake_split.select.return_value = [item]
        fake_split.__len__ = lambda self: 1

        mock_ld.return_value = {"validation": fake_split}

        result = NaturalDatasetLoader.load_hellaswag(num_eval=1)
        assert len(result) == 1
        assert "Context:" in result[0]["prompt"]
        assert "sits down" in result[0]["expected"]

    @patch("semantic_grounding.datasets.natural.loaders.load_dataset")
    def test_load_wikitext_filters(self, mock_ld):
        """load_wikitext filters out short/empty lines."""
        short = {"text": "Hi"}
        long = {"text": "B" * 60}

        mock_ld.return_value = {"test": [short, long]}

        result = NaturalDatasetLoader.load_wikitext(num_samples=10)
        assert len(result) == 1
        assert len(result[0]["expected"]) >= 50

    @patch("semantic_grounding.datasets.natural.loaders.load_dataset")
    def test_load_arc_failure(self, mock_ld):
        """Network failure returns empty lists."""
        mock_ld.side_effect = Exception("No internet")
        train, eval_set = NaturalDatasetLoader.load_arc_easy()
        assert train == []
        assert eval_set == []

    @patch("semantic_grounding.datasets.natural.loaders.load_dataset")
    def test_load_hellaswag_failure(self, mock_ld):
        mock_ld.side_effect = Exception("Timeout")
        result = NaturalDatasetLoader.load_hellaswag()
        assert result == []

    @patch("semantic_grounding.datasets.natural.loaders.load_dataset")
    def test_load_wikitext_failure(self, mock_ld):
        mock_ld.side_effect = Exception("DNS error")
        result = NaturalDatasetLoader.load_wikitext()
        assert result == []
