from typing import List, Tuple, Optional
from transformers import pipeline

from src.utils.logger_config import get_logger

logger = get_logger(__name__)


class NERProcessor:
    def __init__(
        self,
        model_name: str = "d4data/biomedical-ner-all",
        device: Optional[str] = None,
        ner=None,
        adjacent_separator: str = "",
    ):
        self.ner = (
            pipeline("token-classification", model=model_name, device=device)
            if not ner
            else ner
        )
        self.adjacent_separator = adjacent_separator

    def get_ner_entities(self, text: str) -> List[Tuple[str, Optional[str], int]]:
        try:
            entities = self.ner(text, grouped_entities=True)
        except Exception as e:
            logger.error(f"Error in NER Engine: {e}")
            entities = []

        if len(entities) == 0:  # type: ignore
            entity_list = [(text, None, 0)]
        else:
            list_format = []
            index = 0
            entities = sorted(entities, key=lambda x: x["start"])  # type: ignore
            for entity in entities:
                list_format.append((text[index : entity["start"]], None, index))
                entity_category = entity.get("entity") or entity.get("entity_group")
                list_format.append(
                    (
                        text[entity["start"] : entity["end"]],
                        entity_category,
                        entity["start"],
                    )
                )
                index = entity["end"]
            list_format.append((text[index:], None, index))
            entity_list = list_format

        output = []
        running_text, running_category, running_start = None, None, None
        for entity, category, start in entity_list:
            if running_text is None:
                running_text = entity
                running_category = category
                running_start = start
            elif category == running_category:
                running_text += self.adjacent_separator + entity
            elif not entity:
                pass
            else:
                output.append((running_text, running_category, running_start))
                running_text = entity
                running_category = category
                running_start = start
        if running_text is not None:
            output.append((running_text, running_category, running_start))
        return output

    def get_ner_inserted_text(self, text: str) -> str:
        modified_text = text
        offset = 0
        output = self.get_ner_entities(text)
        for word, category, start in output:
            if category:
                insertion = f"[{category}]"
                modified_text = (
                    modified_text[: start + len(word) + offset]
                    + insertion
                    + modified_text[start + len(word) + offset :]
                )
                offset += len(insertion)
        total_text = f"{text}\n\nNER inserted text:\n{modified_text}"
        return total_text


class DummyNERProcessor(NERProcessor):
    def __init__(
        self,
        model_name: str = "dummy",
        device: Optional[str] = None,
        ner=None,
        adjacent_separator: str = "",
    ):
        self.model_name = model_name
        self.device = device
        self.ner = ner
        self.adjacent_separator = adjacent_separator

    def get_ner_entities(self, text: str) -> List[Tuple[str, Optional[str], int]]:
        # This method doesn't actually perform NER. It just returns some hard-coded results.
        return [
            (text, "ORG", 0),
            (text, "LOC", 27),
            (text, "MONEY", 44),
        ]

    def get_ner_inserted_text(self, text: str) -> str:
        # This method doesn't actually perform NER. It just returns the input text with some hard-coded insertions.
        return f"{text}\n\nNER inserted text:\n{text} [ORG] [LOC] [MONEY]"


if __name__ == "__main__":
    # Sample NER function
    def sample_ner(text, grouped_entities=False):
        return [
            {"entity": "ORG", "start": 0, "end": 5},
            {"entity": "LOC", "start": 27, "end": 30},
            {"entity": "MONEY", "start": 44, "end": 53},
        ]

    # Test the function
    ner_processor = NERProcessor(ner=sample_ner)
    text = "Apple is looking at buying U.K. startup for $1 billion."
    result = ner_processor.get_ner_inserted_text(text)
    print(
        result
    )  # Expected output: "Apple [ORG] is looking at buying [LOC] U.K. startup for [MONEY] $1 billion."
