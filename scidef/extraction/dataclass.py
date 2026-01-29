from enum import Enum


class ChunkMode(Enum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SECTION = "section"
    FULL = "full"
    THREE_SENTENCE = "threeSentence"
