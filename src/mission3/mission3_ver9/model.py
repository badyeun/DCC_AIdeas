"""Submission entry point for the DeBERTa dialogue classifier architecture."""
if __package__:
    from .neural import DialogueClassifier
else:
    from neural import DialogueClassifier

__all__ = ["DialogueClassifier"]
