from transformers import pipeline


class SubjectModel:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion classification model.

        Args:
            model_name (str, optional): Name of the pre-trained model to use. Defaults to 'j-hartmann/emotion-english-distilroberta-base'.
        """
        self.classifier = pipeline("text-classification", model=model_name, top_k=None)

    def get_subjects(self, text):
        """
        Predicts the emotions and their scores for the given text.

        Args:
            text (str): Text for which to predict emotions.

        Returns:
            list: List of dictionaries containing predicted emotions as keys and their scores as values.
        """
        return self.classifier(text)