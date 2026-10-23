import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class FakeNewsDetector:
    def __init__(self, model_path='fine_tuned_distilbert'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_path}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def predict(self, text, title=""):
        combined_text = f"{title} {text}".strip()

        encoding = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        label = "REAL" if predicted_class == 1 else "FAKE"

        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'fake': probabilities[0][0].item(),
                'real': probabilities[0][1].item()
            }
        }

def main():
    detector = FakeNewsDetector()

    print("\n" + "="*70)
    print("FAKE NEWS DETECTION")
    print("="*70)

    # Example 1
    example1_title = "Report: Guy Who Sings ‘Thriller’ Has Bunch Of Non-Halloween Music, Too"
    example1_text = """Revealing a vast, forgotten catalog of legitimate pop compositions, a report released Thursday about novelty musical act Michael Joseph Jackson found that the man who recorded the perennial Halloween classic “Thriller” also had a bunch of non-Halloween music. “Jackson is, of course, exclusively known for his haunted hit ‘Thriller’ and singing spooky lyrics like ‘No one’s gonna save you from the beast about to strike,’ but we’ve discovered that he was actually an extremely prolific, earnest musician, having released 10 full-length, mostly non-horror-related albums in his lifetime,” said report co-author Rebecca Rubin of Case Western Reserve University’s Center for Popular Music Studies, pointing out that while Jackson did record a number of lesser-known Halloween novelty songs—like one about a creepy man appearing in the mirror, or another about a woman haunted by the villainous “Smooth Criminal”—the vast majority of his catalog had no connection to the holiday. “Jackson seemed to like to keep a low profile, and even though photographs suggest he was committed to his ghostly Halloween look year round, it appears most of his songs do not begin with a werewolf howl. While today Michael Jackson is fondly remembered as a one-hit Halloween artist, we hope this discovery inspires people to look beyond ‘Thriller’ or even his trick-or-treating anthem ‘Don’t Stop ’Til You Get Enough’ and listen to his more serious works.” Representatives for the Jackson estate confirmed plans to upload his full catalog online to introduce music fans to songs other than his Halloween hit, which they revealed was originally performed as a joke."""

    print("\nExample 1:")
    print(f"Title: {example1_title}")
    print(f"Text: {example1_text[:100]}...")
    result1 = detector.predict(example1_text, example1_title)
    print(f"\nPrediction: {result1['label']}")
    print(f"Confidence: {result1['confidence']:.2%}")
    print(f"Fake probability: {result1['probabilities']['fake']:.2%}")
    print(f"Real probability: {result1['probabilities']['real']:.2%}")

    print("\n" + "-"*70)

    # Example 2
    example2_title = "Trump administration cuts Democrats out of a briefing on US military strikes, top Senate Dem says"
    example2_text = """Trump administration briefed Republicans on Wednesday about US military strikes on suspected drug boats without telling Democratic members about the briefing, according to a top Senate Democrat and a source with direct knowledge of the incident. Senate Intelligence Committee Vice Chairman Mark Warner slammed the Trump administration for cutting out Democratic lawmakers Wednesday, calling the move “a partisan stunt” that “is a slap in the face to Congress’ war powers responsibilities and to the men and women who serve this country.” During the briefing, administration officials showed the Republicans a classified Justice Department OLC memo outlining the administration’s legal justification for the strikes, which CNN first reported on earlier this month. In a statement, Warner said, “shutting Democrats out of a briefing on U.S. military strikes and withholding the legal justification for those strikes from half the Senate is indefensible and dangerous.”"""

    print("\nExample 2:")
    print(f"Title: {example2_title}")
    print(f"Text: {example2_text[:100]}...")
    result2 = detector.predict(example2_text, example2_title)
    print(f"\nPrediction: {result2['label']}")
    print(f"Confidence: {result2['confidence']:.2%}")
    print(f"Fake probability: {result2['probabilities']['fake']:.2%}")
    print(f"Real probability: {result2['probabilities']['real']:.2%}")

    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter 'quit' to exit")

    while True:
        print("\n")
        title = input("Enter news title (or press Enter to skip): ").strip()
        if title.lower() == 'quit':
            break

        text = input("Enter news text: ").strip()
        if text.lower() == 'quit':
            break

        if not text:
            print("Please provide at least the text of the article.")
            continue

        result = detector.predict(text, title)
        print(f"\n{'='*50}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Fake probability: {result['probabilities']['fake']:.2%}")
        print(f"Real probability: {result['probabilities']['real']:.2%}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
