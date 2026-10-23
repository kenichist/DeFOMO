from transformers import DistilBertTokenizer, DistilBertModel

model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

tokenizer.save_pretrained("distilbert-base-uncased")
model.save_pretrained("distilbert-base-uncased")