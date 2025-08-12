from typing import Optional
from transformers import AutoTokenizer
import re
from datasets import Dataset

# Constants
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160 
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7 

class BookItem:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this book cost to the nearest dollar?"
    
    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include = False
    
    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.category = data.get('main_category', '')
        self.parse(data)
    
    
    def clean_text(self, stuff):
        if not stuff:
            return ""
        
        if not isinstance(stuff, str):
            stuff = str(stuff)
            
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
        
        words = stuff.split(' ')
        select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        contents = ""
        if 'description' in data and data['description']:
            if isinstance(data['description'], list):
                contents += '\n'.join(data['description']) + '\n'
            else:
                contents += str(data['description']) + '\n'
        if 'features' in data and data['features']:
            if isinstance(data['features'], list):
                contents += '\n'.join(data['features']) + '\n'
            else:
                contents += str(data['features']) + '\n'
        self.details = data.get('details', '')
        if self.details:
            contents += self.clean_text(self.details) + '\n'

        if 'categories' in data and data['categories']:
            contents += "Categories: " + ", ".join(data['categories']) + '\n'

        if 'author' in data and data['author']:
            contents += "Author: " + str(data['author']) + '\n'
        
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            
            text = f"{self.clean_text(self.title)}\n{self.clean_text(contents)}"
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True
    
    def make_prompt(self, text):
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))
    
    def test_prompt(self):
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX
    
    def __repr__(self):
        return f"<{self.title} = ${self.price}>"
