import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

key = os.getenv("GROQ_API_KEY")
print("Key loaded:", bool(key))

client = Groq(api_key=key)
models = client.models.list()
for m in models.data:
    print(m.id)