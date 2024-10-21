import requests
from bs4 import BeautifulSoup

url = "https://boxrec.com/en/boxer/1"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

print(response.content)  # Print the entire HTML to inspect it

soup = BeautifulSoup(response.content, "html.parser")
