Steps to use this app:

1) Create a virtual environment using "python -m venv venv"
2) Activate venv using "venv/Scripts/activate
3) You need to clone this app now.
4) Install all packages using "python install -r requirements.txt"
5) Create an ".env" file and place your grok api key and google api key (Grok is used for the RAG model and google api is used for embeddings).
6) Add a folder "files" with some pdfs on which you need to query,here I have used an book that consist of Disease and their Ayurvedic medicines.
7) Go to terminal and hit "python api.py"
8) It takes time initially beacuse it has to create the vectors but afterwards the response is fast.

![image](https://github.com/user-attachments/assets/271448f6-a8a3-4b41-a82d-cc1a56ab26a3)
