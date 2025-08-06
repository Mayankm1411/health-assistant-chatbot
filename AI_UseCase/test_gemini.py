import google.generativeai as genai

genai.configure(api_key="AIzaSyBntoybBym3lKOp3DoTqc59TWKoYnqnE8I")

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("What is the capital of France?")
print(response.text)


