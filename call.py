from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import httpx

client = OpenAI(
    base_url="http://localhost:8001/v1", 
    api_key="DUMMY_API_KEY",
    timeout=httpx.Timeout(10.0, connect=5.0)
)

model = "nanonets/Nanonets-OCR2-3B"

def encode_image(image_input):
    if isinstance(image_input, Image.Image):
        buffer = BytesIO()
        image_input.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def infer(img_base64):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {
                            "type": "text",
                            "text": "Extract the text from the above document as if you were reading it naturally.",
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=15000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during inference: {type(e).__name__}: {e}")
        raise