import base64
import binascii

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

jake_paul = "../data/paul.jpg"
app = FastAPI()

origins = ["http://localhost:3000"]


class Images(BaseModel):
    content_image: str
    style_image: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost"])

# Might have to manage some request timeout stuff if algo takes too long to run.. Might be able to handle that on front end though
@app.post("/api/imgs")
async def images(images: Images = Body(...)):
    try:
        content_image = base64.b64decode(images.content_image)
        style_image = base64.b64decode(images.style_image)
        # Add an await for algorithm to run and upload image which will be the file response
        return FileResponse(jake_paul, filename="jake_paul", media_type="image/png")
    except binascii.Error:
        print("caught error")
        raise HTTPException(400, detail="One of the input images not in base64 format")
