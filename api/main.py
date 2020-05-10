import binascii

import torch
import torchvision.models as models
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from neural_transfer import image_loader, run_style_transfer, tensor_to_base64
from pydantic import BaseModel

app = FastAPI()

origins = ["http://localhost:3000"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()


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


@app.post("/api/imgs")
async def images(images: Images = Body(...)):
    try:
        encoded_content_image = images.content_image
        encoded_style_image = images.style_image
        content_tensor = image_loader(encoded_content_image)
        style_tensor = image_loader(encoded_style_image)
        input_img = content_tensor.clone()
        output = run_style_transfer(cnn, content_tensor, style_tensor, input_img,)
        generated_image_b64 = tensor_to_base64(output)
        return {"imageb64": generated_image_b64}
    except binascii.Error:
        raise HTTPException(400, detail="One of the input images not in base64 format")
