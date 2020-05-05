from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse

jake_paul = "../data/paul.jpg"
app = FastAPI()

origins = ["http://localhost:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost"])


# POST request handler for path /api/imgs... User uploads images and receives a generated image as response
@app.post("/api/imgs")
async def upload_file(
    content_image: UploadFile = File(..., title="User Image"),
    style_image: UploadFile = File(..., title="Style Image"),
):
    accepted_types = ["image/png", "image/jpeg"]
    if (content_image.content_type not in accepted_types) or (
        style_image.content_type not in accepted_types
    ):
        raise HTTPException(400, detail="Image must be either .png or .jpeg/jpg")
    # Add an await for algorithm to run and upload image which will be the file response
    return FileResponse(jake_paul, filename="jake_paul", media_type="image/png")


# Might have to manage some request timeout stuff if algo takes too long to run.. Might be able to handle that on front end though
