/* eslint-disable jsx-a11y/img-redundant-alt */
/* eslint-disable react/jsx-no-comment-textnodes */
import React, { useState, useCallback } from "react";
import ReactCrop, { containCrop } from "react-image-crop";
import "./ContentImage.css";
import "react-image-crop/dist/ReactCrop.css";

export default function ContentImage(props) {
  const [image, setImage] = useState(null);
  const [imgRef, setImgRef] = useState(null);

  const [crop, setCrop] = useState({
    width: 256,
    height: 256,
  });

  // API call
  const fetchAPI = async (contentImage) => {
    try {
      const body = JSON.stringify({
        content_image: contentImage.replace(/^data:image\/jpeg;base64,/, ""),
        style_image: props.styleImage.replace(/^data:image\/jpeg;base64,/, ""),
      });

      const response = await fetch("http://localhost:8000/api/imgs", {
        body: body,
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "POST",
      });
      if (response.ok) {
        const blobResponse = await response.blob();
        console.log(blobResponse);
      } else {
        const body = await response.json();
        console.log(body);
      }
    } catch (e) {
      console.log(e);
    }
  };

  const onLoad = useCallback((img) => {
    setImgRef(img);
  }, []);

  // Generate the neural transfer
  const handleClick = async (e) => {
    if (image) {
      const content = getCroppedImg(crop, "content_image.jpeg");
      const generate = await fetchAPI(content);
    }
  };

  // Save Cropped Image
  function getCroppedImg(crop, fileName) {
    let imageCrop = imgRef;
    const canvas = document.createElement("canvas");
    const scaleX = imageCrop.naturalWidth / imageCrop.width;
    const scaleY = imageCrop.naturalHeight / imageCrop.height;
    canvas.width = Math.ceil(crop.width * scaleX);
    canvas.height = Math.ceil(crop.height * scaleY);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(
      imageCrop,
      crop.x * scaleX,
      crop.y * scaleY,
      crop.width * scaleX,
      crop.height * scaleY,
      0,
      0,
      crop.width * scaleX,
      crop.height * scaleY
    );
    // Save Image as base64
    const base64Image = canvas.toDataURL("image/jpeg");
    return base64Image;
  }

  // User uploads image
  function handleChangeImage(e) {
    setImage(URL.createObjectURL(e.target.files[0]));
  }

  return (
    <div className='uk-container uk-margin-top'>
      <h2 className='uk-text-center'>Upload Your Content Image</h2>
      <p className='uk-text-center'>
        Choose your content image and then click 'Generate'.
      </p>
      <div className='uk-flex uk-flex-center'>
        <ReactCrop
          src={image}
          onImageLoaded={onLoad}
          crop={crop}
          onChange={(newCrop) => setCrop(newCrop)}
          locked
          imageStyle={{ width: 256, height: "auto" }}
        />
      </div>
      <div className='uk-flex uk-flex-center uk-margin-top uk-flex-around uk-flex-middle'>
        <input
          type='file'
          id='content-image'
          name='content'
          onChange={handleChangeImage}
          accept='image/png, image/jpeg'
        />
        <button
          className='uk-button uk-button-default uk-button-large uk-button-primary'
          onClick={handleClick}>
          Generate
        </button>
      </div>
    </div>
  );
}
