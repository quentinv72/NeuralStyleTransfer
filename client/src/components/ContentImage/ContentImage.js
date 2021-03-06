/* eslint-disable jsx-a11y/img-redundant-alt */
/* eslint-disable react/jsx-no-comment-textnodes */
import React, { useState, useCallback, useEffect } from "react";
import ReactCrop from "react-image-crop";
import { trackPromise } from "react-promise-tracker";
import { Link } from "react-router-dom";
import "./ContentImage.css";
import "react-image-crop/dist/ReactCrop.css";
const axios = require("axios").default;
export default function ContentImage(props) {
  const [image, setImage] = useState(null);
  const [imgRef, setImgRef] = useState(null);
  const [nopic, setNopic] = useState({ marginTop: "50vh" });
  const [crop, setCrop] = useState(null);

  // API call
  const fetchAPI = async (contentImage) => {
    try {
      const response = await axios({
        method: "post",
        url: "/api/imgs",
        data: {
          content_image: contentImage.replace(/^data:image\/jpeg;base64,/, ""),
          style_image: props.styleImage.replace(
            /^data:image\/jpeg;base64,/,
            ""
          ),
        },
        timeout: 180000,
      });
      if (response.statusText === "OK") {
        const base64GeneratedImage = response.data.imageb64;
        props.genImg(`data:image/jpeg;base64, ${base64GeneratedImage}`);
      }
    } catch (e) {
      console.log(e.data);
    }
  };

  useEffect(() => {
    if (imgRef) {
      setCrop({
        width: 256,
        height:
          256 *
          (imgRef.naturalWidth / imgRef.width) *
          (imgRef.height / imgRef.naturalHeight),
        x: 0,
        y: 0,
      });
      setNopic(null);
    }
  }, [imgRef]);

  const onLoad = useCallback((img) => {
    setImgRef(img);
  }, []);

  // Generate the neural transfer
  const handleClick = async (e) => {
    if (image) {
      const content = getCroppedImg(crop, "content_image.jpeg");
      trackPromise(fetchAPI(content));
    }
  };

  // Save Cropped Image
  function getCroppedImg(crop, fileName) {
    let imageCrop = imgRef;
    const canvas = document.createElement("canvas");
    const scaleX = imageCrop.naturalWidth / imageCrop.width;
    const scaleY = imageCrop.naturalHeight / imageCrop.height;
    canvas.width = Math.round(crop.width * scaleX);

    canvas.height = Math.round(crop.height * scaleY);
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
          style={nopic}
          id='content-image'
          name='content'
          onChange={handleChangeImage}
          accept='image/png, image/jpeg'
        />
        {image ? (
          <Link
            className='uk-button uk-button-default uk-button-large uk-button-primary'
            onClick={handleClick}
            to='/download'>
            Generate
          </Link>
        ) : (
          <button
            className='uk-button uk-button-default uk-button-large uk-button-primary'
            style={nopic}>
            Generate
          </button>
        )}
      </div>
    </div>
  );
}
