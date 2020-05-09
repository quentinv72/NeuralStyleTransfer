/* eslint-disable jsx-a11y/anchor-has-content */
/* eslint-disable jsx-a11y/anchor-is-valid */
import React, { useState } from "react";
import dancing from "./Images/dancing.jpg";
import ernst from "./Images/ernst.jpg";
import krichner from "./Images/krichner.jpg";
import picasso from "./Images/picasso.jpg";
import rembrandt from "./Images/rembrandt.jpg";

import { Link } from "react-router-dom";
import "./StyleImages.css";
import StyleImage from "./StyleImage";

export default function StyleImages(props) {
  const [clicked, setClicked] = useState(null);
  const handleClick = (e) => {
    props.onClick(e);
    setClicked(e.target.id);
  };

  const imageURLS = [picasso, rembrandt, ernst, krichner, dancing]; //will use an object id:url
  const keys = ["picasso", "rembrandt", "ernst", "krichner", "dancing"];

  return (
    <div className='uk-container'>
      <h2 className='uk-text-center'>Choose Your Style Image</h2>
      <div className='grid uk-width-7-10'>
        {imageURLS.map((image, index) => {
          if (image === clicked) {
            return (
              <StyleImage
                src={image}
                key={keys[index]}
                onClick={handleClick}
                clicked={true}
                id={image}
              />
            );
          }
          return (
            <StyleImage
              src={image}
              key={keys[index]}
              onClick={handleClick}
              clicked={false}
              id={image}
            />
          );
        })}
      </div>
      <div className='uk-flex uk-flex-center uk-margin-large uk-margin-bottom'>
        {clicked ? (
          <Link
            className='uk-button uk-button-default uk-button-large uk-button-primary'
            href='#content-image'
            to='/content'>
            Next
          </Link>
        ) : (
          <a
            className='uk-button uk-button-default uk-button-large uk-button-primary'
            href='#'>
            Next
          </a>
        )}
      </div>
    </div>
  );
}
