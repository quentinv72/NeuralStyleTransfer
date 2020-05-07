import React from "react";
import Animation from "../Animation/Animation";
import { Link } from "react-router-dom";

export default function WelcomePage(props) {
  return (
    <div>
      <div className='uk-container uk-container-small uk-margin-top'>
        <h2 className='uk-text-center'>
          Neural Style Transfer Image Generator
        </h2>
      </div>
      <div className='uk-container uk-container-large uk-margin'>
        <p className='uk-text-center'>
          Generate your own art using Neural Style Transfer. Choose an image of
          art you enjoy and apply the style from it to a picture of your
          choosing.
        </p>
      </div>

      <Animation />
      <div className='uk-flex uk-flex-center uk-margin-large'>
        <Link
          className='uk-button uk-button-default uk-button-large uk-button-primary'
          to='/style'>
          Get Started
        </Link>
      </div>
    </div>
  );
}
