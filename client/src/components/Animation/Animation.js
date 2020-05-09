import React from "react";
import "./Animation.css";
import styleTransfer from "./Images/style_transfer.jpeg";
import dancing from "./Images/dancing.jpg";
import picasso from "./Images/picasso.jpg";

export default function Animation(props) {
  return (
    <div className='uk-flex uk-flex-center uk-margin-large'>
      <img src={dancing} alt='dancer' className='left' />

      <img src={picasso} alt='picasso' className='right' />
      <img src={styleTransfer} alt='style-transfer' className='middle' />
    </div>
  );
}
