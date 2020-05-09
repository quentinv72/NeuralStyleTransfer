import React from "react";

export default function StyleImage(props) {
  if (props.clicked === true) {
    return (
      <img
        src={props.src}
        alt='style'
        id={props.id}
        className='choices'
        style={{
          width: "256px",
          border: "3px solid black",
          boxShadow: "2px 2px 2px 1px rgba(0, 0, 0, 0.2)",
        }}
        onClick={props.onClick}
      />
    );
  } else {
    return (
      <img
        src={props.src}
        alt='style'
        id={props.id}
        className='choices'
        style={{
          width: "256px",

          border: "3px solid white",
        }}
        onClick={props.onClick}
      />
    );
  }
}
