import React from "react";
import { usePromiseTracker } from "react-promise-tracker";
import { css } from "@emotion/core";
import { PacmanLoader } from "react-spinners";
const DownloadPage = (props) => {
  const { promiseInProgress } = usePromiseTracker();
  return (
    <div
      className='uk-flex uk-flex-center uk-flex-middle'
      uk-height-viewport='offset-top: true'>
      {promiseInProgress === true ? (
        <div>
          <h2>
            Your image is being generated <span role='img'>😊</span>
          </h2>
          <div className='uk-flex uk-flex-center'>
            <PacmanLoader color={"#FFD500"} />
          </div>
        </div>
      ) : (
        <div>
          <h2 className='uk-text-center'>
            Here is your image <span role='img'>😊</span>
          </h2>
          <div className='uk-flex uk-flex-center uk-margin'>
            <img src={props.src} alt='generated' />
          </div>
          <div className='uk-flex uk-flex-center uk-margin'>
            <a
              href={props.src}
              download
              className='uk-button uk-button-default uk-button-large uk-button-primary'>
              Download
            </a>
          </div>
        </div>
      )}
    </div>
  );
};

export default DownloadPage;
