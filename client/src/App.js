import React, { useState } from "react";
import WelcomePage from "./components/WelcomePage/WelcomePage";
import { BrowserRouter, Route, Switch } from "react-router-dom";
import "./App.css";
import StyleImages from "./components/StyleImages/StyleImages";
import ContentImage from "./components/ContentImage/ContentImage";
import DownloadPage from "./components/DownloadPage/DownloadPage";

function App() {
  const [styleImage, setStyleImage] = useState(null);
  const [generatedImage, setGenerateImg] = useState(null);

  const updateGenImg = (str) => {
    setGenerateImg(str);
  };

  const handleStylePick = (e) => {
    let image = e.target;
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    // Save Image as base64
    const base64Image = canvas.toDataURL("image/jpeg");
    setStyleImage(base64Image);
  };

  return (
    <BrowserRouter>
      <Switch>
        <Route path='/' exact>
          <div className='uk-container'>
            <WelcomePage />
          </div>
        </Route>
        <Route path='/style'>
          <StyleImages onClick={handleStylePick} />
        </Route>
        <Route path='/content'>
          <ContentImage styleImage={styleImage} genImg={updateGenImg} />
        </Route>
        <Route path='/download'>
          <DownloadPage src={generatedImage} />
        </Route>
      </Switch>
    </BrowserRouter>
  );
}

export default App;
