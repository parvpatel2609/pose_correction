import React, { useEffect, useRef, useState } from "react";

import Webcam from "react-webcam";
import axios from "axios";

const videoConstraints = {
  width: 540,
  facingMode: "environment",
};

export default function Camera() {
  const webcamRef = useRef(null);

  const [url, setUrl] = useState(null);

  const capturePhoto = React.useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setUrl(imageSrc);
  }, [webcamRef]);

  const onUserMedia = (e) => {
    console.log(e);
  };

  const handleSubmit = async () => {
    try {
      // e.preventDefault();
      // Define the data you want to send to the Flask backend

      // Make a POST request to your Flask API
      const res = await axios.get(
        "http://localhost:5000/post_example"
      );

      console.log(res);
    } catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    handleSubmit();
  }, []);

  return (
    <div>
      {/* <Webcam
        ref={webcamRef}
        audio={true}
        screenshotFormat="image/png"
        videoConstraints={videoConstraints}
        onUserMedia={onUserMedia}
        mirrored={true}
      /> */}
      <button onClick={capturePhoto}>Capture photo</button>
      <button onClick={() => setUrl(null)}>Refresh</button>
      <button onClick={handleSubmit}>Submit</button>
      {url && <img src={url} alt="ScreeShot" />}
    </div>
  );
}
