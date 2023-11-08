import React, { useEffect, useState } from "react";

import Webcam from "react-webcam";

export default function AllCameras() {
  const [devices, setDevices] = useState([]);

  const handleDevices = React.useCallback(
    (mediaDevices) => {
      setDevices(mediaDevices.filter(({ kind }) => kind === "videoinput"));
    },
    [setDevices]
  );

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(handleDevices);
  }, [handleDevices]);

  return (
    <div>
      {devices.map((device, key) => (
        <div key={key}>
          <Webcam
            audio={false}
            height={720}
            width={1280}
            videoConstraints={{
              deviceId: device.deviceId,
            }}
          />
          {device.label || `Device ${key + 1}`}
        </div>
      ))}
    </div>
  );
}
