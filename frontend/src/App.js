import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

function App() {
  const webcamRef = useRef(null);
  const [existingImageFile, setExistingImageFile] = useState(null);
  const [existingPreview, setExistingPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = "http://localhost:8000/match_faces";

  const handleExistingChange = (e) => {
    const file = e.target.files[0];
    setExistingImageFile(file);
    setExistingPreview(URL.createObjectURL(file));
  };

  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new File([u8arr], filename, { type: mime });
  };

  const checkMatch = async () => {
    if (!existingImageFile) {
      alert("Upload an existing face first.");
      return;
    }

    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) {
      alert("Could not capture webcam photo.");
      return;
    }

    const newImageFile = dataURLtoFile(screenshot, "new.jpg");

    const formData = new FormData();
    formData.append("existing", existingImageFile);
    formData.append("new", newImageFile);

    setLoading(true);

    try {
      const response = await axios.post(API_URL, formData);
      setResult(response.data);
    } catch (err) {
      console.error(err);
      alert("API error — check console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Face Recognition Attendance</h1>

      <h3>1) Upload existing face photo</h3>
      <input type="file" accept="image/*" onChange={handleExistingChange} />
      {existingPreview && (
        <img
          src={existingPreview}
          alt="uploaded"
          style={{ width: 200, marginTop: 10, borderRadius: 8 }}
        />
      )}

      <h3 style={{ marginTop: 20 }}>2) Capture new face (Webcam)</h3>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={320}
      />

      <button 
        onClick={checkMatch} 
        disabled={loading}
        style={{
          marginTop: 20,
          padding: "10px 20px",
          fontSize: "18px",
          cursor: "pointer",
        }}
      >
        {loading ? "Checking..." : "Check Match"}
      </button>

      {result && (
        <div style={{ marginTop: 20 }}>
          {result.error && <p style={{ color: "orange" }}>Error: {result.error}</p>}

          {result.matched ? (
            <h2 style={{ color: "green" }}>✔ Face Matched — Present Marked</h2>
          ) : (
            <h2 style={{ color: "red" }}>✘ Face Not Matched</h2>
          )}

          {result.distance && <p>Distance: {result.distance.toFixed(4)}</p>}
        </div>
      )}
    </div>
  );
}

export default App;
