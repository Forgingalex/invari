import React, { useState } from 'react';
import axios from 'axios';
import './FileUpload.css';

function FileUpload({ onProcessed, isProcessing, setIsProcessing }) {
  const [file, setFile] = useState(null);
  const [alpha, setAlpha] = useState(0.96);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file');
      return;
    }

    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Compare both filters
      const response = await axios.post(
        'http://localhost:8000/compare',
        formData,
        {
          params: { alpha },
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      const { complementary, ekf } = response.data;

      // Extract raw IMU data from file for visualization
      const fileReader = new FileReader();
      fileReader.onload = (e) => {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',');
        
        const data = {
          timestamps: [],
          accelerometer: [],
          gyroscope: [],
        };

        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',');
          if (values.length >= 7) {
            data.timestamps.push(parseFloat(values[0]));
            data.accelerometer.push([
              parseFloat(values[1]),
              parseFloat(values[2]),
              parseFloat(values[3]),
            ]);
            data.gyroscope.push([
              parseFloat(values[4]),
              parseFloat(values[5]),
              parseFloat(values[6]),
            ]);
          }
        }

        onProcessed(data, complementary, ekf);
      };

      fileReader.readAsText(file);
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Error processing file. Make sure the API server is running.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-upload">
      <h2>Upload IMU Data</h2>
      <div className="upload-controls">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          disabled={isProcessing}
        />
        <div className="alpha-control">
          <label>
            Complementary Filter Alpha:
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              disabled={isProcessing}
            />
          </label>
        </div>
        <button
          onClick={handleUpload}
          disabled={!file || isProcessing}
          className="upload-button"
        >
          {isProcessing ? 'Processing...' : 'Process IMU Data'}
        </button>
      </div>
      {file && (
        <p className="file-info">Selected: {file.name}</p>
      )}
    </div>
  );
}

export default FileUpload;

