import React, { useState } from 'react';
import './App.css';
import Logo from './components/Logo';
import OrientationViewer from './components/OrientationViewer';
import SignalPlots from './components/SignalPlots';
import FileUpload from './components/FileUpload';
import FilterControls from './components/FilterControls';

function App() {
  const [imuData, setImuData] = useState(null);
  const [complementaryResults, setComplementaryResults] = useState(null);
  const [ekfResults, setEKFResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileProcessed = (data, compResults, ekfResults) => {
    setImuData(data);
    setComplementaryResults(compResults);
    setEKFResults(ekfResults);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <Logo size="large" />
          <div className="header-text">
            <h1>INVARI</h1>
            <p>Sensor Fusion Platform for IMU Data</p>
          </div>
        </div>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <FileUpload
            onProcessed={handleFileProcessed}
            isProcessing={isProcessing}
            setIsProcessing={setIsProcessing}
          />
        </div>

        {imuData && (
          <>
            <div className="visualization-section">
              <div className="viewer-container">
                <h2>3D Orientation Viewer</h2>
                <OrientationViewer
                  quaternions={
                    ekfResults?.quaternions || complementaryResults?.quaternions
                  }
                  timestamps={
                    ekfResults?.timestamps || complementaryResults?.timestamps
                  }
                />
              </div>
            </div>

            <div className="plots-section">
              <SignalPlots
                imuData={imuData}
                complementaryResults={complementaryResults}
                ekfResults={ekfResults}
              />
            </div>
          </>
        )}
      </main>
      
      <footer className="App-footer">
        <div className="footer-content">
          <p>© 2024 INVARI - Sensor Fusion Platform</p>
          <p className="footer-tagline">Precision orientation estimation through advanced sensor fusion</p>
        </div>
      </footer>
    </div>
  );
}

export default App;

