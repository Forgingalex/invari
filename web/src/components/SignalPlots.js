import React from 'react';
import Plot from 'react-plotly.js';
import './SignalPlots.css';

function SignalPlots({ imuData, complementaryResults, ekfResults }) {
  if (!imuData) return null;

  const plotLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#b0b8c0', family: 'inherit' },
    xaxis: {
      gridcolor: '#3a4147',
      linecolor: '#3a4147',
      zerolinecolor: '#3a4147',
    },
    yaxis: {
      gridcolor: '#3a4147',
      linecolor: '#3a4147',
      zerolinecolor: '#3a4147',
    },
  };

  const rawAccelTrace = {
    x: imuData.timestamps,
    y: imuData.accelerometer.map(a => a[0]),
    type: 'scatter',
    mode: 'lines',
    name: 'ax',
    line: { color: '#ff4444', width: 2 },
  };

  const rawGyroTrace = {
    x: imuData.timestamps,
    y: imuData.gyroscope.map(g => g[0]),
    type: 'scatter',
    mode: 'lines',
    name: 'gx',
    line: { color: '#4a9eff', width: 2 },
  };

  const compRollTrace = complementaryResults ? {
    x: complementaryResults.timestamps,
    y: complementaryResults.orientations.map(o => o[0] * 180 / Math.PI),
    type: 'scatter',
    mode: 'lines',
    name: 'Complementary Roll',
    line: { color: '#00ff88', width: 2 },
  } : null;

  const ekfRollTrace = ekfResults ? {
    x: ekfResults.timestamps,
    y: ekfResults.orientations.map(o => o[0] * 180 / Math.PI),
    type: 'scatter',
    mode: 'lines',
    name: 'EKF Roll',
    line: { color: '#00d4ff', width: 2, dash: 'dash' },
  } : null;

  return (
    <div className="signal-plots">
      <h2>Signal Analysis</h2>
      
      <div className="plot-container">
        <h3>Raw Accelerometer Signal</h3>
        <Plot
          data={[rawAccelTrace]}
          layout={{
            ...plotLayout,
            title: { text: 'Accelerometer X-axis', font: { color: '#ffffff', size: 16 } },
            xaxis: { ...plotLayout.xaxis, title: 'Time (s)' },
            yaxis: { ...plotLayout.yaxis, title: 'Acceleration (m/s²)' },
            height: 300,
          }}
          config={{ displayModeBar: false }}
        />
      </div>

      <div className="plot-container">
        <h3>Raw Gyroscope Signal</h3>
        <Plot
          data={[rawGyroTrace]}
          layout={{
            ...plotLayout,
            title: { text: 'Gyroscope X-axis', font: { color: '#ffffff', size: 16 } },
            xaxis: { ...plotLayout.xaxis, title: 'Time (s)' },
            yaxis: { ...plotLayout.yaxis, title: 'Angular Velocity (rad/s)' },
            height: 300,
          }}
          config={{ displayModeBar: false }}
        />
      </div>

      <div className="plot-container">
        <h3>Fused Orientation Comparison</h3>
        <Plot
          data={[compRollTrace, ekfRollTrace].filter(Boolean)}
          layout={{
            ...plotLayout,
            title: { text: 'Roll Angle Comparison', font: { color: '#ffffff', size: 16 } },
            xaxis: { ...plotLayout.xaxis, title: 'Time (s)' },
            yaxis: { ...plotLayout.yaxis, title: 'Roll (deg)' },
            height: 400,
            legend: {
              bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#b0b8c0' },
            },
          }}
          config={{ displayModeBar: false }}
        />
      </div>
    </div>
  );
}

export default SignalPlots;

