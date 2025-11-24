import React from 'react';
import './Logo.css';

function Logo({ size = 'medium' }) {
  return (
    <div className={`logo logo-${size}`}>
      <svg
        viewBox="0 0 120 120"
        xmlns="http://www.w3.org/2000/svg"
        className="logo-svg"
      >
        {/* Outer ring - represents sensor fusion */}
        <circle
          cx="60"
          cy="60"
          r="50"
          fill="none"
          stroke="url(#gradient1)"
          strokeWidth="3"
          opacity="0.8"
        />
        
        {/* Inner 3D axes - represents IMU orientation */}
        {/* X-axis (red) */}
        <line
          x1="60"
          y1="60"
          x2="85"
          y2="60"
          stroke="#ff4444"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <polygon
          points="85,60 80,55 80,65"
          fill="#ff4444"
        />
        
        {/* Y-axis (green) */}
        <line
          x1="60"
          y1="60"
          x2="60"
          y2="35"
          stroke="#00ff88"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <polygon
          points="60,35 55,40 65,40"
          fill="#00ff88"
        />
        
        {/* Z-axis (blue) - perspective */}
        <line
          x1="60"
          y1="60"
          x2="75"
          y2="45"
          stroke="#4a9eff"
          strokeWidth="4"
          strokeLinecap="round"
          opacity="0.7"
        />
        <circle
          cx="75"
          cy="45"
          r="3"
          fill="#4a9eff"
        />
        
        {/* Center point - fusion core */}
        <circle
          cx="60"
          cy="60"
          r="6"
          fill="url(#gradient2)"
        />
        
        {/* Rotating particles - represents data flow */}
        <circle
          cx="60"
          cy="10"
          r="3"
          fill="#00d4ff"
          opacity="0.8"
          className="particle particle-1"
        />
        <circle
          cx="110"
          cy="60"
          r="3"
          fill="#00d4ff"
          opacity="0.8"
          className="particle particle-2"
        />
        <circle
          cx="60"
          cy="110"
          r="3"
          fill="#00d4ff"
          opacity="0.8"
          className="particle particle-3"
        />
        <circle
          cx="10"
          cy="60"
          r="3"
          fill="#00d4ff"
          opacity="0.8"
          className="particle particle-4"
        />
        
        {/* Gradients */}
        <defs>
          <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00d4ff" />
            <stop offset="100%" stopColor="#4a9eff" />
          </linearGradient>
          <radialGradient id="gradient2" cx="50%" cy="50%">
            <stop offset="0%" stopColor="#00d4ff" />
            <stop offset="100%" stopColor="#4a9eff" />
          </radialGradient>
        </defs>
      </svg>
      <span className="logo-text">INVARI</span>
    </div>
  );
}

export default Logo;

