import React, { useRef, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import './OrientationViewer.css';

function OrientationCube({ quaternion }) {
  const meshRef = useRef();

  useEffect(() => {
    if (meshRef.current && quaternion) {
      const q = new THREE.Quaternion(
        quaternion[1], // x
        quaternion[2], // y
        quaternion[3], // z
        quaternion[0]  // w
      );
      meshRef.current.setRotationFromQuaternion(q);
    }
  }, [quaternion]);

  return (
    <group ref={meshRef}>
      <mesh>
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial 
          color="#00d4ff" 
          metalness={0.7}
          roughness={0.3}
          emissive="#001122"
          emissiveIntensity={0.2}
        />
      </mesh>
      {/* Axes */}
      <arrowHelper args={[new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1.5, 0xff4444]} />
      <arrowHelper args={[new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 1.5, 0x00ff88]} />
      <arrowHelper args={[new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1.5, 0x4a9eff]} />
    </group>
  );
}

function OrientationViewer({ quaternions, timestamps }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isPlaying && quaternions && quaternions.length > 0) {
      intervalRef.current = setInterval(() => {
        setCurrentIndex((prev) => {
          if (prev < quaternions.length - 1) {
            return prev + 1;
          } else {
            setIsPlaying(false);
            return prev;
          }
        });
      }, 50); // 20 FPS playback
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, quaternions]);

  const handlePlay = () => {
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleReset = () => {
    setCurrentIndex(0);
    setIsPlaying(false);
  };

  const currentQuaternion = quaternions && quaternions[currentIndex]
    ? quaternions[currentIndex]
    : [1, 0, 0, 0];

  const currentTime = timestamps && timestamps[currentIndex]
    ? timestamps[currentIndex].toFixed(2)
    : '0.00';

  return (
    <div className="orientation-viewer">
      <Canvas>
        <PerspectiveCamera makeDefault position={[5, 5, 5]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#00d4ff" />
        <Grid args={[10, 10]} cellColor="#3a4147" sectionColor="#2d3439" />
        <OrientationCube quaternion={currentQuaternion} />
        <OrbitControls 
          enableDamping={true}
          dampingFactor={0.05}
          minDistance={3}
          maxDistance={15}
        />
      </Canvas>
      <div className="viewer-controls">
        <button onClick={handlePlay} disabled={isPlaying}>
          Play
        </button>
        <button onClick={handlePause} disabled={!isPlaying}>
          Pause
        </button>
        <button onClick={handleReset}>Reset</button>
        <span className="time-display">Time: {currentTime}s</span>
        <input
          type="range"
          min="0"
          max={quaternions ? quaternions.length - 1 : 0}
          value={currentIndex}
          onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
          className="time-slider"
        />
      </div>
    </div>
  );
}

export default OrientationViewer;

