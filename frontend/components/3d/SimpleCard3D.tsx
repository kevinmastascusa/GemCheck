'use client';

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';

interface SimpleCard3DProps {
  frontImage: string;
  backImage?: string;
  grade?: number;
}

export default function SimpleCard3D({ frontImage, backImage, grade }: SimpleCard3DProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  
  // Card dimensions
  const cardWidth = 2.5;
  const cardHeight = 3.5;
  const cardThickness = 0.02;
  
  // Simple rotation animation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.02;
    }
  });

  return (
    <animated.group ref={groupRef}>
      {/* Card Front */}
      <mesh position={[0, 0, cardThickness / 2]}>
        <boxGeometry args={[cardWidth, cardHeight, cardThickness]} />
        <meshStandardMaterial color="#e0e0e0" />
      </mesh>
      
      {/* Card Back */}
      <mesh position={[0, 0, -cardThickness / 2]}>
        <boxGeometry args={[cardWidth, cardHeight, cardThickness]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>
      
      {/* Grade indicator */}
      {grade && (
        <mesh position={[0, cardHeight/2 + 0.3, 0]}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshStandardMaterial 
            color={grade === 10 ? '#FFD700' : grade >= 9 ? '#00FFFF' : '#00F5A0'} 
          />
        </mesh>
      )}
    </animated.group>
  );
}