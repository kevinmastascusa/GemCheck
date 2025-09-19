'use client';

import React, { useRef, useState, useEffect } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';

interface BasicCard3DProps {
  grade?: number;
  cardImage?: string;
}

export default function BasicCard3D({ grade, cardImage }: BasicCard3DProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  
  // Card dimensions (slightly larger for better visibility)
  const cardWidth = 3;
  const cardHeight = 4;
  const cardThickness = 0.05;
  
  // Load texture when cardImage changes
  useEffect(() => {
    if (cardImage) {
      const loader = new THREE.TextureLoader();
      loader.load(
        cardImage,
        (loadedTexture) => {
          loadedTexture.flipY = false; // Prevent texture flipping
          setTexture(loadedTexture);
        },
        undefined,
        (error) => {
          console.error('Error loading card texture:', error);
          setTexture(null);
        }
      );
    } else {
      setTexture(null);
    }
  }, [cardImage]);
  
  // Simple rotation animation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.3;
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.8) * 0.1;
    }
  });

  return (
    <animated.group ref={groupRef} position={[0, 0, 0]}>
      {/* Card Front */}
      <mesh position={[0, 0, cardThickness / 2]}>
        <planeGeometry args={[cardWidth, cardHeight]} />
        <meshStandardMaterial 
          map={texture}
          color={texture ? "#ffffff" : "#1a1a2e"}
          roughness={0.1}
          metalness={texture ? 0.0 : 0.8}
        />
      </mesh>
      
      {/* Card Back */}
      <mesh position={[0, 0, -cardThickness / 2]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[cardWidth, cardHeight]} />
        <meshStandardMaterial 
          color="#16213e" 
          roughness={0.2}
          metalness={0.6}
        />
      </mesh>
      
      {/* Card Border/Frame - only show if we have an image */}
      {texture && (
        <mesh position={[0, 0, cardThickness / 2 + 0.01]}>
          <planeGeometry args={[cardWidth + 0.1, cardHeight + 0.1]} />
          <meshStandardMaterial 
            color="#00FFFF" 
            roughness={0.3}
            metalness={0.9}
            emissive="#004444"
            emissiveIntensity={0.2}
            transparent={true}
            opacity={0.3}
          />
        </mesh>
      )}
      
      {/* Grade indicator */}
      {grade && (
        <mesh position={[0, cardHeight/2 + 0.5, cardThickness/2 + 0.2]}>
          <sphereGeometry args={[0.3, 32, 32]} />
          <meshStandardMaterial 
            color={grade === 10 ? '#FFD700' : grade >= 9 ? '#00FFFF' : '#00F5A0'}
            roughness={0.1}
            metalness={0.9}
            emissive={grade === 10 ? '#442200' : grade >= 9 ? '#004444' : '#002200'}
            emissiveIntensity={0.3}
          />
        </mesh>
      )}
      
      {/* Grade number text (using a simple box as placeholder) */}
      {grade && (
        <mesh position={[0, cardHeight/2 + 0.5, cardThickness/2 + 0.35]}>
          <boxGeometry args={[0.4, 0.4, 0.05]} />
          <meshStandardMaterial 
            color="#000000"
            roughness={0.8}
          />
        </mesh>
      )}
    </animated.group>
  );
}