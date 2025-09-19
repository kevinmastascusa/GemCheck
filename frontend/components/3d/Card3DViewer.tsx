'use client';

import React, { Suspense, useRef, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  useTexture, 
  Float, 
  Environment,
  ContactShadows,
  Html,
  Sparkles
} from '@react-three/drei';
import { 
  EffectComposer, 
  Bloom, 
  ChromaticAberration, 
  DepthOfField,
  Glitch,
  Noise
} from '@react-three/postprocessing';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';
import { motion, AnimatePresence } from 'framer-motion';
import { BlendFunction, GlitchMode } from 'postprocessing';
import { cn } from '../../lib/utils';

// Types
interface Card3DViewerProps {
  frontImage: string;
  backImage?: string;
  cardName: string;
  grade?: number;
  analysis?: {
    centering: number;
    edges: number;
    corners: number;
    surface: number;
  };
  defects?: Array<{
    type: 'scratch' | 'edge_wear' | 'corner_damage' | 'surface_defect';
    position: { x: number; y: number };
    severity: number;
  }>;
  className?: string;
  onAnalysisComplete?: (analysis: any) => void;
}

// 3D Card Component
function Card3D({ 
  frontImage, 
  backImage, 
  analysis, 
  defects = [],
  grade 
}: Omit<Card3DViewerProps, 'className' | 'cardName' | 'onAnalysisComplete'>) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);
  
  // Load textures
  const [frontTexture, backTexture] = useTexture([
    frontImage || '/placeholder-card-front.jpg',
    backImage || '/placeholder-card-back.jpg'
  ]);
  
  // Configure textures for card-like appearance
  useEffect(() => {
    [frontTexture, backTexture].forEach(texture => {
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.flipY = false;
    });
  }, [frontTexture, backTexture]);
  
  // Animation springs
  const { scale, rotation } = useSpring({
    scale: hovered ? 1.1 : clicked ? 1.05 : 1,
    rotation: clicked ? [0, Math.PI, 0] : [0, 0, 0],
    config: { tension: 300, friction: 40 }
  });
  
  // Hover animation
  useFrame((state) => {
    if (meshRef.current && !clicked) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      meshRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.02;
    }
  });
  
  // Card dimensions (standard trading card ratio)
  const cardWidth = 2.5;
  const cardHeight = 3.5;
  const cardThickness = 0.02;
  
  return (
    <Float speed={1.4} rotationIntensity={0.2} floatIntensity={0.3}>
      <animated.group 
        ref={groupRef}
        scale={scale}
        rotation={rotation as any}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onClick={() => setClicked(!clicked)}
      >
        {/* Card Front */}
        <mesh
          ref={meshRef}
          position={[0, 0, cardThickness / 2]}
        >
          <boxGeometry args={[cardWidth, cardHeight, cardThickness]} />
          <meshStandardMaterial
            map={frontTexture}
            roughness={0.1}
            metalness={0.1}
            transparent
            side={THREE.FrontSide}
          />
        </mesh>
        
        {/* Card Back */}
        <mesh
          position={[0, 0, -cardThickness / 2]}
        >
          <boxGeometry args={[cardWidth, cardHeight, cardThickness]} />
          <meshStandardMaterial
            map={backTexture}
            roughness={0.1}
            metalness={0.1}
            transparent
            side={THREE.BackSide}
          />
        </mesh>
        
        {/* Card Edges */}
        <mesh
          position={[0, 0, 0]}
        >
          <boxGeometry args={[cardWidth + 0.01, cardHeight + 0.01, cardThickness * 1.5]} />
          <meshStandardMaterial
            color="#f0f0f0"
            roughness={0.8}
            metalness={0.0}
            transparent
            opacity={0.9}
          />
        </mesh>
        
        {/* Holographic Overlay */}
        {grade && grade >= 9 && (
          <mesh
            position={[0, 0, cardThickness + 0.01]}
          >
            <planeGeometry args={[cardWidth + 0.1, cardHeight + 0.1]} />
            <meshBasicMaterial
              color="#00FFFF"
              transparent
              opacity={0.1}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        )}
        
        {/* Defect Indicators */}
        {defects.map((defect, index) => (
          <mesh
            key={index}
            position={[
              (defect.position.x - 0.5) * cardWidth,
              (0.5 - defect.position.y) * cardHeight,
              cardThickness + 0.02
            ]}
          >
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshBasicMaterial
              color={
                defect.type === 'scratch' ? '#FF006E' :
                defect.type === 'edge_wear' ? '#FFD60A' :
                defect.type === 'corner_damage' ? '#FF9500' :
                '#FF006E'
              }
              transparent
              opacity={0.8}
            />
          </mesh>
        ))}
        
        {/* Perfect 10 Sparkles */}
        {grade === 10 && (
          <Sparkles
            count={50}
            scale={[cardWidth * 2, cardHeight * 2, 1]}
            size={2}
            speed={0.5}
            opacity={0.6}
            color="#FFD700"
          />
        )}
      </animated.group>
    </Float>
  );
}

// Magic Lens Component
function MagicLens({ position, visible }: { position: [number, number], visible: boolean }) {
  if (!visible) return null;
  
  return (
    <Html position={[position[0], position[1], 2]}>
      <div className="relative">
        <div className={cn(
          'w-32 h-32 rounded-full border-4 border-accent-cyan',
          'bg-black/20 backdrop-blur-md',
          'shadow-glow animate-pulse',
          'flex items-center justify-center'
        )}>
          <div className="w-24 h-24 rounded-full border-2 border-accent-cyan/50">
            <div className="w-full h-full rounded-full bg-gradient-to-br from-transparent to-accent-cyan/10" />
          </div>
        </div>
        <div className="absolute top-1/2 left-1/2 w-0.5 h-8 bg-accent-cyan transform -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute top-1/2 left-1/2 w-8 h-0.5 bg-accent-cyan transform -translate-x-1/2 -translate-y-1/2" />
      </div>
    </Html>
  );
}

// Scan Effect Component
function ScanEffect({ isScanning }: { isScanning: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current && isScanning) {
      meshRef.current.position.y = Math.sin(state.clock.elapsedTime * 3) * 2;
    }
  });
  
  if (!isScanning) return null;
  
  return (
    <mesh
      ref={meshRef}
      position={[0, -2, 1]}
    >
      <planeGeometry args={[6, 0.1]} />
      <meshBasicMaterial
        color="#00FFFF"
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

// Main Component
export default function Card3DViewer({
  frontImage,
  backImage,
  cardName,
  grade,
  analysis,
  defects = [],
  className,
  onAnalysisComplete
}: Card3DViewerProps) {
  const [isScanning, setIsScanning] = useState(false);
  const [showMagicLens, setShowMagicLens] = useState(false);
  const [lensPosition, setLensPosition] = useState<[number, number]>([0, 0]);
  const [cameraMode, setCameraMode] = useState<'orbit' | 'cinematic'>('orbit');
  
  // Handle magic lens
  const handlePointerMove = (event: React.PointerEvent) => {
    if (showMagicLens) {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width - 0.5) * 4;
      const y = (-(event.clientY - rect.top) / rect.height + 0.5) * 4;
      setLensPosition([x, y]);
    }
  };
  
  // Scan animation
  const startScan = () => {
    setIsScanning(true);
    setTimeout(() => {
      setIsScanning(false);
      if (onAnalysisComplete) {
        onAnalysisComplete({
          centering: 87.5,
          edges: 82.1,
          corners: 91.3,
          surface: 89.7,
          confidence: 83.2
        });
      }
    }, 3000);
  };
  
  return (
    <div className={cn('relative w-full h-full', className)}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 5], fov: 45 }}
        className="w-full h-full"
        onPointerMove={handlePointerMove}
      >
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <spotLight 
          position={[10, 10, 10]} 
          intensity={1} 
          angle={0.3} 
          penumbra={1}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <pointLight position={[-10, -10, -10]} intensity={0.3} color="#8B5CF6" />
        <pointLight position={[10, -10, 10]} intensity={0.3} color="#EC4899" />
        
        {/* Environment */}
        <Environment preset="studio" />
        
        {/* 3D Card */}
        <Suspense fallback={null}>
          <Card3D
            frontImage={frontImage}
            backImage={backImage}
            analysis={analysis}
            defects={defects}
            grade={grade}
          />
        </Suspense>
        
        {/* Effects */}
        <ScanEffect isScanning={isScanning} />
        <MagicLens position={lensPosition} visible={showMagicLens} />
        
        {/* Contact Shadows */}
        <ContactShadows
          position={[0, -2, 0]}
          opacity={0.4}
          scale={10}
          blur={1}
          far={4}
        />
        
        {/* Controls */}
        {cameraMode === 'orbit' && (
          <OrbitControls
            enablePan={false}
            enableZoom={true}
            enableRotate={true}
            minDistance={3}
            maxDistance={8}
            minPolarAngle={0}
            maxPolarAngle={Math.PI / 2}
            autoRotate={false}
          />
        )}
        
        {/* Post-processing Effects */}
        <EffectComposer>
          <Bloom
            intensity={0.3}
            luminanceThreshold={0.9}
            luminanceSmoothing={0.9}
            blendFunction={BlendFunction.ADDITIVE}
          />
          <ChromaticAberration
            offset={[0.001, 0.001]}
            blendFunction={BlendFunction.NORMAL}
          />
          <DepthOfField
            focusDistance={0.02}
            focalLength={0.05}
            bokehScale={1}
          />
          <Noise opacity={0.05} />
        </EffectComposer>
      </Canvas>
      
      {/* Control Panel */}
      <div className="absolute bottom-4 left-4 space-y-2">
        <motion.button
          onClick={() => setShowMagicLens(!showMagicLens)}
          className={cn(
            'glass-morphism px-4 py-2 rounded-full text-sm font-medium',
            'magnetic-button ripple',
            showMagicLens ? 'text-accent-cyan shadow-glow' : 'text-white'
          )}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          🔍 Magic Lens
        </motion.button>
        
        <motion.button
          onClick={startScan}
          disabled={isScanning}
          className={cn(
            'glass-morphism px-4 py-2 rounded-full text-sm font-medium',
            'magnetic-button ripple',
            isScanning ? 'text-accent-cyan animate-pulse' : 'text-white'
          )}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isScanning ? '📡 Scanning...' : '🔬 Analyze'}
        </motion.button>
        
        <motion.button
          onClick={() => setCameraMode(cameraMode === 'orbit' ? 'cinematic' : 'orbit')}
          className="glass-morphism px-4 py-2 rounded-full text-sm font-medium magnetic-button ripple text-white"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📹 {cameraMode === 'orbit' ? 'Cinematic' : 'Orbit'}
        </motion.button>
      </div>
      
      {/* Grade Display */}
      {grade && (
        <div className="absolute top-4 right-4">
          <motion.div
            className={cn(
              'glass-morphism rounded-full w-16 h-16 flex items-center justify-center',
              'font-display font-bold text-2xl',
              grade === 10 ? 'text-gold-shimmer shadow-glow-gold' :
              grade >= 9 ? 'text-accent-cyan shadow-glow' :
              grade >= 7 ? 'text-green-400' :
              'text-yellow-400'
            )}
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ 
              type: 'spring',
              stiffness: 300,
              damping: 20,
              delay: 1
            }}
          >
            {grade}
          </motion.div>
        </div>
      )}
      
      {/* Analysis Progress */}
      <AnimatePresence>
        {isScanning && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center bg-black/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="glass-morphism rounded-lg p-6 text-center">
              <div className="dna-spinner mx-auto mb-4" />
              <div className="text-lg font-display text-holographic mb-2">
                Analyzing Card Condition
              </div>
              <div className="text-sm text-gray-400">
                Please wait while we process your card...
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}