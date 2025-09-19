'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { cn } from '../../lib/utils';

interface HeatMapData {
  x: number;
  y: number;
  intensity: number;
  type: 'good' | 'warning' | 'error';
}

interface HeatMapOverlayProps {
  width: number;
  height: number;
  data: HeatMapData[];
  mode: 'condition' | 'defects' | 'analysis' | 'xray';
  isAnimating?: boolean;
  showGrid?: boolean;
  className?: string;
}

// 3D Heat Map Shader Material
const heatMapVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const heatMapFragmentShader = `
  uniform float time;
  uniform vec2 resolution;
  uniform sampler2D heatTexture;
  uniform float intensity;
  uniform int mode;
  varying vec2 vUv;

  vec3 heatMapColor(float value) {
    // Heat map color gradient
    vec3 color1 = vec3(0.0, 0.0, 1.0); // Blue (cold)
    vec3 color2 = vec3(0.0, 1.0, 0.0); // Green (warm)
    vec3 color3 = vec3(1.0, 1.0, 0.0); // Yellow (hot)
    vec3 color4 = vec3(1.0, 0.0, 0.0); // Red (very hot)
    
    if (value < 0.33) {
      return mix(color1, color2, value * 3.0);
    } else if (value < 0.66) {
      return mix(color2, color3, (value - 0.33) * 3.0);
    } else {
      return mix(color3, color4, (value - 0.66) * 3.0);
    }
  }

  vec3 xrayColor(float value) {
    // X-ray mode colors
    return vec3(0.0, 1.0, 1.0) * value + vec3(0.2, 0.2, 0.8) * (1.0 - value);
  }

  void main() {
    vec2 uv = vUv;
    float heatValue = texture2D(heatTexture, uv).r;
    
    // Add scanning effect
    float scanLine = sin(uv.y * 50.0 + time * 5.0) * 0.1 + 0.9;
    heatValue *= scanLine;
    
    vec3 color;
    if (mode == 3) { // X-ray mode
      color = xrayColor(heatValue);
    } else {
      color = heatMapColor(heatValue);
    }
    
    // Add pulsing effect
    float pulse = sin(time * 2.0) * 0.1 + 0.9;
    color *= pulse;
    
    gl_FragColor = vec4(color, heatValue * intensity);
  }
`;

// 3D Heat Map Component
function HeatMap3D({ 
  data, 
  mode, 
  isAnimating, 
  width, 
  height 
}: Omit<HeatMapOverlayProps, 'className' | 'showGrid'>) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  // Create heat map texture
  const heatTexture = React.useMemo(() => {
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');
    
    if (context) {
      // Create gradient background
      const gradient = context.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
      gradient.addColorStop(0, 'rgba(255, 0, 0, 0.8)');
      gradient.addColorStop(0.3, 'rgba(255, 255, 0, 0.6)');
      gradient.addColorStop(0.6, 'rgba(0, 255, 0, 0.4)');
      gradient.addColorStop(1, 'rgba(0, 0, 255, 0.2)');
      
      context.fillStyle = gradient;
      context.fillRect(0, 0, size, size);
      
      // Add data points
      data.forEach(point => {
        const x = point.x * size;
        const y = point.y * size;
        const radius = point.intensity * 20;
        
        const pointGradient = context.createRadialGradient(x, y, 0, x, y, radius);
        pointGradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        pointGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        context.fillStyle = pointGradient;
        context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
      });
    }
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  }, [data]);
  
  useFrame(({ clock }) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = clock.elapsedTime;
    }
  });
  
  return (
    <mesh position={[0, 0, 0.01]}>
      <planeGeometry args={[width, height]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={heatMapVertexShader}
        fragmentShader={heatMapFragmentShader}
        uniforms={{
          time: { value: 0 },
          resolution: { value: new THREE.Vector2(width, height) },
          heatTexture: { value: heatTexture },
          intensity: { value: 0.7 },
          mode: { value: mode === 'xray' ? 3 : mode === 'condition' ? 0 : mode === 'defects' ? 1 : 2 }
        }}
        transparent
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

export default function HeatMapOverlay({
  width,
  height,
  data,
  mode,
  isAnimating = false,
  showGrid = true,
  className
}: HeatMapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [animationFrame, setAnimationFrame] = useState(0);
  
  // 2D Canvas heat map fallback
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    canvas.width = width;
    canvas.height = height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid if enabled
    if (showGrid) {
      ctx.strokeStyle = 'rgba(102, 126, 234, 0.2)';
      ctx.lineWidth = 1;
      
      const gridSize = 20;
      for (let x = 0; x <= width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      
      for (let y = 0; y <= height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }
    
    // Draw heat map points
    data.forEach((point, index) => {
      const x = point.x * width;
      const y = point.y * height;
      const radius = Math.max(10, point.intensity * 50);
      
      // Create radial gradient based on type and mode
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      
      let colors: [string, string];
      switch (mode) {
        case 'condition':
          colors = point.type === 'good' ? 
            ['rgba(0, 245, 160, 0.8)', 'rgba(0, 245, 160, 0)'] :
            point.type === 'warning' ?
            ['rgba(255, 214, 10, 0.8)', 'rgba(255, 214, 10, 0)'] :
            ['rgba(255, 0, 110, 0.8)', 'rgba(255, 0, 110, 0)'];
          break;
        case 'defects':
          colors = ['rgba(255, 0, 110, 0.9)', 'rgba(255, 0, 110, 0)'];
          break;
        case 'xray':
          colors = ['rgba(0, 255, 255, 0.7)', 'rgba(0, 255, 255, 0)'];
          break;
        default:
          colors = ['rgba(102, 126, 234, 0.6)', 'rgba(102, 126, 234, 0)'];
      }
      
      gradient.addColorStop(0, colors[0]);
      gradient.addColorStop(1, colors[1]);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      
      // Add pulsing animation
      if (isAnimating) {
        const pulseRadius = radius * (1 + Math.sin(Date.now() * 0.005 + index) * 0.3);
        const pulseGradient = ctx.createRadialGradient(x, y, 0, x, y, pulseRadius);
        pulseGradient.addColorStop(0, colors[0].replace('0.8', '0.4').replace('0.9', '0.4').replace('0.7', '0.3').replace('0.6', '0.3'));
        pulseGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        ctx.fillStyle = pulseGradient;
        ctx.beginPath();
        ctx.arc(x, y, pulseRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
  }, [width, height, data, mode, showGrid, isAnimating, animationFrame]);
  
  // Animation loop
  useEffect(() => {
    if (!isAnimating) return;
    
    const animate = () => {
      setAnimationFrame(prev => prev + 1);
      requestAnimationFrame(animate);
    };
    
    const animationId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationId);
  }, [isAnimating]);
  
  return (
    <div className={cn('relative', className)}>
      {/* 2D Canvas Fallback */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ mixBlendMode: mode === 'xray' ? 'screen' : 'multiply' }}
      />
      
      {/* Data Point Labels */}
      <AnimatePresence>
        {data.map((point, index) => (
          <motion.div
            key={index}
            className="absolute pointer-events-none"
            style={{
              left: `${point.x * 100}%`,
              top: `${point.y * 100}%`,
              transform: 'translate(-50%, -50%)',
            }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className={cn(
              'glass-morphism rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold',
              point.type === 'good' ? 'text-green-400' :
              point.type === 'warning' ? 'text-yellow-400' :
              'text-red-400'
            )}>
              {Math.round(point.intensity * 100)}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
      
      {/* Scanning Line Effect */}
      {isAnimating && mode !== 'xray' && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div
            className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-accent-cyan to-transparent"
            animate={{
              top: ['0%', '100%', '0%']
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'linear'
            }}
            style={{ boxShadow: '0 0 10px rgba(0, 255, 255, 0.8)' }}
          />
        </div>
      )}
      
      {/* X-Ray Grid Overlay */}
      {mode === 'xray' && (
        <div className="absolute inset-0 pointer-events-none">
          <svg className="w-full h-full">
            <defs>
              <pattern id="xray-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path 
                  d="M 20 0 L 0 0 0 20" 
                  fill="none" 
                  stroke="rgba(0, 255, 255, 0.3)" 
                  strokeWidth="1"
                />
              </pattern>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge> 
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>
            <rect 
              width="100%" 
              height="100%" 
              fill="url(#xray-grid)"
              filter="url(#glow)"
            />
          </svg>
        </div>
      )}
    </div>
  );
}