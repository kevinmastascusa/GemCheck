'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Canvas, useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { cn } from '../../lib/utils';

interface GradeDNAProps {
  scores: {
    centering: number;
    edges: number;
    corners: number;
    surface: number;
    overall?: number;
  };
  isAnimating?: boolean;
  showLabels?: boolean;
  size?: number;
  className?: string;
}

interface DNAPoint {
  angle: number;
  radius: number;
  label: string;
  value: number;
  color: string;
}

// 3D DNA Helix Component
function DNAHelix({ 
  scores, 
  isAnimating 
}: { 
  scores: GradeDNAProps['scores']; 
  isAnimating: boolean; 
}) {
  const groupRef = useRef<THREE.Group>(null);
  const [time, setTime] = useState(0);
  
  useFrame((state, delta) => {
    setTime(prev => prev + delta);
    if (groupRef.current && isAnimating) {
      groupRef.current.rotation.y = time * 0.5;
    }
  });
  
  const helixPoints = React.useMemo(() => {
    const points1: THREE.Vector3[] = [];
    const points2: THREE.Vector3[] = [];
    const height = 4;
    const radius = 1;
    const turns = 3;
    const segments = 50;
    
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const angle1 = t * Math.PI * 2 * turns;
      const angle2 = angle1 + Math.PI;
      const y = (t - 0.5) * height;
      
      // Modulate radius based on scores
      const scoreModulation = (
        (scores.centering / 100) * Math.sin(angle1) +
        (scores.edges / 100) * Math.sin(angle1 * 2) +
        (scores.corners / 100) * Math.sin(angle1 * 3) +
        (scores.surface / 100) * Math.sin(angle1 * 4)
      ) * 0.2;
      
      const r1 = radius + scoreModulation;
      const r2 = radius - scoreModulation;
      
      points1.push(new THREE.Vector3(
        Math.cos(angle1) * r1,
        y,
        Math.sin(angle1) * r1
      ));
      
      points2.push(new THREE.Vector3(
        Math.cos(angle2) * r2,
        y,
        Math.sin(angle2) * r2
      ));
    }
    
    return { points1, points2 };
  }, [scores]);
  
  const colors = {
    centering: '#00FFFF',
    edges: '#EC4899',
    corners: '#8B5CF6',
    surface: '#00F5A0'
  };
  
  return (
    <group ref={groupRef}>
      {/* DNA Strands */}
      <Line
        points={helixPoints.points1}
        color={colors.centering}
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      <Line
        points={helixPoints.points2}
        color={colors.surface}
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      
      {/* Connecting Rungs */}
      {helixPoints.points1.map((point1, index) => {
        if (index % 5 !== 0) return null;
        const point2 = helixPoints.points2[index];
        return (
          <Line
            key={index}
            points={[point1, point2]}
            color="#FFD700"
            lineWidth={2}
            transparent
            opacity={0.6}
          />
        );
      })}
      
      {/* Score Spheres */}
      {Object.entries(scores).map(([key, value], index) => {
        if (key === 'overall') return null;
        const angle = (index / 4) * Math.PI * 2;
        const radius = 2 + (value / 100) * 0.5;
        return (
          <Circle
            key={key}
            args={[0.1, 16]}
            position={[
              Math.cos(angle) * radius,
              Math.sin(time + index) * 0.5,
              Math.sin(angle) * radius
            ]}
          >
            <meshBasicMaterial 
              color={colors[key as keyof typeof colors]}
              transparent
              opacity={0.8}
            />
          </Circle>
        );
      })}
    </group>
  );
}

// 2D Radar Chart Component
function RadarChart({ scores, size, isAnimating }: {
  scores: GradeDNAProps['scores'];
  size: number;
  isAnimating: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const dataPoints: DNAPoint[] = React.useMemo(() => [
    {
      angle: 0,
      radius: scores.centering,
      label: 'Centering',
      value: scores.centering,
      color: '#00FFFF'
    },
    {
      angle: Math.PI / 2,
      radius: scores.edges,
      label: 'Edges',
      value: scores.edges,
      color: '#EC4899'
    },
    {
      angle: Math.PI,
      radius: scores.corners,
      label: 'Corners',
      value: scores.corners,
      color: '#8B5CF6'
    },
    {
      angle: 3 * Math.PI / 2,
      radius: scores.surface,
      label: 'Surface',
      value: scores.surface,
      color: '#00F5A0'
    }
  ], [scores]);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const centerX = size / 2;
    const centerY = size / 2;
    const maxRadius = size * 0.35;
    
    let animationTime = 0;
    
    const draw = () => {
      ctx.clearRect(0, 0, size, size);
      
      // Draw background circles
      ctx.strokeStyle = 'rgba(102, 126, 234, 0.2)';
      ctx.lineWidth = 1;
      for (let i = 1; i <= 5; i++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, (maxRadius * i) / 5, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      // Draw axes
      ctx.strokeStyle = 'rgba(102, 126, 234, 0.3)';
      ctx.lineWidth = 1;
      dataPoints.forEach((point) => {
        const endX = centerX + Math.cos(point.angle - Math.PI / 2) * maxRadius;
        const endY = centerY + Math.sin(point.angle - Math.PI / 2) * maxRadius;
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      });
      
      // Draw data polygon
      ctx.beginPath();
      dataPoints.forEach((point, index) => {
        const animatedRadius = isAnimating ? 
          (point.radius / 100) * maxRadius * (0.5 + 0.5 * Math.sin(animationTime * 0.02 + index)) :
          (point.radius / 100) * maxRadius;
          
        const x = centerX + Math.cos(point.angle - Math.PI / 2) * animatedRadius;
        const y = centerY + Math.sin(point.angle - Math.PI / 2) * animatedRadius;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();
      
      // Fill with gradient
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, maxRadius);
      gradient.addColorStop(0, 'rgba(102, 126, 234, 0.3)');
      gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.2)');
      gradient.addColorStop(1, 'rgba(236, 72, 153, 0.1)');
      
      ctx.fillStyle = gradient;
      ctx.fill();
      
      ctx.strokeStyle = '#667eea';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw data points
      dataPoints.forEach((point, index) => {
        const animatedRadius = isAnimating ? 
          (point.radius / 100) * maxRadius * (0.5 + 0.5 * Math.sin(animationTime * 0.02 + index)) :
          (point.radius / 100) * maxRadius;
          
        const x = centerX + Math.cos(point.angle - Math.PI / 2) * animatedRadius;
        const y = centerY + Math.sin(point.angle - Math.PI / 2) * animatedRadius;
        
        // Glowing point
        const pointGradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
        pointGradient.addColorStop(0, point.color);
        pointGradient.addColorStop(1, 'transparent');
        
        ctx.fillStyle = pointGradient;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Inner point
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
      
      if (isAnimating) {
        animationTime++;
        animationRef.current = requestAnimationFrame(draw);
      }
    };
    
    draw();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [dataPoints, size, isAnimating]);
  
  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className="absolute inset-0"
    />
  );
}

export default function GradeDNA({
  scores,
  isAnimating = false,
  showLabels = true,
  size = 300,
  className
}: GradeDNAProps) {
  const [mode, setMode] = useState<'2d' | '3d'>('2d');
  const [showDetails, setShowDetails] = useState(false);
  
  const overallScore = scores.overall || (
    (scores.centering + scores.edges + scores.corners + scores.surface) / 4
  );
  
  const getScoreGrade = (score: number) => {
    if (score >= 97) return { grade: 10, label: 'Gem Mint' };
    if (score >= 92) return { grade: 9, label: 'Mint' };
    if (score >= 85) return { grade: 8, label: 'NM-Mint' };
    if (score >= 78) return { grade: 7, label: 'Near Mint' };
    if (score >= 72) return { grade: 6, label: 'Excellent' };
    if (score >= 66) return { grade: 5, label: 'VG-EX' };
    if (score >= 60) return { grade: 4, label: 'Good' };
    if (score >= 54) return { grade: 3, label: 'Fair' };
    if (score >= 48) return { grade: 2, label: 'Poor' };
    return { grade: 1, label: 'Authentic' };
  };
  
  const predictedGrade = getScoreGrade(overallScore);
  
  return (
    <div className={cn('relative', className)}>
      {/* Mode Toggle */}
      <div className="absolute top-4 right-4 z-10">
        <motion.button
          onClick={() => setMode(mode === '2d' ? '3d' : '2d')}
          className="glass-morphism px-3 py-1 rounded-full text-xs font-medium magnetic-button"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {mode === '2d' ? '🧬 3D DNA' : '📊 2D Radar'}
        </motion.button>
      </div>
      
      {/* Main Visualization */}
      <div 
        className="relative cursor-pointer"
        style={{ width: size, height: size }}
        onClick={() => setShowDetails(!showDetails)}
      >
        {mode === '2d' ? (
          <>
            <RadarChart scores={scores} size={size} isAnimating={isAnimating} />
            
            {/* Labels */}
            {showLabels && (
              <>
                {/* Centering */}
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full">
                  <div className="glass-morphism px-2 py-1 rounded text-xs font-medium text-accent-cyan">
                    Centering: {scores.centering.toFixed(1)}
                  </div>
                </div>
                
                {/* Edges */}
                <div className="absolute top-1/2 right-0 transform translate-x-full -translate-y-1/2">
                  <div className="glass-morphism px-2 py-1 rounded text-xs font-medium text-accent-pink">
                    Edges: {scores.edges.toFixed(1)}
                  </div>
                </div>
                
                {/* Corners */}
                <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full">
                  <div className="glass-morphism px-2 py-1 rounded text-xs font-medium text-accent-purple">
                    Corners: {scores.corners.toFixed(1)}
                  </div>
                </div>
                
                {/* Surface */}
                <div className="absolute top-1/2 left-0 transform -translate-x-full -translate-y-1/2">
                  <div className="glass-morphism px-2 py-1 rounded text-xs font-medium text-status-success">
                    Surface: {scores.surface.toFixed(1)}
                  </div>
                </div>
              </>
            )}
          </>
        ) : (
          <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
            <ambientLight intensity={0.6} />
            <pointLight position={[10, 10, 10]} intensity={0.8} />
            <pointLight position={[-10, -10, -10]} intensity={0.4} color="#8B5CF6" />
            
            <DNAHelix scores={scores} isAnimating={isAnimating} />
          </Canvas>
        )}
        
        {/* Center Grade Display */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <motion.div
            className={cn(
              'glass-morphism rounded-full w-20 h-20 flex flex-col items-center justify-center',
              'font-display font-bold border-2',
              predictedGrade.grade === 10 ? 'border-gold-shimmer text-gold-shimmer' :
              predictedGrade.grade >= 9 ? 'border-accent-cyan text-accent-cyan' :
              predictedGrade.grade >= 7 ? 'border-status-success text-status-success' :
              'border-status-warning text-status-warning'
            )}
            animate={{
              scale: isAnimating ? [1, 1.1, 1] : 1,
              rotate: isAnimating ? [0, 5, -5, 0] : 0
            }}
            transition={{
              duration: 2,
              repeat: isAnimating ? Infinity : 0,
              ease: 'easeInOut'
            }}
          >
            <div className="text-xl">{predictedGrade.grade}</div>
            <div className="text-xs opacity-70">{overallScore.toFixed(1)}</div>
          </motion.div>
        </div>
      </div>
      
      {/* Detailed Breakdown */}
      <AnimatePresence>
        {showDetails && (
          <motion.div
            className="absolute top-full left-0 right-0 mt-4 glass-morphism rounded-lg p-4 z-20"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div className="text-sm font-medium mb-3 text-holographic">
              Grade DNA Analysis
            </div>
            
            <div className="space-y-2">
              {Object.entries(scores).map(([key, value]) => {
                if (key === 'overall') return null;
                
                const colors = {
                  centering: '#00FFFF',
                  edges: '#EC4899',
                  corners: '#8B5CF6',
                  surface: '#00F5A0'
                };
                
                const color = colors[key as keyof typeof colors];
                
                return (
                  <div key={key} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: color, boxShadow: `0 0 10px ${color}` }}
                      />
                      <span className="text-sm capitalize">{key}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-16 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ backgroundColor: color }}
                          initial={{ width: 0 }}
                          animate={{ width: `${value}%` }}
                          transition={{ duration: 1, delay: 0.2 }}
                        />
                      </div>
                      <span className="text-sm font-mono w-12 text-right">
                        {value.toFixed(1)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
            
            <div className="mt-4 pt-3 border-t border-gray-700">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Predicted Grade:</span>
                <span className={cn(
                  'font-bold',
                  predictedGrade.grade === 10 ? 'text-gold-shimmer' :
                  predictedGrade.grade >= 9 ? 'text-accent-cyan' :
                  predictedGrade.grade >= 7 ? 'text-status-success' :
                  'text-status-warning'
                )}>
                  {predictedGrade.label} {predictedGrade.grade}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}