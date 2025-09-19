'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { Suspense } from 'react';
import Card3DViewer from '../3d/Card3DViewer';
import GradeDNA from '../analysis/GradeDNA';
import HeatMapOverlay from '../analysis/HeatMapOverlay';
import { cn, getGradeColor, getGradeLabel, formatPercentage } from '../../lib/utils';

interface CardAnalysis {
  id: string;
  cardName: string;
  frontImage: string;
  backImage?: string;
  predictedGrade: number;
  confidence: number;
  scores: {
    centering: number;
    edges: number;
    corners: number;
    surface: number;
  };
  defects: Array<{
    type: 'scratch' | 'edge_wear' | 'corner_damage' | 'surface_defect';
    position: { x: number; y: number };
    severity: number;
  }>;
  marketValue?: {
    current: number;
    projected: number;
    change: number;
  };
  analysisTimestamp: Date;
}

interface LiveFeedItem {
  id: string;
  type: 'analysis' | 'comparison' | 'market' | 'prediction';
  message: string;
  timestamp: Date;
  severity: 'info' | 'warning' | 'success' | 'error';
}

export default function MainDashboard() {
  const [currentCard, setCurrentCard] = useState<CardAnalysis | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'standard' | 'xray' | 'battle' | 'time-machine'>('standard');
  const [liveFeed, setLiveFeed] = useState<LiveFeedItem[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  
  const feedRef = useRef<HTMLDivElement>(null);
  const dashboardRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(dashboardRef);

  // Mock data for demonstration
  const mockAnalysis: CardAnalysis = {
    id: '1',
    cardName: '2000 Pokemon Base Set Charizard #4',
    frontImage: '/api/placeholder/400/560',
    backImage: '/api/placeholder/400/560',
    predictedGrade: 9,
    confidence: 87.3,
    scores: {
      centering: 92.1,
      edges: 88.5,
      corners: 91.2,
      surface: 85.7
    },
    defects: [
      { type: 'scratch', position: { x: 0.3, y: 0.7 }, severity: 0.2 },
      { type: 'edge_wear', position: { x: 0.95, y: 0.5 }, severity: 0.4 },
      { type: 'corner_damage', position: { x: 0.1, y: 0.1 }, severity: 0.3 }
    ],
    marketValue: {
      current: 8500,
      projected: 9200,
      change: 8.2
    },
    analysisTimestamp: new Date()
  };

  // Simulate live feed updates
  useEffect(() => {
    const interval = setInterval(() => {
      const feedItems: LiveFeedItem[] = [
        {
          id: Date.now().toString(),
          type: 'analysis',
          message: 'Neural network detected micro-scratch at coordinates (127, 341)',
          timestamp: new Date(),
          severity: 'warning'
        },
        {
          id: (Date.now() + 1).toString(),
          type: 'market',
          message: 'Market value increased 3.2% for Charizard Grade 9',
          timestamp: new Date(),
          severity: 'success'
        },
        {
          id: (Date.now() + 2).toString(),
          type: 'prediction',
          message: 'Confidence level: 87.3% → Grade 9 prediction',
          timestamp: new Date(),
          severity: 'info'
        }
      ];

      setLiveFeed(prev => [...feedItems, ...prev].slice(0, 50));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setIsAnalyzing(true);
      
      // Simulate analysis
      setTimeout(() => {
        setCurrentCard({
          ...mockAnalysis,
          frontImage: URL.createObjectURL(file),
          cardName: file.name.replace(/\.[^/.]+$/, "")
        });
        setIsAnalyzing(false);
      }, 3000);
    }
  };

  const heatMapData = currentCard?.defects.map(defect => ({
    x: defect.position.x,
    y: defect.position.y,
    intensity: defect.severity,
    type: defect.severity > 0.6 ? 'error' : defect.severity > 0.3 ? 'warning' : 'good'
  })) || [];

  return (
    <div 
      ref={dashboardRef}
      className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 relative overflow-hidden"
    >
      {/* Background Matrix Rain Effect */}
      <div className="matrix-rain">
        {Array.from({ length: 50 }, (_, i) => (
          <div
            key={i}
            className="matrix-char"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 20}s`,
              animationDuration: `${15 + Math.random() * 10}s`
            }}
          >
            {String.fromCharCode(0x30A0 + Math.random() * 96)}
          </div>
        ))}
      </div>

      {/* Main Dashboard Layout */}
      <div className="relative z-10 h-screen flex">
        
        {/* Left Panel - 3D Viewer (40%) */}
        <motion.div
          className="w-2/5 h-full relative"
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <div className="h-full glass-morphism-dark m-2 rounded-2xl p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-display holographic-text">
                3D Analysis Chamber
              </h2>
              <div className="flex space-x-2">
                {(['standard', 'xray', 'battle', 'time-machine'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setAnalysisMode(mode)}
                    className={cn(
                      'px-3 py-1 rounded-full text-xs font-medium transition-all',
                      analysisMode === mode 
                        ? 'bg-accent-cyan/20 text-accent-cyan shadow-glow' 
                        : 'glass-morphism text-gray-300 hover:text-white'
                    )}
                  >
                    {mode.replace('-', ' ').toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 3D Card Viewer */}
            <div className="h-full relative rounded-xl overflow-hidden">
              {currentCard ? (
                <Card3DViewer
                  frontImage={currentCard.frontImage}
                  backImage={currentCard.backImage}
                  cardName={currentCard.cardName}
                  grade={currentCard.predictedGrade}
                  analysis={currentCard.scores}
                  defects={currentCard.defects}
                  onAnalysisComplete={(analysis) => {
                    console.log('Analysis complete:', analysis);
                  }}
                />
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-gray-400">
                  <div className="w-32 h-32 rounded-full border-4 border-dashed border-gray-600 flex items-center justify-center mb-6">
                    <svg className="w-16 h-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium mb-2">Upload Your Card</h3>
                  <p className="text-sm text-center mb-6">
                    Drop your card image here or click to browse
                  </p>
                  <label className="magnetic-button ripple glass-morphism px-6 py-3 rounded-full cursor-pointer">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    Choose File
                  </label>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Center Panel - Analysis Data (35%) */}
        <motion.div
          className="w-[35%] h-full relative"
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
        >
          <div className="h-full flex flex-col space-y-2 p-2">
            
            {/* Grade DNA Visualization */}
            <div className="flex-1 glass-morphism-dark rounded-2xl p-4">
              <h3 className="text-xl font-display holographic-text mb-4">
                Grade DNA Analysis
              </h3>
              {currentCard ? (
                <GradeDNA
                  scores={currentCard.scores}
                  predictedGrade={currentCard.predictedGrade}
                  confidence={currentCard.confidence}
                  mode="helix"
                />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <div className="dna-spinner mx-auto mb-4" />
                    <p>Waiting for card analysis...</p>
                  </div>
                </div>
              )}
            </div>

            {/* Heat Map Analysis */}
            <div className="flex-1 glass-morphism-dark rounded-2xl p-4">
              <h3 className="text-xl font-display holographic-text mb-4">
                Defect Heat Map
              </h3>
              {currentCard ? (
                <div className="h-full relative">
                  <HeatMapOverlay
                    width={300}
                    height={200}
                    data={heatMapData}
                    mode={analysisMode === 'xray' ? 'xray' : 'defects'}
                    isAnimating={true}
                    className="w-full h-full"
                  />
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500">
                  <p>No defects detected yet</p>
                </div>
              )}
            </div>

            {/* Score Breakdown */}
            <div className="glass-morphism-dark rounded-2xl p-4">
              <h3 className="text-lg font-display text-white mb-3">Score Breakdown</h3>
              {currentCard ? (
                <div className="space-y-3">
                  {Object.entries(currentCard.scores).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-sm text-gray-300 capitalize">{key}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full bg-gradient-to-r from-accent-cyan to-green-400 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${value}%` }}
                            transition={{ duration: 1, delay: 0.5 }}
                          />
                        </div>
                        <span className="text-sm font-mono w-12 text-right">
                          {value.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Scores will appear after analysis</p>
              )}
            </div>
          </div>
        </motion.div>

        {/* Right Panel - Live Feed & Controls (25%) */}
        <motion.div
          className="w-1/4 h-full relative"
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
        >
          <div className="h-full flex flex-col space-y-2 p-2">
            
            {/* Grade Prediction */}
            {currentCard && (
              <div className="glass-morphism-dark rounded-2xl p-4">
                <div className="text-center">
                  <div className="text-sm text-gray-400 mb-2">Predicted Grade</div>
                  <motion.div
                    className={cn(
                      'w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-3',
                      'font-display font-bold text-3xl glass-morphism',
                      currentCard.predictedGrade === 10 ? 'text-gold-shimmer shadow-glow-gold' :
                      currentCard.predictedGrade >= 9 ? 'text-accent-cyan shadow-glow' :
                      'text-green-400'
                    )}
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ type: 'spring', stiffness: 300, delay: 1 }}
                  >
                    {currentCard.predictedGrade}
                  </motion.div>
                  <div className="text-sm text-gray-300">
                    {getGradeLabel(currentCard.predictedGrade)}
                  </div>
                  <div className="text-xs text-accent-cyan mt-1">
                    {formatPercentage(currentCard.confidence / 100)} confidence
                  </div>
                </div>
              </div>
            )}

            {/* Market Value */}
            {currentCard?.marketValue && (
              <div className="glass-morphism-dark rounded-2xl p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Market Value</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Current</span>
                    <span className="font-mono text-white">
                      ${currentCard.marketValue.current.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Projected</span>
                    <span className="font-mono text-green-400">
                      ${currentCard.marketValue.projected.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Change</span>
                    <span className={cn(
                      'font-mono text-xs',
                      currentCard.marketValue.change > 0 ? 'text-green-400' : 'text-red-400'
                    )}>
                      {currentCard.marketValue.change > 0 ? '+' : ''}
                      {formatPercentage(currentCard.marketValue.change / 100)}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Live Analysis Feed */}
            <div className="flex-1 glass-morphism-dark rounded-2xl p-4">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Live Analysis Feed</h3>
              <div
                ref={feedRef}
                className="h-full overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-gray-600"
              >
                <AnimatePresence>
                  {liveFeed.map((item) => (
                    <motion.div
                      key={item.id}
                      className={cn(
                        'p-2 rounded-lg text-xs',
                        item.severity === 'success' ? 'bg-green-500/10 text-green-400' :
                        item.severity === 'warning' ? 'bg-yellow-500/10 text-yellow-400' :
                        item.severity === 'error' ? 'bg-red-500/10 text-red-400' :
                        'bg-blue-500/10 text-blue-400'
                      )}
                      initial={{ x: 50, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      exit={{ x: -50, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="flex items-start justify-between">
                        <span className="flex-1">{item.message}</span>
                        <span className="text-gray-500 text-xs ml-2 shrink-0">
                          {item.timestamp.toLocaleTimeString('en-US', { 
                            hour12: false, 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Analysis Progress Overlay */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="glass-morphism rounded-2xl p-8 text-center max-w-md">
              <div className="dna-spinner mx-auto mb-6 scale-150" />
              <h3 className="text-2xl font-display holographic-text mb-4">
                Neural Analysis in Progress
              </h3>
              <div className="space-y-2 text-sm text-gray-300">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  🔍 Scanning card surface...
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.5 }}
                >
                  🧠 Processing neural networks...
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 2.5 }}
                >
                  📊 Calculating grade prediction...
                </motion.div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}