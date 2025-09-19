'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MinimalDashboard from '../components/dashboard/MinimalDashboard';
import { cn } from '../lib/utils';

export default function HomePage() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [showIntro, setShowIntro] = useState(true);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1000);

    // Hide intro after 3 seconds or on user interaction
    const introTimer = setTimeout(() => {
      setShowIntro(false);
    }, 3000);

    return () => {
      clearTimeout(timer);
      clearTimeout(introTimer);
    };
  }, []);

  const skipIntro = () => {
    setShowIntro(false);
  };

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex items-center justify-center">
        <div className="text-center">
          <div className="dna-spinner mx-auto mb-8 scale-150" />
          <motion.div
            className="flex items-center justify-center mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <img 
              src="/images/gemcheck-logo.png" 
              alt="GemCheck Logo" 
              className="h-12 w-auto mr-4"
            />
            <h1 className="text-4xl font-display holographic-text">
              GemCheck
            </h1>
          </motion.div>
          <motion.p
            className="text-gray-300 text-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            Initializing Neural Networks...
          </motion.p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Intro Sequence */}
      <AnimatePresence>
        {showIntro && (
          <motion.div
            className="fixed inset-0 z-50 bg-black flex items-center justify-center"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1 }}
          >
            <div className="text-center">
              <motion.div
                className="mb-8"
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ 
                  duration: 1.5, 
                  type: "spring", 
                  stiffness: 200,
                  damping: 20 
                }}
              >
                <div className="w-32 h-32 mx-auto relative">
                  <div className="absolute inset-0 rounded-full bg-gradient-to-r from-accent-cyan to-purple-500 animate-spin" 
                       style={{ animationDuration: '3s' }} />
                  <div className="absolute inset-2 rounded-full bg-black flex items-center justify-center">
                    <img 
                      src="/images/gemcheck-logo.png" 
                      alt="GemCheck" 
                      className="w-16 h-16 object-contain"
                    />
                  </div>
                </div>
              </motion.div>

              <motion.h1
                className="text-6xl font-display holographic-text mb-4"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1, delay: 0.5 }}
              >
                GemCheck
              </motion.h1>

              <motion.div
                className="space-y-2 text-xl text-gray-300 mb-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1, delay: 1 }}
              >
                <div>The Future of Card Authentication</div>
                <div className="text-accent-cyan font-medium">PSA Pre-Grading • Computational Photography • Real-time Analysis</div>
              </motion.div>

              <motion.button
                onClick={skipIntro}
                className="glass-morphism px-8 py-4 rounded-full font-medium magnetic-button ripple"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 1.5 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Enter Analysis Chamber →
              </motion.button>

              <motion.div
                className="mt-8 text-sm text-gray-500"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 2 }}
              >
                Click anywhere to continue
              </motion.div>
            </div>

            {/* Background particles */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              {Array.from({ length: 100 }, (_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-1 h-1 bg-accent-cyan rounded-full opacity-20"
                  initial={{
                    x: Math.random() * window.innerWidth,
                    y: Math.random() * window.innerHeight,
                    scale: 0
                  }}
                  animate={{
                    scale: [0, 1, 0],
                    opacity: [0, 0.6, 0]
                  }}
                  transition={{
                    duration: 3,
                    delay: Math.random() * 2,
                    repeat: Infinity,
                    repeatDelay: Math.random() * 5
                  }}
                />
              ))}
            </div>

            {/* Click handler */}
            <div 
              className="absolute inset-0 cursor-pointer"
              onClick={skipIntro}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Application */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: showIntro ? 0 : 1 }}
        transition={{ duration: 1 }}
        className={cn(showIntro ? 'pointer-events-none' : 'pointer-events-auto')}
      >
        <MinimalDashboard />
      </motion.div>
    </>
  );
}