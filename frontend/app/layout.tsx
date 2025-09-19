import type { Metadata } from 'next';
import { Inter, Orbitron, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });
const orbitron = Orbitron({ subsets: ['latin'], variable: '--font-display' });
const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' });

export const metadata: Metadata = {
  title: 'GemCheck | Revolutionary PSA Card Pre-Grading with AI',
  description: 'Next-generation trading card grading by GemCheck. AI-powered analysis, real-time computational photography overlays, and professional PSA-style grading. The future of card authentication.',
  keywords: ['GemCheck', 'PSA', 'card grading', 'AI grading', 'trading cards', 'computer vision', 'card authentication'],
  authors: [{ name: 'GemCheck Team' }],
  creator: 'GemCheck',
  publisher: 'GemCheck',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://psa-pregrader.com'),
  openGraph: {
    title: 'GemCheck | Revolutionary PSA Card Pre-Grading with AI',
    description: 'Experience the future of card grading with GemCheck - cutting-edge computational photography overlays and AI-powered PSA analysis.',
    url: 'https://gemcheck.ai',
    siteName: 'GemCheck',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'PSA Pre-Grader 3D Interface',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'GemCheck | Revolutionary PSA Card Pre-Grading with AI',
    description: 'Experience the future of card grading with GemCheck - cutting-edge computational photography and AI analysis.',
    images: ['/twitter-image.jpg'],
    creator: '@gemcheck',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html 
      lang="en" 
      className={cn(
        inter.variable, 
        orbitron.variable,
        jetbrainsMono.variable,
        'font-sans'
      )}
      suppressHydrationWarning
    >
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="theme-color" content="#667eea" />
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <link rel="icon" type="image/png" href="/favicon.png" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
        
        {/* Preload critical resources */}
        <link rel="preload" href="/sounds/ui-hover.mp3" as="audio" type="audio/mpeg" />
        <link rel="preload" href="/sounds/grade-reveal.mp3" as="audio" type="audio/mpeg" />
        <link rel="preload" href="/sounds/perfect-ten.mp3" as="audio" type="audio/mpeg" />
        
        {/* WebGL/Three.js optimizations */}
        <meta name="webgl" content="required" />
        <meta name="gpu-memory" content="preferred" />
      </head>
      <body 
        className={cn(
          'min-h-screen bg-gradient-to-br from-deep-black via-gray-950 to-deep-black',
          'text-white antialiased overflow-x-hidden',
          'selection:bg-accent-cyan/20 selection:text-accent-cyan'
        )}
      >
        {/* Background Effects */}
        <div className="fixed inset-0 z-0">
          {/* Ambient animated background */}
          <div className="absolute inset-0 opacity-30">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 via-accent-purple/10 to-accent-pink/10 animate-holographic" />
          </div>
          
          {/* Geometric patterns */}
          <div className="absolute inset-0 opacity-20">
            <div className="absolute top-0 left-0 w-full h-full">
              <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(102, 126, 234, 0.1)" strokeWidth="1"/>
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />
              </svg>
            </div>
          </div>
          
          {/* Particle effects */}
          <div className="absolute inset-0 overflow-hidden">
            {Array.from({ length: 20 }, (_, i) => (
              <div
                key={i}
                className={cn(
                  'absolute w-1 h-1 bg-accent-cyan rounded-full opacity-60',
                  'animate-float'
                )}
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 6}s`,
                  animationDuration: `${6 + Math.random() * 4}s`,
                }}
              />
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="relative z-10 min-h-screen">
          {children}
        </div>

        {/* Loading overlay */}
        <div 
          id="loading-overlay" 
          className="fixed inset-0 z-100 bg-deep-black flex items-center justify-center transition-opacity duration-1000"
        >
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin mb-4 mx-auto" />
            <div className="text-xl font-display text-holographic">
              Initializing GemCheck Interface...
            </div>
            <div className="text-sm text-gray-400 mt-2">
              Loading WebGL components
            </div>
          </div>
        </div>

        {/* Performance monitoring */}
        {process.env.NODE_ENV === 'development' && (
          <div className="fixed bottom-4 left-4 z-90 glass-morphism rounded-lg p-2 text-xs font-mono">
            <div id="fps-counter"></div>
            <div id="memory-usage"></div>
          </div>
        )}

        {/* Initialize app */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Remove loading overlay when page is fully loaded
              window.addEventListener('load', () => {
                const overlay = document.getElementById('loading-overlay');
                if (overlay) {
                  overlay.style.opacity = '0';
                  setTimeout(() => overlay.remove(), 1000);
                }
              });
              
              // Performance monitoring
              if (typeof window !== 'undefined' && window.performance) {
                let fps = 0;
                let lastFrame = performance.now();
                
                function updatePerformance() {
                  const now = performance.now();
                  fps = Math.round(1000 / (now - lastFrame));
                  lastFrame = now;
                  
                  const fpsElement = document.getElementById('fps-counter');
                  const memoryElement = document.getElementById('memory-usage');
                  
                  if (fpsElement) fpsElement.textContent = \`FPS: \${fps}\`;
                  if (memoryElement && performance.memory) {
                    const memory = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
                    memoryElement.textContent = \`Memory: \${memory}MB\`;
                  }
                  
                  requestAnimationFrame(updatePerformance);
                }
                
                if (${process.env.NODE_ENV === 'development'}) {
                  updatePerformance();
                }
              }
            `,
          }}
        />
      </body>
    </html>
  );
}