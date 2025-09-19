'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface AnalysisBox {
  id: string;
  type: 'centering' | 'edges' | 'corners' | 'surface';
  x: number;
  y: number;
  width: number;
  height: number;
  measurement: string;
  score: number;
  color: string;
}

interface CardAnalysisOverlayProps {
  imageRef: React.RefObject<HTMLImageElement>;
  isAnalyzing: boolean;
  onAnalysisComplete: (scores: any, measurements: any) => void;
  onLogMessage?: (message: string, severity: 'info' | 'success' | 'warning' | 'error') => void;
}

export default function CardAnalysisOverlay({ 
  imageRef, 
  isAnalyzing, 
  onAnalysisComplete,
  onLogMessage
}: CardAnalysisOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [analysisBoxes, setAnalysisBoxes] = useState<AnalysisBox[]>([]);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [progress, setProgress] = useState(0);

  // Real card analysis algorithm
  const analyzeCard = async () => {
    if (!imageRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    
    // Set canvas size to match image
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    // Draw image to canvas for analysis
    ctx?.drawImage(img, 0, 0);
    
    const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);
    if (!imageData) return;
    
    const boxes: AnalysisBox[] = [];
    const measurements: any = {};
    const scores: any = {};

    // Step 1: Detect card edges
    setCurrentStep('Detecting card edges...');
    setProgress(20);
    onLogMessage?.('Starting edge detection analysis...', 'info');
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const edgeBoxes = detectEdges(canvas.width, canvas.height, imageData);
    const averageEdgeQuality = edgeBoxes.reduce((sum, box) => sum + box.score, 0) / edgeBoxes.length;
    boxes.push(...edgeBoxes);
    setAnalysisBoxes([...boxes]);
    onLogMessage?.(`Detected ${edgeBoxes.length} card edges with average quality ${averageEdgeQuality.toFixed(1)}/100`, 'success');
    
    // Step 2: Measure centering
    setCurrentStep('Measuring centering accuracy...');
    setProgress(40);
    await new Promise(resolve => setTimeout(resolve, 1200));
    
    const centeringData = measureCentering(canvas.width, canvas.height);
    boxes.push(centeringData.box);
    measurements.centering = centeringData.measurement;
    scores.centering = centeringData.score;
    setAnalysisBoxes([...boxes]);
    onLogMessage?.(`Centering measured: ${centeringData.measurement} ratio (Score: ${centeringData.score.toFixed(1)})`, 'info');
    
    // Step 3: Analyze corners
    setCurrentStep('Analyzing corner sharpness...');
    setProgress(60);
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const cornerBoxes = analyzeCorners(canvas.width, canvas.height, imageData);
    boxes.push(...cornerBoxes);
    const cornerScore = cornerBoxes.reduce((sum, box) => sum + box.score, 0) / cornerBoxes.length;
    scores.corners = cornerScore;
    measurements.corners = `${cornerBoxes.length} corners analyzed`;
    setAnalysisBoxes([...boxes]);
    onLogMessage?.(`Corner analysis complete: Average sharpness ${cornerScore.toFixed(1)}%`, 'success');
    
    // Step 4: Surface analysis
    setCurrentStep('Scanning surface for defects...');
    setProgress(80);
    await new Promise(resolve => setTimeout(resolve, 1300));
    
    const surfaceData = analyzeSurface(canvas.width, canvas.height, imageData);
    boxes.push(...surfaceData.boxes);
    scores.surface = surfaceData.score;
    scores.edges = edgeBoxes.reduce((sum, box) => sum + box.score, 0) / edgeBoxes.length;
    measurements.surface = `${surfaceData.defects} defects found`;
    measurements.edges = `${edgeBoxes.length} edges measured`;
    
    if (surfaceData.defects === 0) {
      onLogMessage?.('Surface analysis complete: No defects detected', 'success');
    } else {
      onLogMessage?.(`Surface analysis found ${surfaceData.defects} defects affecting score`, 'warning');
    }
    
    setAnalysisBoxes([...boxes]);
    setProgress(100);
    setCurrentStep('Analysis complete');
    
    // Calculate raw measurements
    const rawMeasurements = {
      centeringRatio: measurements.centering,
      edgeQuality: `${scores.edges.toFixed(1)}/100`,
      edgeGradient: `avg ${averageEdgeQuality.toFixed(1)}px intensity`,
      cornerSharpness: `${scores.corners.toFixed(1)}%`,
      cornerSobel: `avg ${scores.corners.toFixed(1)} magnitude`,
      surfaceDefects: surfaceData.defects,
      surfaceVariance: surfaceData.defects > 0 ? `${surfaceData.defects} regions flagged` : 'clean surface'
    };
    
    // Calculate overall grade
    const overallScore = (scores.centering + scores.edges + scores.corners + scores.surface) / 4;
    const grade = Math.min(10, Math.floor(overallScore / 10) + 7);
    
    onLogMessage?.(`Analysis complete! Overall grade: ${grade} (${overallScore.toFixed(1)}% condition)`, 'success');
    onAnalysisComplete(scores, rawMeasurements);
  };

  // Real edge detection algorithm using actual pixel analysis
  const detectEdges = (width: number, height: number, imageData: ImageData): AnalysisBox[] => {
    const boxes: AnalysisBox[] = [];
    const edgeThickness = Math.max(10, Math.min(width, height) * 0.02); // Dynamic edge thickness based on image size
    const data = imageData.data;
    
    // Function to calculate edge sharpness from pixel data
    const calculateEdgeSharpness = (x: number, y: number, w: number, h: number): number => {
      let gradientSum = 0;
      let pixelCount = 0;
      
      for (let py = y; py < y + h && py < height - 1; py++) {
        for (let px = x; px < x + w && px < width - 1; px++) {
          const idx = (py * width + px) * 4;
          const nextIdx = ((py * width + px) + 1) * 4;
          
          // Calculate grayscale values
          const gray1 = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          const gray2 = (data[nextIdx] + data[nextIdx + 1] + data[nextIdx + 2]) / 3;
          
          // Calculate gradient (edge intensity)
          const gradient = Math.abs(gray1 - gray2);
          gradientSum += gradient;
          pixelCount++;
        }
      }
      
      // Convert gradient to quality score (0-100)
      const averageGradient = gradientSum / pixelCount;
      return Math.min(100, averageGradient / 2.55); // Normalize to 0-100 scale
    };
    
    // Analyze each edge with real pixel data
    const topSharpness = calculateEdgeSharpness(0, 0, width, edgeThickness);
    const bottomSharpness = calculateEdgeSharpness(0, height - edgeThickness, width, edgeThickness);
    const leftSharpness = calculateEdgeSharpness(0, 0, edgeThickness, height);
    const rightSharpness = calculateEdgeSharpness(width - edgeThickness, 0, edgeThickness, height);
    
    // Create boxes with real measurements
    boxes.push({
      id: 'edge-top',
      type: 'edges',
      x: 0,
      y: 0,
      width: width,
      height: edgeThickness,
      measurement: `Top edge sharpness: ${topSharpness.toFixed(1)}%`,
      score: topSharpness,
      color: 'rgb(59, 130, 246)'
    });
    
    boxes.push({
      id: 'edge-bottom',
      type: 'edges',
      x: 0,
      y: height - edgeThickness,
      width: width,
      height: edgeThickness,
      measurement: `Bottom edge sharpness: ${bottomSharpness.toFixed(1)}%`,
      score: bottomSharpness,
      color: 'rgb(59, 130, 246)'
    });
    
    boxes.push({
      id: 'edge-left',
      type: 'edges',
      x: 0,
      y: 0,
      width: edgeThickness,
      height: height,
      measurement: `Left edge sharpness: ${leftSharpness.toFixed(1)}%`,
      score: leftSharpness,
      color: 'rgb(59, 130, 246)'
    });
    
    boxes.push({
      id: 'edge-right',
      type: 'edges',
      x: width - edgeThickness,
      y: 0,
      width: edgeThickness,
      height: height,
      measurement: `Right edge sharpness: ${rightSharpness.toFixed(1)}%`,
      score: rightSharpness,
      color: 'rgb(59, 130, 246)'
    });
    
    return boxes;
  };

  // Real centering measurement using actual card border detection
  const measureCentering = (width: number, height: number, imageData: ImageData) => {
    const data = imageData.data;
    
    // Function to detect card borders by finding significant brightness changes
    const findCardBorders = () => {
      let leftBorder = 0;
      let rightBorder = width - 1;
      let topBorder = 0;
      let bottomBorder = height - 1;
      
      // Find left border - scan from left to right
      for (let x = 0; x < width * 0.3; x++) {
        let edgeStrength = 0;
        for (let y = height * 0.2; y < height * 0.8; y++) {
          const idx = (y * width + x) * 4;
          const nextIdx = (y * width + (x + 1)) * 4;
          const brightness1 = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          const brightness2 = (data[nextIdx] + data[nextIdx + 1] + data[nextIdx + 2]) / 3;
          edgeStrength += Math.abs(brightness1 - brightness2);
        }
        if (edgeStrength > 1000) { // Threshold for detecting card edge
          leftBorder = x;
          break;
        }
      }
      
      // Find right border - scan from right to left
      for (let x = width - 1; x > width * 0.7; x--) {
        let edgeStrength = 0;
        for (let y = height * 0.2; y < height * 0.8; y++) {
          const idx = (y * width + x) * 4;
          const prevIdx = (y * width + (x - 1)) * 4;
          const brightness1 = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          const brightness2 = (data[prevIdx] + data[prevIdx + 1] + data[prevIdx + 2]) / 3;
          edgeStrength += Math.abs(brightness1 - brightness2);
        }
        if (edgeStrength > 1000) {
          rightBorder = x;
          break;
        }
      }
      
      // Find top border
      for (let y = 0; y < height * 0.3; y++) {
        let edgeStrength = 0;
        for (let x = width * 0.2; x < width * 0.8; x++) {
          const idx = (y * width + x) * 4;
          const nextIdx = ((y + 1) * width + x) * 4;
          const brightness1 = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          const brightness2 = (data[nextIdx] + data[nextIdx + 1] + data[nextIdx + 2]) / 3;
          edgeStrength += Math.abs(brightness1 - brightness2);
        }
        if (edgeStrength > 1000) {
          topBorder = y;
          break;
        }
      }
      
      // Find bottom border
      for (let y = height - 1; y > height * 0.7; y--) {
        let edgeStrength = 0;
        for (let x = width * 0.2; x < width * 0.8; x++) {
          const idx = (y * width + x) * 4;
          const prevIdx = ((y - 1) * width + x) * 4;
          const brightness1 = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          const brightness2 = (data[prevIdx] + data[prevIdx + 1] + data[prevIdx + 2]) / 3;
          edgeStrength += Math.abs(brightness1 - brightness2);
        }
        if (edgeStrength > 1000) {
          bottomBorder = y;
          break;
        }
      }
      
      return { leftBorder, rightBorder, topBorder, bottomBorder };
    };
    
    const borders = findCardBorders();
    const cardWidth = borders.rightBorder - borders.leftBorder;
    const cardHeight = borders.bottomBorder - borders.topBorder;
    
    // Calculate actual centering measurements
    const leftMargin = borders.leftBorder;
    const rightMargin = width - borders.rightBorder;
    const topMargin = borders.topBorder;
    const bottomMargin = height - borders.bottomBorder;
    
    // Calculate horizontal centering ratio
    const totalHorizontalMargin = leftMargin + rightMargin;
    const leftRatio = totalHorizontalMargin > 0 ? (leftMargin / totalHorizontalMargin) * 100 : 50;
    const rightRatio = 100 - leftRatio;
    
    // Calculate vertical centering ratio
    const totalVerticalMargin = topMargin + bottomMargin;
    const topRatio = totalVerticalMargin > 0 ? (topMargin / totalVerticalMargin) * 100 : 50;
    const bottomRatio = 100 - topRatio;
    
    // Calculate centering score based on deviation from perfect 50/50
    const horizontalDeviation = Math.abs(leftRatio - 50);
    const verticalDeviation = Math.abs(topRatio - 50);
    const overallDeviation = (horizontalDeviation + verticalDeviation) / 2;
    const centeringScore = Math.max(0, 100 - (overallDeviation * 4)); // Penalty factor of 4
    
    return {
      box: {
        id: 'centering-main',
        type: 'centering' as const,
        x: borders.leftBorder,
        y: borders.topBorder,
        width: cardWidth,
        height: cardHeight,
        measurement: `H: ${leftRatio.toFixed(1)}/${rightRatio.toFixed(1)}, V: ${topRatio.toFixed(1)}/${bottomRatio.toFixed(1)}`,
        score: centeringScore,
        color: 'rgb(239, 68, 68)'
      },
      measurement: `${leftRatio.toFixed(1)}/${rightRatio.toFixed(1)}`,
      score: centeringScore
    };
  };

  // Real corner analysis using pixel-based sharpness detection
  const analyzeCorners = (width: number, height: number, imageData: ImageData): AnalysisBox[] => {
    const cornerSize = Math.max(30, Math.min(width, height) * 0.08);
    const data = imageData.data;
    
    // Function to calculate corner sharpness using Sobel operator
    const calculateCornerSharpness = (x: number, y: number, size: number): number => {
      let totalSharpness = 0;
      let pixelCount = 0;
      
      for (let py = y + 1; py < y + size - 1 && py < height - 1; py++) {
        for (let px = x + 1; px < x + size - 1 && px < width - 1; px++) {
          // Get surrounding pixel indices
          const tl = ((py - 1) * width + (px - 1)) * 4; // top-left
          const tm = ((py - 1) * width + px) * 4;       // top-middle  
          const tr = ((py - 1) * width + (px + 1)) * 4; // top-right
          const ml = (py * width + (px - 1)) * 4;       // middle-left
          const mr = (py * width + (px + 1)) * 4;       // middle-right
          const bl = ((py + 1) * width + (px - 1)) * 4; // bottom-left
          const bm = ((py + 1) * width + px) * 4;       // bottom-middle
          const br = ((py + 1) * width + (px + 1)) * 4; // bottom-right
          
          // Convert to grayscale and apply Sobel operator
          const grayTL = (data[tl] + data[tl + 1] + data[tl + 2]) / 3;
          const grayTM = (data[tm] + data[tm + 1] + data[tm + 2]) / 3;
          const grayTR = (data[tr] + data[tr + 1] + data[tr + 2]) / 3;
          const grayML = (data[ml] + data[ml + 1] + data[ml + 2]) / 3;
          const grayMR = (data[mr] + data[mr + 1] + data[mr + 2]) / 3;
          const grayBL = (data[bl] + data[bl + 1] + data[bl + 2]) / 3;
          const grayBM = (data[bm] + data[bm + 1] + data[bm + 2]) / 3;
          const grayBR = (data[br] + data[br + 1] + data[br + 2]) / 3;
          
          // Sobel X and Y gradients
          const sobelX = (grayTR + 2 * grayMR + grayBR) - (grayTL + 2 * grayML + grayBL);
          const sobelY = (grayBL + 2 * grayBM + grayBR) - (grayTL + 2 * grayTM + grayTR);
          
          // Calculate magnitude
          const magnitude = Math.sqrt(sobelX * sobelX + sobelY * sobelY);
          totalSharpness += magnitude;
          pixelCount++;
        }
      }
      
      // Convert to quality score (0-100)
      const averageSharpness = pixelCount > 0 ? totalSharpness / pixelCount : 0;
      return Math.min(100, averageSharpness / 3); // Normalize to 0-100 scale
    };
    
    const corners = [
      { x: 0, y: 0, name: 'Top Left' },
      { x: width - cornerSize, y: 0, name: 'Top Right' },
      { x: 0, y: height - cornerSize, name: 'Bottom Left' },
      { x: width - cornerSize, y: height - cornerSize, name: 'Bottom Right' }
    ];
    
    return corners.map((corner, index) => {
      const sharpness = calculateCornerSharpness(corner.x, corner.y, cornerSize);
      return {
        id: `corner-${index}`,
        type: 'corners' as const,
        x: corner.x,
        y: corner.y,
        width: cornerSize,
        height: cornerSize,
        measurement: `${corner.name}: ${sharpness.toFixed(1)}%`,
        score: sharpness,
        color: 'rgb(34, 197, 94)' // green-500
      };
    });
  };

  // Real surface analysis using pixel-based defect detection
  const analyzeSurface = (width: number, height: number, imageData: ImageData) => {
    const boxes: AnalysisBox[] = [];
    const data = imageData.data;
    let totalScore = 100;
    let defectCount = 0;
    
    // Define analysis regions (avoid edges)
    const margin = Math.min(width, height) * 0.1;
    const analysisWidth = width - 2 * margin;
    const analysisHeight = height - 2 * margin;
    const regionSize = 50; // Size of each analysis region
    
    // Function to detect surface defects in a region
    const analyzeRegion = (startX: number, startY: number, regionW: number, regionH: number) => {
      const endX = Math.min(startX + regionW, width);
      const endY = Math.min(startY + regionH, height);
      
      let totalVariance = 0;
      let pixelCount = 0;
      let averageBrightness = 0;
      
      // Calculate average brightness first
      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const idx = (y * width + x) * 4;
          const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          averageBrightness += brightness;
          pixelCount++;
        }
      }
      averageBrightness /= pixelCount;
      
      // Calculate variance (defects show as brightness anomalies)
      pixelCount = 0;
      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const idx = (y * width + x) * 4;
          const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          const diff = Math.abs(brightness - averageBrightness);
          totalVariance += diff * diff;
          pixelCount++;
        }
      }
      
      const variance = totalVariance / pixelCount;
      const standardDeviation = Math.sqrt(variance);
      
      // High standard deviation indicates potential defects
      return { variance, standardDeviation, averageBrightness };
    };
    
    // Scan surface in regions
    const defectThreshold = 25; // Threshold for defect detection
    
    for (let y = margin; y < height - margin; y += regionSize) {
      for (let x = margin; x < width - margin; x += regionSize) {
        const regionW = Math.min(regionSize, width - margin - x);
        const regionH = Math.min(regionSize, height - margin - y);
        
        const analysis = analyzeRegion(x, y, regionW, regionH);
        
        // Check if this region has potential defects
        if (analysis.standardDeviation > defectThreshold) {
          const severity = Math.min(20, (analysis.standardDeviation - defectThreshold) / 2);
          const defectTypes = ['Surface variation', 'Print inconsistency', 'Minor mark', 'Texture anomaly'];
          const defectType = defectTypes[Math.floor((analysis.standardDeviation / 10) % defectTypes.length)];
          
          totalScore -= severity;
          defectCount++;
          
          boxes.push({
            id: `surface-defect-${defectCount}`,
            type: 'surface',
            x,
            y,
            width: regionW,
            height: regionH,
            measurement: `${defectType} (σ=${analysis.standardDeviation.toFixed(1)})`,
            score: 100 - severity,
            color: 'rgb(245, 158, 11)' // amber-500
          });
        }
      }
    }
    
    // Add overall surface quality region if no defects found
    if (defectCount === 0) {
      boxes.push({
        id: 'surface-clean',
        type: 'surface',
        x: margin,
        y: margin,
        width: analysisWidth,
        height: analysisHeight,
        measurement: 'Surface: Clean (no defects detected)',
        score: 100,
        color: 'rgb(34, 197, 94)' // green-500
      });
    }
    
    return {
      boxes,
      score: Math.max(totalScore, 60),
      defects: defectCount
    };
  };

  // Start analysis when isAnalyzing becomes true
  useEffect(() => {
    if (isAnalyzing) {
      setAnalysisBoxes([]);
      setProgress(0);
      setCurrentStep('Starting analysis...');
      analyzeCard();
    } else {
      setAnalysisBoxes([]);
      setCurrentStep('');
      setProgress(0);
    }
  }, [isAnalyzing]);

  if (!imageRef.current) return null;

  // Get the actual displayed image dimensions accounting for object-contain
  const img = imageRef.current;
  const imgRect = img.getBoundingClientRect();
  
  // Calculate the actual image display size within the container (accounting for object-contain)
  const imageAspectRatio = img.naturalWidth / img.naturalHeight;
  const containerAspectRatio = imgRect.width / imgRect.height;
  
  let displayWidth, displayHeight;
  if (imageAspectRatio > containerAspectRatio) {
    // Image is wider - constrained by width
    displayWidth = imgRect.width;
    displayHeight = imgRect.width / imageAspectRatio;
  } else {
    // Image is taller - constrained by height
    displayHeight = imgRect.height;
    displayWidth = imgRect.height * imageAspectRatio;
  }
  
  const scaleX = displayWidth / img.naturalWidth;
  const scaleY = displayHeight / img.naturalHeight;

  return (
    <>
      {/* Hidden canvas for analysis */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      {/* Analysis overlays */}
      <div 
        className="absolute pointer-events-none"
        style={{
          width: displayWidth,
          height: displayHeight,
          left: (imgRect.width - displayWidth) / 2,
          top: (imgRect.height - displayHeight) / 2
        }}
      >
        <AnimatePresence>
          {analysisBoxes.map((box) => (
            <motion.div
              key={box.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 0.8, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="absolute border-2 backdrop-blur-sm rounded"
              style={{
                left: box.x * scaleX,
                top: box.y * scaleY,
                width: box.width * scaleX,
                height: box.height * scaleY,
                borderColor: box.color,
                backgroundColor: `${box.color}20`
              }}
            >
              <div 
                className="absolute -top-8 left-0 px-2 py-1 text-xs font-medium text-white rounded shadow-lg whitespace-nowrap z-10"
                style={{ backgroundColor: box.color }}
              >
                {box.measurement}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Analysis progress indicator */}
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-4 left-4 bg-black/80 text-white px-4 py-3 rounded-lg backdrop-blur-sm"
          >
            <div className="flex items-center space-x-3">
              <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
              <div>
                <div className="text-sm font-medium">{currentStep}</div>
                <div className="w-48 bg-gray-700 rounded-full h-2 mt-1">
                  <motion.div
                    className="bg-blue-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="text-xs text-gray-300 mt-1">{progress}% complete</div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </>
  );
}