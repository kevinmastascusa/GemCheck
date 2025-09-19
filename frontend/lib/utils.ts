import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(amount: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(amount);
}

export function formatNumber(number: number): string {
  return new Intl.NumberFormat('en-US').format(number);
}

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(d);
}

export function formatDateTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(d);
}

export function formatPercentage(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function lerp(start: number, end: number, factor: number): number {
  return start + (end - start) * factor;
}

export function getGradeColor(grade: number): string {
  if (grade === 10) return '#FFD700'; // Gold
  if (grade >= 9) return '#00FFFF'; // Cyan
  if (grade >= 8) return '#00F5A0'; // Green
  if (grade >= 7) return '#FFD60A'; // Yellow
  if (grade >= 6) return '#FF9500'; // Orange
  return '#FF006E'; // Red/Pink
}

export function getGradeLabel(grade: number): string {
  const labels: Record<number, string> = {
    10: 'Gem Mint 10',
    9: 'Mint 9',
    8: 'NM-Mint 8',
    7: 'Near Mint 7',
    6: 'Excellent 6',
    5: 'VG-EX 5',
    4: 'Good 4',
    3: 'Fair 3',
    2: 'Poor 2',
    1: 'Authentic 1',
  };
  return labels[grade] || `Grade ${grade}`;
}

export function getConfidenceColor(confidence: number): string {
  if (confidence >= 90) return '#00F5A0'; // High confidence - green
  if (confidence >= 70) return '#FFD60A'; // Medium confidence - yellow
  return '#FF9500'; // Low confidence - orange
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

export function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  } else {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    return new Promise((resolve, reject) => {
      if (document.execCommand('copy')) {
        resolve();
      } else {
        reject(new Error('Failed to copy to clipboard'));
      }
      textArea.remove();
    });
  }
}

export function downloadFile(data: Blob, filename: string): void {
  const url = URL.createObjectURL(data);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function isValidImageUrl(url: string): boolean {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif'];
  const urlLower = url.toLowerCase();
  return imageExtensions.some(ext => urlLower.includes(ext)) || 
         urlLower.includes('image') || 
         urlLower.includes('img');
}

export function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

export function resizeImage(
  file: File, 
  maxWidth: number, 
  maxHeight: number, 
  quality = 0.8
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      const { width, height } = img;
      const ratio = Math.min(maxWidth / width, maxHeight / height);
      
      canvas.width = width * ratio;
      canvas.height = height * ratio;

      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error('Failed to resize image'));
          }
        },
        'image/jpeg',
        quality
      );
    };

    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

export function validateCertNumber(certNumber: string): boolean {
  // PSA cert numbers are typically 8-9 digits
  return /^\d{8,9}$/.test(certNumber.replace(/\s/g, ''));
}

export function formatCertNumber(certNumber: string): string {
  const cleaned = certNumber.replace(/\D/g, '');
  if (cleaned.length === 8) {
    return cleaned.replace(/(\d{4})(\d{4})/, '$1 $2');
  }
  if (cleaned.length === 9) {
    return cleaned.replace(/(\d{1})(\d{4})(\d{4})/, '$1 $2 $3');
  }
  return cleaned;
}

export function getAnalysisColor(score: number): string {
  if (score >= 90) return '#00F5A0'; // Excellent - green
  if (score >= 80) return '#FFD60A'; // Good - yellow
  if (score >= 70) return '#FF9500'; // Fair - orange
  if (score >= 60) return '#EC4899'; // Poor - pink
  return '#FF006E'; // Very poor - red
}

export function calculateOverallScore(scores: {
  centering: number;
  edges: number;
  corners: number;
  surface: number;
  weights?: {
    centering?: number;
    edges?: number;
    corners?: number;
    surface?: number;
  };
}): number {
  const weights = scores.weights || {
    centering: 0.3,
    edges: 0.25,
    corners: 0.25,
    surface: 0.2,
  };

  return (
    scores.centering * weights.centering +
    scores.edges * weights.edges +
    scores.corners * weights.corners +
    scores.surface * weights.surface
  );
}

export function generateMockDefects(count = 3): Array<{
  type: 'scratch' | 'edge_wear' | 'corner_damage' | 'surface_defect';
  position: { x: number; y: number };
  severity: number;
}> {
  const types = ['scratch', 'edge_wear', 'corner_damage', 'surface_defect'] as const;
  
  return Array.from({ length: count }, () => ({
    type: types[Math.floor(Math.random() * types.length)],
    position: {
      x: Math.random(),
      y: Math.random(),
    },
    severity: Math.random(),
  }));
}