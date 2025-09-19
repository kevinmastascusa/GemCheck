/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Premium Color Palette
        primary: {
          gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          50: '#f0f4ff',
          100: '#e0e9ff',
          200: '#c7d8ff',
          300: '#a4bcff',
          400: '#8195ff',
          500: '#667eea',
          600: '#5a67d8',
          700: '#4c51bf',
          800: '#434190',
          900: '#3c366b',
          950: '#764ba2',
        },
        accent: {
          neon: '#00FFFF',
          cyan: '#00FFFF',
          pink: '#EC4899',
          purple: '#8B5CF6',
        },
        gold: {
          shimmer: '#FFD700',
          light: '#FFA500',
          dark: '#B8860B',
        },
        glass: {
          white: 'rgba(255, 255, 255, 0.1)',
          dark: 'rgba(0, 0, 0, 0.1)',
        },
        deep: {
          black: '#0A0A0A',
          gray: '#1A1A1A',
        },
        status: {
          error: '#FF006E',
          success: '#00F5A0',
          warning: '#FFD60A',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'holographic': 'linear-gradient(45deg, #667eea, #764ba2, #EC4899, #8B5CF6)',
        'gold-shimmer': 'linear-gradient(45deg, #FFD700, #FFA500, #FFD700)',
        'glass-morphism': 'linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))',
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-gentle': 'bounce 2s infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'holographic': 'holographic 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'radar': 'radar 2s linear infinite',
        'matrix-rain': 'matrixRain 20s linear infinite',
        'grade-reveal': 'gradeReveal 3s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
        'confetti': 'confetti 3s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 255, 255, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 255, 255, 0.8), 0 0 30px rgba(0, 255, 255, 0.6)' },
        },
        holographic: {
          '0%, 100%': { 
            backgroundPosition: '0% 50%',
            filter: 'hue-rotate(0deg)',
          },
          '50%': { 
            backgroundPosition: '100% 50%',
            filter: 'hue-rotate(180deg)',
          },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        radar: {
          '0%': { 
            transform: 'rotate(0deg)',
            opacity: '1',
          },
          '100%': { 
            transform: 'rotate(360deg)',
            opacity: '0.3',
          },
        },
        matrixRain: {
          '0%': { transform: 'translateY(-100vh)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        gradeReveal: {
          '0%': { 
            transform: 'scale(0) rotate(-180deg)',
            opacity: '0',
          },
          '50%': { 
            transform: 'scale(1.1) rotate(-90deg)',
            opacity: '0.8',
          },
          '100%': { 
            transform: 'scale(1) rotate(0deg)',
            opacity: '1',
          },
        },
        confetti: {
          '0%': { transform: 'scale(0) rotate(0deg)', opacity: '1' },
          '100%': { transform: 'scale(1) rotate(720deg)', opacity: '0' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Orbitron', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
        '3xl': ['2rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.5rem', { lineHeight: '2.75rem' }],
        '5xl': ['3rem', { lineHeight: '3.25rem' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.5rem',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 255, 255, 0.6)',
        'glow-pink': '0 0 20px rgba(236, 72, 153, 0.6)',
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.6)',
        'glow-gold': '0 0 20px rgba(255, 215, 0, 0.6)',
        'inner-glow': 'inset 0 0 10px rgba(255, 255, 255, 0.1)',
        'card-float': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        'holographic': '0 8px 32px rgba(102, 126, 234, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05)',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
      screens: {
        '3xl': '1920px',
        '4xl': '2560px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
    function({ addUtilities }) {
      const newUtilities = {
        '.text-holographic': {
          background: 'linear-gradient(45deg, #667eea, #764ba2, #EC4899, #8B5CF6)',
          backgroundSize: '400% 400%',
          backgroundClip: 'text',
          '-webkit-background-clip': 'text',
          color: 'transparent',
          animation: 'holographic 3s ease-in-out infinite',
        },
        '.glass-morphism': {
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        },
        '.glass-morphism-dark': {
          background: 'rgba(0, 0, 0, 0.1)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
        '.perspective-1000': {
          perspective: '1000px',
        },
        '.preserve-3d': {
          transformStyle: 'preserve-3d',
        },
        '.backface-hidden': {
          backfaceVisibility: 'hidden',
        },
      };
      addUtilities(newUtilities);
    },
  ],
};