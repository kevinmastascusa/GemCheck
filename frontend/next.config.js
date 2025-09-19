/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    typedRoutes: true,
  },
  images: {
    domains: [
      'localhost',
      'api.psacard.com',
      'images.psacard.com',
      'cert.psacard.com'
    ],
    formats: ['image/webp', 'image/avif'],
  },
  webpack: (config, { dev, isServer }) => {
    // Handle WebGL and Three.js
    config.module.rules.push({
      test: /\.(glsl|vs|fs|vert|frag)$/,
      use: ['raw-loader', 'glslify-loader'],
    });

    // Handle audio files for sound effects
    config.module.rules.push({
      test: /\.(mp3|wav|ogg|flac)$/,
      use: {
        loader: 'url-loader',
        options: {
          limit: 8192,
          fallback: 'file-loader',
          publicPath: '/_next/static/sounds/',
          outputPath: 'static/sounds/',
        },
      },
    });

    // Optimize for production
    if (!dev && !isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        '@': path.resolve(__dirname, './'),
      };
    }

    return config;
  },
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:3001',
    ENABLE_3D_EFFECTS: process.env.ENABLE_3D_EFFECTS || 'true',
    ENABLE_SOUND: process.env.ENABLE_SOUND || 'true',
    PERFORMANCE_MODE: process.env.PERFORMANCE_MODE || 'high',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:3001'}/api/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ];
  },
};

const path = require('path');

module.exports = nextConfig;