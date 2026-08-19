/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: "#b60021",
        "primary-container": "#e4002c",
        "on-primary": "#ffffff",
        "surface-tint": "#bf0024",
        surface: "#fff8f7",
        "surface-variant": "#fedad9",
        "surface-container": "#ffe9e8",
        "surface-container-high": "#ffe1e0",
        "surface-container-highest": "#fedad9",
        "on-surface": "#291616",
        "on-surface-variant": "#5e3f3d",
        secondary: "#575a8c",
        "secondary-container": "#c2c5fe",
        tertiary: "#505a6c",
        "neon-green": "#39ff14",
        "neon-yellow": "#eaff00",
        brand: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          900: '#134e4a',
        },
        slate: {
          850: '#151e2e',
          950: '#0b0f17',
        }
      },
      borderWidth: {
        '3': '3px',
        '4': '4px',
      },
      fontFamily: {
        body: ['Inter', 'sans-serif'],
        headline: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        label: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-fast': 'pulse 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ripple': 'ripple 2s linear infinite',
      }
    },
  },
  plugins: [],
}
