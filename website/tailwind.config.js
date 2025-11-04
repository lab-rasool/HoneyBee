/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        primary: '#FFA726', // Orange-amber color for honey theme
        secondary: '#795548', // Brown color
        dark: '#212121',
      },
    },
  },
  plugins: [],
}