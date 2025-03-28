import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  site: 'https://lab-rasool.github.io', // Your GitHub username or org name + .github.io
  base: '/HoneyBee', // Repository name
  markdown: {
    shikiConfig: {
      // Choose from Shiki's built-in themes (or add your custom theme)
      theme: 'github-dark',
      // Add custom languages
      langs: [],
      // Enable word wrap to better handle mobile display
      wrap: true,
    },
  },
  integrations: [tailwind()],
});