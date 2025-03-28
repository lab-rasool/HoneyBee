import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  site: 'https://lab-rasool.github.io', // Your GitHub username or org name + .github.io
  base: '/HoneyBee', // Repository name
  integrations: [tailwind()],
});