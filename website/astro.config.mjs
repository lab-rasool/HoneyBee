import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import icon from "astro-icon";
import sitemap from '@astrojs/sitemap';
import remarkDirective from 'remark-directive';
import { visit } from 'unist-util-visit';

// Custom plugin to transform code blocks
function remarkCodeBlocks() {
  return (tree) => {
    visit(tree, 'code', (node, index, parent) => {
      const { lang, value } = node;
      const filename = node.meta?.match(/filename="([^"]*)"/)?.[1];
      
      // Replace the code node with an mdxJsxFlowElement
      parent.children[index] = {
        type: 'html',
        value: `<CodeBlock code="${encodeURIComponent(value)}" lang="${lang || 'text'}" ${filename ? `filename="${filename}"` : ''} />`
      };
    });
  };
}

// https://astro.build/config
export default defineConfig({
  site: 'https://lab-rasool.github.io',
  base: '/HoneyBee',
  integrations: [
    tailwind(), 
    icon(),
    sitemap({
      filter: (page) => !page.includes('/_')
    })
  ],
  markdown: {
    remarkPlugins: [remarkDirective, remarkCodeBlocks],
  },
});