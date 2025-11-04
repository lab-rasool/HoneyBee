# HoneyBee Website

This directory contains the source code for the HoneyBee project website, built with [Astro](https://astro.build/) and deployed to GitHub Pages.

The website showcases the HoneyBee framework, a modular, open-source toolkit for multimodal oncology data processing and analysis.

## Local Development

### Prerequisites

- [Node.js](https://nodejs.org/en/) (v18 or higher)
- npm (comes with Node.js)

### Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/lab-rasool/HoneyBee.git
   cd HoneyBee/website
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open your browser and visit `http://localhost:4321` to see the website.

## Building for Production

To build the website for production:

```bash
npm run build
```

This will generate the static files in the `dist` directory.

## Deployment

The website is automatically deployed to GitHub Pages via GitHub Actions whenever changes are pushed to the main branch. The workflow is defined in `.github/workflows/deploy.yml`.

## Project Structure

```
website/
├── public/                 # Static assets
│   ├── favicon.svg
│   └── images/
├── src/
│   ├── components/         # Reusable UI components
│   ├── layouts/            # Page layouts
│   ├── pages/              # Each .astro file becomes a route
│   └── styles/             # Global styles
├── astro.config.mjs        # Astro configuration
├── package.json
└── tailwind.config.js      # Tailwind CSS configuration
```

## Customizing

- **Content**: Update the content in the `src/pages` directory.
- **Styling**: Modify the global styles in `src/styles/global.css` or component-specific styles in their respective files.
- **Images**: Add or replace images in the `public/images` directory.

## Contributing

If you'd like to contribute to the website, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This website is available under the same license as the HoneyBee project.
