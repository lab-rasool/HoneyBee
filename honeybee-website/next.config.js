/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath: process.env.NODE_ENV === 'production' ? '/HoneyBee' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/HoneyBee/' : '',
}

export default nextConfig;