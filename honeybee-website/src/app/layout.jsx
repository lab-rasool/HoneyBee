import { Inter } from 'next/font/google';
import './globals.css';
import RootLayout from '../components/layout/RootLayout';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'HoneyBee - AI for Oncology',
  description: 'A scalable modular framework for creating multimodal oncology datasets with foundational embedding models',
};

export default function Layout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        {children}
      </body>
    </html>
  );
}