import { Inter } from 'next/font/google';
import { ThemeProvider } from 'next-themes';
import Header from './Header';
import Footer from './Footer';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({ children }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light">
      <div className={`min-h-screen flex flex-col ${inter.className} bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-50`}>
        <Header />
        <main className="flex-grow">{children}</main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}