import Link from 'next/link';
import Image from 'next/image';

interface HeroProps {
  title: string;
  subtitle: string;
  ctaText: string;
  ctaLink: string;
}

export default function Hero({ title, subtitle, ctaText, ctaLink }: HeroProps) {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="container px-4 md:px-6 mx-auto flex flex-col md:flex-row items-center gap-8">
        <div className="flex flex-col justify-center space-y-4 md:w-1/2">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
              {title}
            </h1>
            <p className="text-gray-500 dark:text-gray-400 md:text-xl">
              {subtitle}
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-4">
            <Link 
              href={ctaLink} 
              className="inline-flex h-10 items-center justify-center rounded-md bg-blue-600 px-8 text-sm font-medium text-white shadow transition-colors hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              {ctaText}
            </Link>
            <Link 
              href="https://github.com/lab-rasool/HoneyBee" 
              className="inline-flex h-10 items-center justify-center rounded-md border border-gray-200 bg-white px-8 text-sm font-medium shadow-sm transition-colors hover:bg-gray-100 hover:text-gray-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 dark:border-gray-800 dark:bg-gray-950 dark:hover:bg-gray-800 dark:hover:text-gray-50 dark:focus-visible:ring-gray-300"
            >
              GitHub Repository
            </Link>
          </div>
        </div>
        <div className="md:w-1/2 flex justify-center">
          <div className="relative w-full max-w-md">
            <Image 
              src="/assets/hero-image.svg" 
              alt="HoneyBee Framework Illustration" 
              width={500} 
              height={500} 
              className="relative z-10"
            />
            <div className="absolute -z-10 top-1/4 right-1/4 w-32 h-32 bg-blue-200 dark:bg-blue-900 rounded-full blur-2xl opacity-50"></div>
            <div className="absolute -z-10 bottom-1/4 left-1/4 w-40 h-40 bg-yellow-200 dark:bg-yellow-900 rounded-full blur-3xl opacity-40"></div>
          </div>
        </div>
      </div>
    </section>
  );
}