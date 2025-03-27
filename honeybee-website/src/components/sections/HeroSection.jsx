import Image from 'next/image';
import Button from '../ui/Button';
import { motion } from 'framer-motion';

export default function HeroSection() {
  return (
    <div className="relative overflow-hidden bg-gradient-to-b from-amber-50 to-white dark:from-slate-900 dark:to-slate-950 py-16 sm:py-24">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:grid lg:grid-cols-12 lg:gap-8 items-center">
          <div className="sm:text-center lg:text-left lg:col-span-7">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-4xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-5xl md:text-6xl"
            >
              <span className="block">HoneyBee</span>
              <span className="block text-amber-600 dark:text-amber-500">AI for Oncology</span>
            </motion.h1>
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mt-3 text-base text-slate-600 dark:text-slate-300 sm:mt-5 sm:text-xl lg:text-lg xl:text-xl"
            >
              A scalable modular framework for creating multimodal oncology datasets with foundational embedding models. 
              HoneyBee provides tools for medical data loading, embedding generation, and advanced RAG support.
            </motion.p>
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mt-8 sm:flex sm:justify-center lg:justify-start gap-4"
            >
              <Button href="https://github.com/lab-rasool/HoneyBee" size="lg" target="_blank">
                Get Started
              </Button>
              <Button href="https://arxiv.org/abs/2405.07460" variant="outline" size="lg" target="_blank">
                Read the Paper
              </Button>
            </motion.div>
          </div>
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-12 lg:mt-0 lg:col-span-5 flex justify-center"
          >
            <div className="relative w-full max-w-lg">
              <div className="absolute top-0 left-0 w-72 h-72 bg-amber-300 dark:bg-amber-700 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
              <div className="absolute top-0 right-0 w-72 h-72 bg-sky-300 dark:bg-sky-700 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
              <div className="absolute bottom-0 left-0 w-72 h-72 bg-purple-300 dark:bg-purple-700 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
              <div className="relative bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-6 border border-slate-200 dark:border-slate-700">
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2 h-48 rounded-lg bg-slate-100 dark:bg-slate-700 flex items-center justify-center p-4">
                    <Image 
                      src="/assets/logo.png" 
                      alt="HoneyBee Logo" 
                      width={128} 
                      height={128}
                      className="object-contain"
                    />
                  </div>
                  <div className="h-24 rounded-lg bg-slate-100 dark:bg-slate-700 flex items-center justify-center p-4">
                    <Image src="/file.svg" alt="Files" width={40} height={40} />
                  </div>
                  <div className="h-24 rounded-lg bg-slate-100 dark:bg-slate-700 flex items-center justify-center p-4">
                    <Image src="/window.svg" alt="Interface" width={40} height={40} />
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}