import Image from 'next/image';
import Link from 'next/link';
import Hero from '@/components/home/Hero';
import FeatureSection from '@/components/home/FeatureSection';
import ModelShowcase from '@/components/home/ModelShowcase';
import CitationCard from '@/components/home/CitationCard';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between">
      <Hero 
        title="HoneyBee Framework"
        subtitle="A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models"
        ctaText="Get Started"
        ctaLink="/documentation"
      />
      
      <FeatureSection />
      
      <section className="w-full py-12 md:py-24 bg-gray-50 dark:bg-gray-900">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter text-center mb-12">
            Supported Foundation Models
          </h2>
          <ModelShowcase />
        </div>
      </section>
      
      <section className="w-full py-12 md:py-24">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter text-center mb-12">
            Citation
          </h2>
          <CitationCard />
        </div>
      </section>
    </main>
  );
}