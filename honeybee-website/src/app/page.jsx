import RootLayout from '../components/layout/RootLayout';
import HeroSection from '../components/sections/HeroSection';
import FeaturesSection from '../components/sections/FeaturesSection';
import DataLoadersSection from '../components/sections/DataLoadersSection';
import CitationSection from '../components/sections/CitationSection';

export default function Home() {
  return (
    <RootLayout>
      <HeroSection />
      <FeaturesSection />
      <DataLoadersSection />
      <CitationSection />
    </RootLayout>
  );
}