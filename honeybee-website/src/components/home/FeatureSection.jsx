'use client';

import { 
  BeakerIcon, 
  DocumentTextIcon, 
  LightBulbIcon, 
  ServerIcon 
} from '@heroicons/react/24/outline';

const features = [
  {
    name: 'Modular Framework',
    description: 'Build customizable workflows for oncology data processing with our plug-and-play architecture.',
    icon: ServerIcon,
  },
  {
    name: 'Multimodal Integration',
    description: 'Seamlessly combine imaging, genomic, and clinical data into unified representations.',
    icon: BeakerIcon,
  },
  {
    name: 'Foundation Model Integration',
    description: 'Leverage state-of-the-art foundation models to extract rich features from your oncology datasets.',
    icon: LightBulbIcon,
  },
  {
    name: 'Research Ready',
    description: 'Generate publication-quality datasets with proper citations and provenance tracking.',
    icon: DocumentTextIcon,
  },
];

export default function FeatureSection() {
  return (
    <section className="py-16 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Key Features
          </h2>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-600">
            HoneyBee provides powerful tools for oncology research
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4">
          {features.map((feature, index) => (
            <div 
              key={index} 
              className="bg-amber-50 rounded-lg p-6 border border-amber-100 hover:shadow-md transition-shadow duration-300"
            >
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-amber-500 text-white mx-auto">
                <feature.icon className="h-6 w-6" aria-hidden="true" />
              </div>
              <div className="mt-5 text-center">
                <h3 className="text-lg font-medium text-gray-900">{feature.name}</h3>
                <p className="mt-2 text-base text-gray-600">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-16 text-center">
          <a 
            href="https://github.com/yourusername/honeybee" 
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-amber-600 hover:bg-amber-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-500"
          >
            Explore Documentation
          </a>
        </div>
      </div>
    </section>
  );
}