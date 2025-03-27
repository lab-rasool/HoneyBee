'use client';

import { useState } from 'react';
import Image from 'next/image';

const models = [
  {
    id: 'embeddings',
    name: 'Multimodal Embeddings',
    description: 'Generate unified embeddings from medical images, genomic data, and clinical reports using our custom transformer models.',
    image: '/assets/embedding-visualization.png',
  },
  {
    id: 'classification',
    name: 'Disease Classification',
    description: 'Apply foundation models to classify cancer types and stages with state-of-the-art accuracy.',
    image: '/assets/classification-model.png',
  },
  {
    id: 'integration',
    name: 'Data Integration',
    description: 'Unify diverse data sources into standardized formats for consistent analysis and model training.',
    image: '/assets/data-integration.png',
  }
];

export default function ModelShowcase() {
  const [activeModel, setActiveModel] = useState(models[0].id);
  
  const selectedModel = models.find(model => model.id === activeModel);

  return (
    <section className="py-16 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Model Capabilities
          </h2>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-600">
            Explore how HoneyBee's models transform oncology research
          </p>
        </div>

        <div className="lg:grid lg:grid-cols-12 lg:gap-8">
          {/* Model Selection */}
          <div className="lg:col-span-4">
            <div className="bg-white rounded-lg shadow-sm p-4 sticky top-4">
              <h3 className="text-lg font-medium text-gray-900 mb-6">
                Available Models
              </h3>
              <nav className="space-y-1" aria-label="Models">
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setActiveModel(model.id)}
                    className={`w-full group flex items-center px-3 py-4 text-sm font-medium rounded-md ${
                      activeModel === model.id
                        ? 'bg-amber-100 text-amber-700'
                        : 'text-gray-600 hover:bg-gray-50'
                    }`}
                  >
                    <span className="truncate">{model.name}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Model Details */}
          <div className="mt-10 lg:mt-0 lg:col-span-8">
            <div className="bg-white rounded-lg shadow-sm overflow-hidden">
              <div className="relative h-64 sm:h-72 md:h-80 lg:h-96 bg-gray-200">
                {selectedModel && (
                  <div className="w-full h-full flex items-center justify-center text-gray-500">
                    <div className="text-center px-4">
                      <p>Image visualization for {selectedModel.name}</p>
                      <p className="text-xs mt-2">Add actual model images to /public/assets/</p>
                    </div>
                  </div>
                )}
              </div>
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900">
                  {selectedModel?.name}
                </h3>
                <p className="mt-3 text-base text-gray-600">
                  {selectedModel?.description}
                </p>
                <div className="mt-6">
                  <a 
                    href="#" 
                    className="text-amber-600 hover:text-amber-800 font-medium"
                  >
                    Learn more about this model →
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}