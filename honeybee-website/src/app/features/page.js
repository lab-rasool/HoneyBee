'use client';

// Add this line below to force dynamic rendering
export const dynamic = 'force-dynamic';

import { useSearchParams } from 'next/navigation';
import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';

export default function FeaturesPage() {
  const searchParams = useSearchParams();
  const [paramValue, setParamValue] = useState(null);
  const [activeCategory, setActiveCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  useEffect(() => {
    // Handle search parameters client-side
    const value = searchParams.get('yourParam');
    setParamValue(value);

    const category = searchParams.get('category') || 'all';
    const query = searchParams.get('query') || '';
    
    setActiveCategory(category);
    setSearchQuery(query);
  }, [searchParams]);
  
  const features = [
    {
      id: 'dataloader',
      category: 'data',
      title: 'Advanced Data Loaders',
      description: 'Easily load and preprocess medical images from DICOM, NIfTI, and WSI formats.',
      icon: '📊'
    },
    {
      id: 'multimodal',
      category: 'analysis',
      title: 'Multimodal Analysis',
      description: 'Combine insights from radiology, pathology, and clinical data for comprehensive diagnostics.',
      icon: '🔬'
    },
    {
      id: 'ai',
      category: 'ai',
      title: 'State-of-the-art AI Models',
      description: 'Access cutting-edge deep learning architectures specialized for medical imaging.',
      icon: '🧠'
    },
    {
      id: 'visualization',
      category: 'visualization',
      title: '3D Visualization',
      description: 'Render complex medical images with interactive 3D visualization tools.',
      icon: '🔄'
    },
    {
      id: 'api',
      category: 'developer',
      title: 'Comprehensive API',
      description: 'Well-documented Python API for seamless integration into existing workflows.',
      icon: '⚙️'
    },
    {
      id: 'survival',
      category: 'analysis',
      title: 'Survival Analysis',
      description: 'Advanced statistical tools for patient outcome prediction and analysis.',
      icon: '📈'
    }
  ];

  const filteredFeatures = features.filter(feature => {
    const matchesCategory = activeCategory === 'all' || feature.category === activeCategory;
    const matchesSearch = feature.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          feature.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const handleCategoryChange = (category) => {
    // Client-side navigation without full reload
    const url = new URL(window.location);
    url.searchParams.set('category', category);
    window.history.pushState({}, '', url);
    setActiveCategory(category);
  };

  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    // Update URL without page reload
    const url = new URL(window.location);
    if (query) {
      url.searchParams.set('query', query);
    } else {
      url.searchParams.delete('query');
    }
    window.history.pushState({}, '', url);
  };

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold text-center mb-8">HoneyBee Features</h1>
      
      <div className="max-w-xl mx-auto mb-8">
        <input
          type="text"
          placeholder="Search features..."
          value={searchQuery}
          onChange={handleSearch}
          className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-yellow-400"
        />
      </div>
      
      <div className="flex flex-wrap justify-center gap-2 mb-10">
        {['all', 'data', 'analysis', 'ai', 'visualization', 'developer'].map(category => (
          <button
            key={category}
            onClick={() => handleCategoryChange(category)}
            className={`px-4 py-2 rounded-full ${
              activeCategory === category 
                ? 'bg-yellow-400 text-gray-900' 
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </button>
        ))}
      </div>
      
      {filteredFeatures.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredFeatures.map(feature => (
            <div 
              key={feature.id}
              className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow"
            >
              <div className="text-3xl mb-3">{feature.icon}</div>
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-gray-600 mb-4">{feature.description}</p>
              <Link 
                href={`/documentation#${feature.id}`}
                className="text-yellow-600 hover:text-yellow-800 font-medium"
              >
                Learn more →
              </Link>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-10">
          <p className="text-xl text-gray-500">No features found matching your criteria.</p>
          <button 
            onClick={() => {
              setActiveCategory('all');
              setSearchQuery('');
              const url = new URL(window.location);
              url.searchParams.delete('category');
              url.searchParams.delete('query');
              window.history.pushState({}, '', url);
            }}
            className="mt-4 px-4 py-2 bg-yellow-400 rounded-md hover:bg-yellow-500"
          >
            Reset filters
          </button>
        </div>
      )}
      
      <div className="mt-16 bg-gray-100 rounded-lg p-8 text-center">
        <h2 className="text-2xl font-bold mb-4">Want to see HoneyBee in action?</h2>
        <p className="mb-6">Check out our examples or schedule a demo with our team.</p>
        <div className="flex justify-center gap-4">
          <Link 
            href="/examples"
            className="px-6 py-2 bg-yellow-400 rounded-md hover:bg-yellow-500"
          >
            View Examples
          </Link>
          <Link 
            href="/contact"
            className="px-6 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900"
          >
            Request Demo
          </Link>
        </div>
      </div>
    </div>
  );
}