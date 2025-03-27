import { useState } from 'react';
import { motion } from 'framer-motion';
import { Tab } from '@headlessui/react';
import clsx from 'clsx';

export default function DataLoadersSection() {
  const categories = {
    'Data Loaders': [
      { name: 'SVS', description: 'Load and process Aperio SVS whole slide images.' },
      { name: 'DICOM', description: 'Support for medical imaging DICOM standard files.' },
      { name: 'NIFTI', description: 'Neuroimaging Informatics Technology Initiative format support.' },
      { name: 'TIFF', description: 'Tagged Image File Format commonly used in scientific imaging.' },
      { name: 'PDF', description: 'Extract text and images from medical PDF documents.' },
      { name: 'MINDS', description: 'Integration with the MINDS dataset format.' },
    ],
    'Embedding Models': [
      { name: 'GatorTron', description: 'Medical text embedding using the GatorTron model.' },
      { name: 'BioBERT', description: 'Biomedical text representations with BioBERT.' },
      { name: 'REMEDIS', description: 'Medical domain-specific embeddings.' },
      { name: 'RadImageNet', description: 'Radiology image embeddings from foundation models.' },
      { name: 'SeNMo', description: 'Semantic neuroimaging model embeddings.' },
    ],
  };

  return (
    <section className="py-16 bg-slate-50 dark:bg-slate-950">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white sm:text-4xl">
            Supported Formats & Models
          </h2>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-slate-600 dark:text-slate-300">
            HoneyBee supports a wide range of medical data formats and embedding models
          </p>
        </div>

        <div className="w-full max-w-3xl mx-auto">
          <Tab.Group>
            <Tab.List className="flex space-x-4 rounded-xl bg-slate-200/60 dark:bg-slate-800/60 p-1">
              {Object.keys(categories).map((category) => (
                <Tab
                  key={category}
                  className={({ selected }) =>
                    clsx(
                      'w-full py-3 text-sm font-medium leading-5 rounded-lg',
                      'focus:outline-none focus:ring-2 ring-amber-500 ring-opacity-60',
                      selected
                        ? 'bg-white dark:bg-slate-700 shadow text-amber-600 dark:text-amber-500'
                        : 'text-slate-700 dark:text-slate-300 hover:bg-white/[0.12] hover:text-slate-900 dark:hover:text-white'
                    )
                  }
                >
                  {category}
                </Tab>
              ))}
            </Tab.List>
            <Tab.Panels className="mt-6">
              {Object.values(categories).map((items, idx) => (
                <Tab.Panel
                  key={idx}
                  className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-md border border-slate-200 dark:border-slate-700"
                >
                  <motion.ul
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-4"
                  >
                    {items.map((item) => (
                      <motion.li
                        key={item.name}
                        whileHover={{ x: 5 }}
                        className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors"
                      >
                        <h3 className="text-lg font-medium text-slate-900 dark:text-white">
                          {item.name}
                        </h3>
                        <p className="mt-1 text-slate-600 dark:text-slate-300">{item.description}</p>
                      </motion.li>
                    ))}
                  </motion.ul>
                </Tab.Panel>
              ))}
            </Tab.Panels>
          </Tab.Group>
        </div>
      </div>
    </section>
  );
}