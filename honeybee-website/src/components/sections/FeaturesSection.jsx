import { motion } from 'framer-motion';
import Card from '../ui/Card';
import { 
  DocumentTextIcon, 
  ServerIcon, 
  CpuChipIcon, 
  CloudArrowUpIcon 
} from '@heroicons/react/24/outline';

export default function FeaturesSection() {
  const features = [
    {
      title: 'Medical Data Loading',
      description: 'Support for multiple medical data formats including SVS, DICOM, NIFTI, TIFF, PDF, and more.',
      icon: <DocumentTextIcon className="h-6 w-6" />
    },
    {
      title: 'Embedding Generation',
      description: 'Generate embeddings using foundation models like GatorTron, BioBERT, REMEDIS, RadImageNet, and SeNMo.',
      icon: <CpuChipIcon className="h-6 w-6" />
    },
    {
      title: 'Instruction Tuning',
      description: 'Create datasets for HuggingFace instruction tuning specifically tailored for oncology AI models.',
      icon: <ServerIcon className="h-6 w-6" />
    },
    {
      title: 'Advanced RAG Support',
      description: 'Retrieval augmented generation tools specialized for oncology data and medical research.',
      icon: <CloudArrowUpIcon className="h-6 w-6" />
    }
  ];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <section className="py-16 bg-white dark:bg-slate-900">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white sm:text-4xl">Key Features</h2>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-slate-600 dark:text-slate-300">
            HoneyBee provides a comprehensive set of tools for oncology AI research
          </p>
        </div>

        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
        >
          {features.map((feature, index) => (
            <motion.div key={index} variants={item}>
              <Card 
                title={feature.title}
                icon={feature.icon}
                className="h-full"
              >
                <p>{feature.description}</p>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}